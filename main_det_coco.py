import os, sys
import argparse
import torch as tc
import torchvision as tv
import warnings
import numpy as np
import pickle
import types

import util
import data
import model
import learning
import uncertainty

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def main(args):

    ##---- init datasets
    print("##---- init source datasets: %s"%(args.data.src))
    ds_src = getattr(data, args.data.src)(
        root=os.path.join('data', args.data.src.lower()),
        batch_size=args.data.batch_size,
        train_rnd=True, val_rnd=False, test_rnd=False, # val_rnd=False due to prediction set
        train_aug=args.data.aug_src is not None, val_aug=args.data.aug_src is not None, test_aug=args.data.aug_src is not None,
        aug_types=args.data.aug_src,
        split_ratio={'train': None, 'val': 0.5, 'test': 0.5},
        target_labels=args.data.target_labels,
    )
    print("##---- init target datasets: %s"%(args.data.tar))
    assert(args.data.src == args.data.tar)
    ds_tar = ds_src
    print()

    
    ##---- init a model
    print("##---- init models: %s"%(args.model.base))
    mdl = getattr(model, args.model.base)(n_labels=args.data.n_labels,
                                          path_pretrained=args.model.path_pretrained,
                                          rpn_pre_nms_top_n_test=5000, #100000
                                          rpn_post_nms_top_n_test=5000, #100000
                                          rpn_score_thresh=0.0,
                                          rpn_nms_thresh=0.0,
                                          box_score_thresh=0.0,
    )
    print()

    
    ##---- learning
    print('##---- learn a detector')
    l = learning.DetLearner(mdl, args.train)
    print("## train...skip")
    # if not args.model.pretrained:
    #     l.train(ds_src.train, ds_src.val, ds_tar.test) ## dummy: only load a pre-trained model
    if args.run_test:
        print("## test...")
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
    print()

    ## --- filter unused labels
    mdl = model.FilterLabel(mdl, target_labels=args.data.target_labels) # only see selected labels
    
    ##---- prediction set estimation and/or calibration
    if args.estimate:
        assert(args.train_predset.method == 'pac_predset')

        ## count objects in validation set
        n = 0
        for x, y in ds_tar.val:
            n += np.sum([len(y_i['labels']) for y_i in y])
        print(f'# number of objects in the calibration set = {n}')
        args.model_predset.n = n
        
        ##---- proposal: eps=0.09
        if args.estimate_proposal:
            print('##---- prediction set for the proposal predictor')

            ## proposal data loader
            dsld_src_prp = model.DetectionDatasetLoader(model.ProposalView(mdl), ds_src, args.train_predset.device, ideal=False)
            dsld_tar_prp = model.DetectionDatasetLoader(model.ProposalView(mdl), ds_tar, args.train_predset.device, ideal=False)

            ## prediction set estimation
            mdl_prp_ps = model.PredSetPrp(model.Dummy()) #, args.model_predset_prp.eps, args.model_predset.delta_comp, args.model_predset.n) #TODO: simplify
            l = uncertainty.PredSetConstructor_BC(mdl_prp_ps, args.train_predset, name_postfix='prp')
            l.train(dsld_src_prp.val, types.SimpleNamespace(n=args.model_predset.n_prp, eps=args.model_predset_prp.eps, delta=args.model_predset.delta_comp, verbose=True, save=True))
            _, error_prp = l.test(dsld_tar_prp.test, ld_name=args.data.tar, verbose=True)
            print()
        
        
        ##---- objectness: eps=0.02
        if args.estimate_objectness:
            print('##---- prediction set for the class predictor')
            ## class data loader
            dsld_src_obj = model.DetectionDatasetLoader(model.ObjectnessView(mdl), ds_src, args.train_predset.device, ideal=True)
            dsld_tar_obj = model.DetectionDatasetLoader(model.ObjectnessView(mdl), ds_tar, args.train_predset.device, ideal=True)

            ## prediction set estimation
            mdl_obj_ps = model.PredSetCls(model.Dummy()) #, args.model_predset_obj.eps, args.model_predset.delta_comp, args.model_predset.n_obj) #TODO: simplify
            l = uncertainty.PredSetConstructor_BC(mdl_obj_ps, args.train_predset, name_postfix='obj')
            l.train(dsld_src_obj.val, types.SimpleNamespace(n=args.model_predset.n_obj, eps=args.model_predset_obj.eps, delta=args.model_predset.delta_comp, verbose=True, save=True))
            _, error_obj = l.test(dsld_tar_obj.test, ld_name=args.data.tar, verbose=True)
            print()

        
        ##---- location: eps=0.09
        if args.estimate_location:
            print('##---- prediction set for the location predictor')
            dsld_src_loc = model.DetectionDatasetLoader(model.LocationView(mdl), ds_src, args.train_predset.device, ideal=True)
            dsld_tar_loc = model.DetectionDatasetLoader(model.LocationView(mdl), ds_tar, args.train_predset.device, ideal=True)

            ## prediction set estimation
            mdl_loc_ps = model.PredSetBox(model.Dummy()) #, args.model_predset_loc.eps, args.model_predset.delta_comp, args.model_predset.n_loc)
            l = uncertainty.PredSetConstructor_BC(mdl_loc_ps, args.train_predset, name_postfix='loc')
            l.train(dsld_src_loc.val, types.SimpleNamespace(n=args.model_predset.n_loc, eps=args.model_predset_loc.eps, delta=args.model_predset.delta_comp, verbose=True, save=True))
            _, error_loc = l.test(dsld_tar_loc.test, ld_name=args.data.tar, verbose=True)
            print()
            
        ##---- evaluate a composed model
        print('##---- compose all prediction sets')
        mdl_prp_ps = model.PredSetPrp(model.Dummy()) #, args.model_predset_prp.eps, args.model_predset.delta_comp, args.model_predset.n_prp)
        l = uncertainty.PredSetConstructor_BC(mdl_prp_ps, args.train_predset, name_postfix='prp')
        l.train(None, types.SimpleNamespace(n=args.model_predset.n_prp, eps=args.model_predset_prp.eps, delta=args.model_predset.delta_comp, verbose=True, save=True)) # load a model
        mdl_obj_ps = model.PredSetCls(model.Dummy()) #, args.model_predset_obj.eps, args.model_predset.delta_comp, args.model_predset.n_obj)
        l = uncertainty.PredSetConstructor_BC(mdl_obj_ps, args.train_predset, name_postfix='obj')
        l.train(None, types.SimpleNamespace(n=args.model_predset.n_obj, eps=args.model_predset_obj.eps, delta=args.model_predset.delta_comp, verbose=True, save=True)) # load a model
        mdl_loc_ps = model.PredSetReg(model.Dummy()) #, args.model_predset_loc.eps, args.model_predset.delta_comp, args.model_predset.n_loc)
        l = uncertainty.PredSetConstructor_BC(mdl_loc_ps, args.train_predset, name_postfix='loc')
        l.train(None, types.SimpleNamespace(n=args.model_predset.n_loc, eps=args.model_predset_loc.eps, delta=args.model_predset.delta_comp, verbose=True, save=True)) # load a model
            
        mdl_det_ps = model.PredSetDet(mdl, mdl_prp_ps, mdl_obj_ps, mdl_loc_ps, args.data.target_labels)
        l = uncertainty.PredSetConstructor_BC(mdl_det_ps, args.train_predset, name_postfix='det')
        _, error_det = l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)

        ## save error
        if args.estimate_proposal and args.estimate_objectness and args.estimate_location:
            pickle.dump({'error_prp': error_prp.detach().cpu().numpy(),
                         'error_obj': error_obj.detach().cpu().numpy(),
                         'error_loc': error_loc.detach().cpu().numpy(),
                         'error_det': error_det.detach().cpu().numpy(),
                         'eps_prp': args.model_predset_prp.eps, 'eps_obj': args.model_predset_obj.eps, 'eps_loc': args.model_predset_loc.eps,
                         'eps': args.model_predset.eps,
                         'delta': args.model_predset.delta,
                    }, open(os.path.join(args.snapshot_root, args.exp_name, 'det_error_stats.pk'), 'wb'))
            #plot_error_det([error_prp, error_obj, error_loc, error_det],
                       
    
    ## visualize results
    n_vis = np.inf    
    i = 1

    mdl_prp_ps = model.PredSetPrp(model.Dummy()) #, args.model_predset_prp.eps, args.model_predset.delta_comp, args.model_predset.n_prp)
    l = uncertainty.PredSetConstructor_BC(mdl_prp_ps, args.train_predset, name_postfix='prp')
    l.train(None, types.SimpleNamespace(n=args.model_predset.n_prp, eps=args.model_predset_prp.eps, delta=args.model_predset.delta_comp, verbose=True, save=True)) # load a model
    mdl_obj_ps = model.PredSetCls(model.Dummy()) #, args.model_predset_obj.eps, args.model_predset.delta_comp, args.model_predset.n_obj)
    l = uncertainty.PredSetConstructor_BC(mdl_obj_ps, args.train_predset, name_postfix='obj')
    l.train(None, types.SimpleNamespace(n=args.model_predset.n_obj, eps=args.model_predset_obj.eps, delta=args.model_predset.delta_comp, verbose=True, save=True)) # load a model
    mdl_loc_ps = model.PredSetReg(model.Dummy()) #, args.model_predset_loc.eps, args.model_predset.delta_comp, args.model_predset.n_loc)
    l = uncertainty.PredSetConstructor_BC(mdl_loc_ps, args.train_predset, name_postfix='loc')
    l.train(None, types.SimpleNamespace(n=args.model_predset.n_loc, eps=args.model_predset_loc.eps, delta=args.model_predset.delta_comp, verbose=True, save=True)) # load a model

    mdl_det_ps = model.PredSetDet(mdl, mdl_prp_ps, mdl_obj_ps, mdl_loc_ps, args.data.target_labels)

    results = []
    for images, targets in ds_tar.test:
        images = [img for img in learning.to_device(images, args.device)]
        with tc.no_grad():
            # prediction set results
            locs_ps, labels_ps = mdl_det_ps(images)

            # original detection prediction
            pred = mdl(images)

            locs_pred_all = [p['boxes'] for p in pred]
            scores_pred_all = [p['scores'] for p in pred]
            labels_pred_all = [p['labels'] for p in pred]
            locs_pred = [l[s>=0.5] for l, s in zip(locs_pred_all, scores_pred_all)]
            labels_pred = [l[s>=0.5] for l, s in zip(labels_pred_all, scores_pred_all)]

            # ground truth
            locs_gt = [t['boxes'] for t in targets]
            labels_gt = [t['labels'] for t in targets]

        for image, loc, label, loc_gt, label_gt, loc_pred, label_pred, loc_pred_all, label_pred_all, score_pred_all in \
            zip(images, locs_ps, labels_ps, locs_gt, labels_gt, locs_pred, labels_pred, locs_pred_all, labels_pred_all, scores_pred_all):
            image, loc, loc_pred, loc_gt = image.cpu(), loc.cpu(), loc_pred.cpu(), loc_gt.cpu()
            label, label_pred, loc_gt = label.cpu(), label_pred.cpu(), loc_gt.cpu()
            loc_pred_all, label_pred_all, score_pred_all = loc_pred_all.cpu(), label_pred_all.cpu(), score_pred_all.cpu()

            # result summary: all boxes are in (xmin, ymin, xmax, ymax) format
            # target classes are "person" and "car"
            result = {
                'image': image,
                'bbox_ps': loc, # the postprocessed location prediction set: the tightest bounding box that contains all bounding boxes in the prediction set
                'label_ps': label, # the postprocessed label prediction set: the object label when the label prediction set contains the "valid" label
                'bbox_det': loc_pred, # bounding boxes from the object detector when score >= 0.5 
                'label_det': label_pred, # labels of the bounding boxes when score >= 0.5
                'bbox_det_raw': loc_pred_all, # bounding boxes from the object detector
                'score_det_raw': score_pred_all, # scores of the bounding boxes
                'label_det_raw': label_pred_all, # labels of the bounding boxes
                'bbox_gt': loc_gt,
                'label_gt': label_gt,
            }
            results.append(result)
            for k, v in result.items():
                print(f'{k} =', v, v.shape)
            print()

        #     # plot results and save
        #     colors = ['green']*len(loc) + ['white']*len(loc_gt) + ['red']*len(loc_pred)
        #     loc_plot = tc.cat([loc, loc_gt, loc_pred], 0)
        #     label_plot = [' '+COCO_INSTANCE_CATEGORY_NAMES[l] for l in tc.cat([label, label_gt, label_pred])]

        #     image_boxes = tv.utils.draw_bounding_boxes(
        #         (image*255).byte(),
        #         loc_plot,
        #         label_plot,
        #         colors=colors,
        #         width=3,
        #     )
        #     image_boxes = (image_boxes/255.0).float()
        #     fn = os.path.join(args.snapshot_root, args.exp_name, 'figs', 'predset', f'{i}.png')
        #     print(fn)
        #     os.makedirs(os.path.dirname(fn), exist_ok=True)
        #     tv.utils.save_image(image_boxes, fn)
        #     i += 1
        # if i > n_vis:
        #     break
    
    pickle.dump(results, open('results_coco_test.pk', 'wb'))

                
def parse_args():
    ## init a parser
    parser = argparse.ArgumentParser(description='learning')

    ## meta args
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--estimate', action='store_true')
    parser.add_argument('--estimate_proposal', action='store_true')
    parser.add_argument('--estimate_objectness', action='store_true')
    parser.add_argument('--estimate_location', action='store_true')

    ## data args
    parser.add_argument('--data.batch_size', type=int, default=3)
    parser.add_argument('--data.src', type=str, default='COCO')
    parser.add_argument('--data.tar', type=str, default='COCO')
    parser.add_argument('--data.n_labels', type=int, default=91) # 90 + background
    #parser.add_argument('--data.img_size', type=int, nargs=3)
    parser.add_argument('--data.aug_src', type=str, nargs='*')
    parser.add_argument('--data.aug_tar', type=str, nargs='*')
    parser.add_argument('--data.target_labels', type=lambda v: None if v=='None' else [int(v_i) for v_i in v], nargs='*', default=[1, 3])

    ## model args
    parser.add_argument('--model.base', type=str, default='ProbFasterRCNN_resnet50_fpn')
    parser.add_argument('--model.path_pretrained', type=str)
    # parser.add_argument('--model.cal', type=str, default='Temp')
    # parser.add_argument('--model.sd', type=str, default='BigFNN')

    ## predset model args
    parser.add_argument('--model_predset.eps', type=float, default=0.2) # we use relaxed eps. 0.2 = 0.06*2 + 0.01 + 0.07
    parser.add_argument('--model_predset.delta', type=float, default=1e-5)
    parser.add_argument('--model_predset_prp.eps', type=float, default=0.06) #0.09
    parser.add_argument('--model_predset_obj.eps', type=float, default=0.01) #0.02
    parser.add_argument('--model_predset_loc.eps', type=float, default=0.07) #0.09
    #parser.add_argument('--model_predset.alpha', type=float, default=0.1) # we use relaxed alpha
    parser.add_argument('--model_predset.delta_comp', type=float, default=0.25*1e-5) #1e-5
    parser.add_argument('--model_predset.n_prp', type=int, default=6565)
    parser.add_argument('--model_predset.n_obj', type=int, default=6019)
    parser.add_argument('--model_predset.n_loc', type=int, default=6019)



    ## train args
    parser.add_argument('--train.rerun', action='store_true')
    parser.add_argument('--train.load_final', action='store_true')
    parser.add_argument('--train.optimizer', type=str, default='SGD')
    parser.add_argument('--train.n_epochs', type=int, default=100)
    parser.add_argument('--train.lr', type=float, default=0.01)
    parser.add_argument('--train.momentum', type=float, default=0.9)
    parser.add_argument('--train.weight_decay', type=float, default=0.0)
    parser.add_argument('--train.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--train.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--train.val_period', type=int, default=1)

    # ## calibration args
    # parser.add_argument('--cal.rerun', action='store_true')
    # parser.add_argument('--cal.load_final', action='store_true')
    # parser.add_argument('--cal.optimizer', type=str, default='SGD')
    # parser.add_argument('--cal.n_epochs', type=int, default=100)
    # parser.add_argument('--cal.lr', type=float, default=0.01)
    # parser.add_argument('--cal.momentum', type=float, default=0.9)
    # parser.add_argument('--cal.weight_decay', type=float, default=0.0)
    # parser.add_argument('--cal.lr_decay_epoch', type=int, default=20)
    # parser.add_argument('--cal.lr_decay_rate', type=float, default=0.5)
    # parser.add_argument('--cal.val_period', type=int, default=1)    

    # ## calibration args
    # parser.add_argument('--cal_reg.rerun', action='store_true')
    # parser.add_argument('--cal_reg.load_final', action='store_true')
    # parser.add_argument('--cal_reg.optimizer', type=str, default='SGD')
    # parser.add_argument('--cal_reg.n_epochs', type=int, default=100)
    # parser.add_argument('--cal_reg.lr', type=float, default=0.00001) ##TODO: too small
    # parser.add_argument('--cal_reg.momentum', type=float, default=0.9)
    # parser.add_argument('--cal_reg.weight_decay', type=float, default=0.0)
    # parser.add_argument('--cal_reg.lr_decay_epoch', type=int, default=20)
    # parser.add_argument('--cal_reg.lr_decay_rate', type=float, default=0.5)
    # parser.add_argument('--cal_reg.val_period', type=int, default=1)    
    # parser.add_argument('--cal_reg.normalizer', type=float, default=1.0)
 
    # ## train args for a source discriminator
    # parser.add_argument('--train_sd.rerun', action='store_true')
    # parser.add_argument('--train_sd.load_final', action='store_true')
    # parser.add_argument('--train_sd.optimizer', type=str, default='SGD')
    # parser.add_argument('--train_sd.n_epochs', type=int, default=10) ##TODO: 100
    # parser.add_argument('--train_sd.lr', type=float, default=0.01)
    # parser.add_argument('--train_sd.momentum', type=float, default=0.9)
    # parser.add_argument('--train_sd.weight_decay', type=float, default=0.0)
    # parser.add_argument('--train_sd.lr_decay_epoch', type=int, default=2) ##TODO: 20
    # parser.add_argument('--train_sd.lr_decay_rate', type=float, default=0.5)
    # parser.add_argument('--train_sd.val_period', type=int, default=1)

    # ## calibration args for a source discriminator
    # parser.add_argument('--cal_sd.rerun', action='store_true')
    # parser.add_argument('--cal_sd.load_final', action='store_true')
    # parser.add_argument('--cal_sd.optimizer', type=str, default='SGD')
    # parser.add_argument('--cal_sd.n_epochs', type=int, default=100) 
    # parser.add_argument('--cal_sd.lr', type=float, default=0.01)
    # parser.add_argument('--cal_sd.momentum', type=float, default=0.9)
    # parser.add_argument('--cal_sd.weight_decay', type=float, default=0.0)
    # parser.add_argument('--cal_sd.lr_decay_epoch', type=int, default=20)
    # parser.add_argument('--cal_sd.lr_decay_rate', type=float, default=0.5)
    # parser.add_argument('--cal_sd.val_period', type=int, default=1)    

    ## uncertainty estimation args
    parser.add_argument('--train_predset.method', type=str, default='pac_predset')
    parser.add_argument('--train_predset.rerun', action='store_true')
    parser.add_argument('--train_predset.load_final', action='store_true')
    parser.add_argument('--train_predset.binary_search', action='store_true')
    parser.add_argument('--train_predset.bnd_type', type=str, default='direct')
    
    
    args = parser.parse_args()
    args = util.to_tree_namespace(args)
    args.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args = util.propagate_args(args, 'device')
    args = util.propagate_args(args, 'exp_name')
    args = util.propagate_args(args, 'snapshot_root')

    
    ## print args
    util.print_args(args)
    
    ## setup logger
    os.makedirs(os.path.join(args.snapshot_root, args.exp_name), exist_ok=True)
    sys.stdout = util.Logger(os.path.join(args.snapshot_root, args.exp_name, 'out'))
    
    
    return args    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)


