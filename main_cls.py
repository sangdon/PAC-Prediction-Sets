import os, sys
import argparse
import torch as tc

import util
import data
import model
import learning
import uncertainty

def main(args):

    ## init datasets
    print("## init source datasets: %s"%(args.data.src))
    ds_src = getattr(data, args.data.src)(
        root=os.path.join('data', args.data.src.lower()),
        batch_size=args.data.batch_size,
        image_size=args.data.img_size[1],
        train_rnd=True, val_rnd=False, test_rnd=False, # val_rnd=False due to prediction set
        train_aug=args.data.aug_src is not None, val_aug=args.data.aug_src is not None, test_aug=args.data.aug_src is not None,
        aug_types=args.data.aug_src,
        color=True if args.data.img_size[0]==3 else False,
    )
    print("## init target datasets: %s"%(args.data.tar))
    ds_tar = getattr(data, args.data.tar)(
        root=os.path.join('data', args.data.tar.lower()),
        batch_size=args.data.batch_size,
        image_size=args.data.img_size[1],
        train_rnd=True, val_rnd=False, test_rnd=False, # val_rnd=False due to prediction set
        train_aug=args.data.aug_tar is not None, val_aug=args.data.aug_tar is not None, test_aug=args.data.aug_tar is not None,
        aug_types=args.data.aug_tar,
        color=True if args.data.img_size[0]==3 else False,
    )

    ##TODO: simplify code
    print("## init domain datasets: src = %s, tar = %s"%(args.data.src, args.data.tar))
    ds_src_dom = getattr(data, args.data.src)(
        root=os.path.join('data', args.data.src.lower()),
        batch_size=args.data.batch_size,
        image_size=args.data.img_size[1],
        train_rnd=True, val_rnd=False, test_rnd=False, # val_rnd=False due to prediction set
        train_aug=args.data.aug_src is not None, val_aug=args.data.aug_src is not None, test_aug=args.data.aug_src is not None,
        aug_types=args.data.aug_src,
        color=True if args.data.img_size[0]==3 else False,
        domain_label=1,
    )
    print("## init target datasets: %s"%(args.data.tar))
    ds_tar_dom = getattr(data, args.data.tar)(
        root=os.path.join('data', args.data.tar.lower()),
        batch_size=args.data.batch_size,
        image_size=args.data.img_size[1],
        train_rnd=True, val_rnd=False, test_rnd=False, # val_rnd=False due to prediction set
        train_aug=args.data.aug_tar is not None, val_aug=args.data.aug_tar is not None, test_aug=args.data.aug_tar is not None,
        aug_types=args.data.aug_tar,
        color=True if args.data.img_size[0]==3 else False,
        domain_label=0,
    )
    print()

    ## init a model
    print("## init models: %s"%(args.model.base))
    mdl = getattr(model, args.model.base)(n_labels=args.data.n_labels, pretrained=args.model.pretrained)
    print()
    
    ## learning
    l = learning.ClsLearner(mdl, args.train)
    if not args.model.pretrained:
        print("## train...")
        l.train(ds_src.train, ds_src.val, ds_tar.test)
    print("## test...")
    l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
    print()

    ## calibration
    if args.calibrate:
        mdl_cal = getattr(model, args.model.cal)(mdl)
        l = uncertainty.TempScalingLearner(mdl_cal, args.cal)
        print("## calibrate...")
        l.train(ds_src.val, ds_src.val, ds_tar.test)
        print("## test...")
        l.test(ds_tar.test, ld_name=args.data.src, verbose=True)
        print()

    
    ## uncertainty
    if args.estimate:
        if args.train_predset.method == 'pac_predset':
            mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
            l = learning.PredSetConstructor(mdl_predset, args.train_predset)
        else:
            raise NotImplementedError
        l.train(ds_src.val)
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)

    
def parse_args():
    ## init a parser
    parser = argparse.ArgumentParser(description='learning')

    ## meta args
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--train_cal_iw', action='store_true')
    parser.add_argument('--estimate', action='store_true')

    ## data args
    parser.add_argument('--data.batch_size', type=int, default=200)
    parser.add_argument('--data.src', type=str, required=True)
    parser.add_argument('--data.tar', type=str, required=True)
    parser.add_argument('--data.n_labels', type=int)
    parser.add_argument('--data.img_size', type=int, nargs=3)
    parser.add_argument('--data.aug_src', type=str, nargs='*')
    parser.add_argument('--data.aug_tar', type=str, nargs='*')

    ## model args
    parser.add_argument('--model.base', type=str)
    parser.add_argument('--model.path_pretrained', type=str)
    parser.add_argument('--model.cal', type=str, default='Temp')
    parser.add_argument('--model.sd', type=str, default='BigFNN')

    ## predset model args
    parser.add_argument('--model_predset.eps', type=float, default=0.01)
    parser.add_argument('--model_predset.alpha', type=float, default=0.01)
    parser.add_argument('--model_predset.delta', type=float, default=1e-5)
    parser.add_argument('--model_predset.n', type=int)

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

    ## calibration args
    parser.add_argument('--cal.rerun', action='store_true')
    parser.add_argument('--cal.load_final', action='store_true')
    parser.add_argument('--cal.optimizer', type=str, default='SGD')
    parser.add_argument('--cal.n_epochs', type=int, default=100)
    parser.add_argument('--cal.lr', type=float, default=0.01)
    parser.add_argument('--cal.momentum', type=float, default=0.9)
    parser.add_argument('--cal.weight_decay', type=float, default=0.0)
    parser.add_argument('--cal.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--cal.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--cal.val_period', type=int, default=1)    

    ## uncertainty estimation args
    parser.add_argument('--train_predset.method', type=str, default='pac_predset')
    parser.add_argument('--train_predset.rerun', action='store_true')
    parser.add_argument('--train_predset.load_final', action='store_true')
    parser.add_argument('--train_predset.binary_search', action='store_true')
    parser.add_argument('--train_predset.bnd_type', type=str, default='direct')
    
    
    args = parser.parse_args()
    args = util.to_tree_namespace(args)


    ##TODO: generalize
    ## additional args
    args.train.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args.train.exp_name = args.exp_name
    args.train.snapshot_root = args.snapshot_root

    args.train_predset.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args.train_predset.exp_name = args.exp_name
    args.train_predset.snapshot_root = args.snapshot_root

    args.cal.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args.cal.exp_name = args.exp_name
    args.cal.snapshot_root = args.snapshot_root


    ## dataset specific parameters
    if args.data.src == 'MNIST':
        if args.data.n_labels is None:
            args.data.n_labels = 10
        if args.data.img_size is None:
            args.data.img_size = (3, 32, 32)
        if args.model.base is None:
            args.model.base = 'ResNet18'
        if args.model.path_pretrained is None:
            args.model.pretrained = False
        if args.model_predset.n is None:
            args.model_predset.n = 10000
            
    elif args.data.src == 'ImageNet':
        if args.data.n_labels is None:
            args.data.n_labels = 1000
        if args.data.img_size is None:
            args.data.img_size = (3, 224, 224)
        if args.model.base is None:
            args.model.base = 'ResNet152'
        if args.model.path_pretrained is None:
            args.model.path_pretrained = 'pytorch'
        if args.model.path_pretrained  == 'pytorch':
            args.model.pretrained = True
        if args.model_predset.n is None:
            args.model_predset.n = 25000
    else:
        raise NotImplementedError    
        
    
    ## print args
    util.print_args(args)
    
    ## setup logger
    os.makedirs(os.path.join(args.snapshot_root, args.exp_name), exist_ok=True)
    sys.stdout = util.Logger(os.path.join(args.snapshot_root, args.exp_name, 'out'))
    
    
    return args    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)


