import os, sys
import numpy as np
import warnings

import torch as tc
from torchvision.models.detection.roi_heads import box_ops
from typing import Optional, List, Dict, Tuple

from .pred_set import PredSet, PredSetReg
from .util import *

# ## ps for prp with iou considered
# class PredSetPrp(PredSet):
#     """
#     T \in [0, \infty]
#     """
#     def __init__(self, mdl, eps=0.0, delta=0.0, n=0):#, iou_thres=0.25): #TODO: iou_thres=0.5
#         super().__init__(mdl, eps, delta, n)
#         #self.iou_thres = iou_thres
#         raise NotImplementedError # proposal computation need to be done in feature space, but currently proposals are in image space

        
#     def forward(self, pred, targets=None, e=0.0):
#         proposals, obj_scores = pred['proposal'], pred['score']
#         device = proposals[0].device
        
#         if targets is not None:
#             ## for each groundtruth (gt) object, find a matching proposal and its score
#             scores_match = []
#             for tar, prp, objsc in zip(targets, proposals, obj_scores):
#                 ## compute iou between gt objects and proposals
#                 iou_tar_prp = box_ops.box_iou(tar, prp)
#                 if len(iou_tar_prp.shape) == 1:
#                     iou_tar_prp = iou_tar_prp.unsqueeze(0)
#                 iou_top, idx_top = iou_tar_prp.max(1)
#                 obj_scores_top = objsc[idx_top]
#                 scores_match.append(iou_top * obj_scores_top)                    
#             scores_match = tc.cat(scores_match)
            
#             return -scores_match # take negative due to the construction algorithm convention
#         else:
#             return proposals, obj_scores

        
#     def set(self, x):
#         with tc.no_grad():
#             ## for each proposal, find the largest bounding box such that objectness*IoU(proposal, bonding box) = T
#             prps, obj_scores = self.forward(x)
#             T = -self.T
#             prps_ret = []
#             for prp, objsc in zip(prps, obj_scores):
#                 w, h = prp[:, 2] - prp[:, 0], prp[:, 3] - prp[:, 1]
#                 iou_des = T / objsc
#                 i_update = iou_des <= 1.0
#                 # print('w', w[w<0].shape)
#                 # print('h', h[h<0].shape)
#                 # print('bb h', prp[h<0, :])
#                 # print('c', c[c<0].shape)
#                 # print()
#                 assert(all(w>=0) and all(h>=0) and all(iou_des>=0))
#                 dw, dh = (w - w*iou_des) / (1 + iou_des), (h - h*iou_des) / (1 + iou_des)

#                 assert(all(dw[i_update]>=0) and all(dh[i_update]>=0))
#                 prp_new = prp.clone()
#                 prp_new[i_update, 0] = prp_new[i_update, 0] - dw[i_update]
#                 prp_new[i_update, 2] = prp_new[i_update, 2] + dw[i_update]
#                 prp_new[i_update, 1] = prp_new[i_update, 1] - dh[i_update]
#                 prp_new[i_update, 3] = prp_new[i_update, 3] + dh[i_update]
#                 w, h = prp_new[:, 2] - prp_new[:, 0], prp_new[:, 3] - prp_new[:, 1]
#                 assert(all(w>=0) and all(h>=0))
#                 prps_ret.append(prp_new[i_update])

#         return prps_ret

#     def membership(self, x, y):
        
#         with tc.no_grad():
#             membership = self.forward(x, y) <= self.T
#             #print(membership.sum().item(), membership.shape[0], 'correctness =', (membership.sum().float() / membership.shape[0]).item())

#         return membership
    
#     # def membership(self, x, y):
#     #     device = x['proposal'][0].device
#     #     with tc.no_grad():
#     #         s = self.set(x)
            
#     #         membership = []
#     #         for tar, prp in zip(y, s):
#     #             if prp.shape[0] == 0:
#     #                 warnings.warn('vacuous proposals')
#     #                 membership.append(tc.tensor([False]*tar.shape[0], device=device))
#     #             else:
#     #                 ## check each target is included in at least one proposal
#     #                 f_inside = ((prp[:, 0].unsqueeze(0) <= tar[:, 0].unsqueeze(1)) &
#     #                             (prp[:, 1].unsqueeze(0) <= tar[:, 1].unsqueeze(1)) &
#     #                             (prp[:, 2].unsqueeze(0) >= tar[:, 2].unsqueeze(1)) &
#     #                             (prp[:, 3].unsqueeze(0) >= tar[:, 3].unsqueeze(1))).sum(1) >= 1
#     #                 membership.append(f_inside)

#     #         membership = tc.cat(membership)
#     #         print(membership.sum().item(), membership.shape[0], 'correctness =', (membership.sum().float() / membership.shape[0]).item())
#     #     return membership

    
#     def size(self, x):
#         with tc.no_grad():
#             sz = tc.cat([tc.tensor([s.shape[0]]) for s in self.set(x)])
#         return sz


class PredSetBox(PredSetReg):
    def __init__(self, mdl, eps=0.0, delta=0.0, n=0, var_min=1e-16):
        super().__init__(mdl=mdl, eps=eps, delta=delta, n=n, var_min=var_min)


    def size_volume(self, x, y=None):
        lb, ub = self.set_boxapprox(x)

        width = ub[:, 2] - lb[:, 0]
        height = ub[:, 3] - lb[:, 1]
        return width*height        


class PredSetPrp(PredSet):
    """
    T \in [0, \infty]
    """
    def __init__(self, mdl, eps=0.0, delta=0.0, n=0, iou_thres=0.25): #TODO: iou_thres=0.25/0.5
        super().__init__(mdl, eps, delta, n)
        self.iou_thres = iou_thres

        
    def forward(self, pred, targets=None, e=0.0):
        proposals, scores = pred['proposal'], pred['score']
        device = proposals[0].device
        
        if targets is not None:
            ## for each groundtruth (gt) object, find a matching proposal and its score
            scores_match = []
            for tar, prp in zip(targets, proposals):
                ## compute iou btwteen gt objects and proposals
                iou_tar_prp = box_ops.box_iou(tar, prp)
                if len(iou_tar_prp.shape) == 1:
                    iou_tar_prp = iou_tar_prp.unsqueeze(0)
                    
                ## for each object
                for iou_obj in iou_tar_prp:
                    if all(iou_obj < self.iou_thres):
                        # no matching proposal
                        scores_match.append(tc.rand(1, device=device)*1e-16)
                    else:
                        # choose the proposal with the smallest score among scores >= self.iou_thres
                        i_inval = iou_obj < self.iou_thres
                        s_min, i_min = (iou_obj * i_inval.float()*np.inf).min(0)
                        scores_match.append(tc.tensor([iou_obj[i_min]], device=device))
            if len(scores_match) == 0:
                # if no labels in the image, this happens
                scores_match = tc.tensor([], device=device)
            else:
                scores_match = tc.cat(scores_match)
            return -scores_match # take negative due to the construction algorithm convention
        else:
            return proposals, scores

        
    def set(self, x):
        with tc.no_grad():
            prps, scores = self.forward(x)
            s = [pr[sc >= (-self.T)] for pr, sc in zip(prps, scores)]
        return s

    
    def membership(self, x, y):
        device = x['proposal'][0].device
        with tc.no_grad():
            s = self.set(x)
            
            membership = []
            for tar, prp in zip(y, s):
                if prp.shape[0] == 0:
                    warnings.warn('vacuous proposals')
                    membership.append(tc.tensor([False]*tar.shape[0], device=device))
                elif tar.shape[0] == 0:
                    # no bounding boxes on the given image
                    continue
                else:
                    iou = box_ops.box_iou(tar, prp).max(1)[0]
                    membership.append(iou>=self.iou_thres)
            if len(membership) == 0:
                membership = tc.tensor(membership, device=device)
            else:
                membership = tc.cat(membership)
        return membership

    
    def size(self, x, y=None):
        with tc.no_grad():
            sz = tc.cat([tc.tensor([s.shape[0]]) for s in self.set(x)])
        return sz



class PredSetDet(PredSet):
    def __init__(self, mdl, mdl_prp_ps, mdl_obj_ps, mdl_loc_ps, target_labels=None):
        n = 0
        eps = mdl_prp_ps.eps + mdl_obj_ps.eps + mdl_loc_ps.eps
        delta = mdl_prp_ps.delta + mdl_obj_ps.delta + mdl_loc_ps.delta
        super().__init__(mdl, eps.item(), delta.item(), n)

        self.mdl.eval()
        self.mdl_prp_ps = mdl_prp_ps.eval()
        self.mdl_obj_ps = mdl_obj_ps.eval()
        self.mdl_loc_ps = mdl_loc_ps.eval()
        self.target_labels = tc.tensor(target_labels) if target_labels is not None else target_labels


    def _mdl_prp(self, images, targets):
        ## keep original image size
        original_image_sizes = tc.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        ## transform inputs
        images, targets = self.mdl.get_mdl().transform(images, targets)

        ## extract features
        features = self.mdl.get_mdl().backbone(images.tensors)
        if isinstance(features, tc.Tensor):
            features = OrderedDict([('0', features)])

        ## get proposals
        proposals, proposal_losses, scores = self.mdl.get_mdl().rpn(images, features, targets, return_score=True)

        return proposals, scores, features, targets, images.image_sizes, original_image_sizes

    
    def _mdl_head(self, features, proposals, scores, image_sizes, targets, image_sizes_ori):

        detections, detector_losses = self.mdl.get_mdl().roi_heads(features, proposals, scores, image_sizes, targets)
        detections = self.mdl.get_mdl().transform.postprocess(detections, image_sizes, image_sizes_ori)
        detections = self.mdl.get_mdl()._get_boxes_logvar(detections)
        return detections

    
    def forward(self, images, targets=None):
        images = [i for i in images]
        assert(targets is None) ##TODO: update
        
        if targets is not None:
            targets = [t for t in targets]
        
        ## get proposals
        proposals, scores, features, targets, image_sizes, image_sizes_ori = self._mdl_prp(images, targets)
        ## get prediction sets on proposals
        proposals = self.mdl_prp_ps.set({'proposal': proposals, 'score': scores})

        ## get detections: f_loc(b | x, r, c) and f_prs(x, r, c, b)
        pred_list = self._mdl_head(features, proposals, scores, image_sizes, targets, image_sizes_ori)
        det_per_img = [p['labels'].shape[0] for p in pred_list]
        scores = tc.cat([p['scores'] for p in pred_list])
        labels = tc.cat([p['labels'] for p in pred_list])
        assert(all(labels>0))
        mu = tc.cat([pred['boxes'] for pred in pred_list], 0)
        logvar = tc.cat([pred['boxes_logvar'] for pred in pred_list], 0)

        ## get prediction sets on location predictions: C_loc(x, r, c)
        pred = {'mu': mu, 'logvar': logvar}
        loc_lb, loc_ub = self.mdl_loc_ps.set_boxapprox(pred)
        loc_ps = tc.cat([loc_lb[:, :2], loc_ub[:, 2:]], 1)
 
        ## get prediction sets on class predictions
        ph = tc.stack([1-scores, scores], 1)
        pred = {'ph': ph, 'logph': ph.log(), 'yh_top': ph.argmax(-1), 'ph_top': ph.max(-1)[0]}
        obj_ps = self.mdl_obj_ps.set(pred)

        
        # scores = tc.cat([p['scores'] for p in pred_list])
        # labels = tc.cat([p['labels'] for p in pred_list])
        # assert(all(labels>0))
        
        # ##TODO: redundant
        # ph = tc.where(labels.unsqueeze(1)>0, tc.stack([1-scores, scores], 1), tc.stack([scores, 1-scores], 1))
        # pred = {'ph': ph, 'logph': ph.log(), 'yh_top': ph.argmax(-1), 'ph_top': ph.max(-1)[0]}
        # obj_ps = self.mdl_obj_ps.set(pred)

        ## return location and class if the corresponding object is presented, assuming the non-presented object are not useful in the downstream applications
        obj_ps = tc.split(obj_ps, det_per_img)
        loc_ps = tc.split(loc_ps, det_per_img)
        labels = tc.split(labels, det_per_img)
        obj_val = [o[:, 1] for o in obj_ps]
        loc_ps_val = [l[v] for l, v in zip(loc_ps, obj_val)]
        labels_val = [l[v] for l, v in zip(labels, obj_val)]
        
        ## filter labels
        if self.target_labels is not None:
            self.target_labels = self.target_labels.to(labels_val[0].device)
            i_choose_list = [tc.cat([tc.tensor([l in self.target_labels]) for l in labels_i]) for labels_i in labels_val]
            loc_ps_val = [l[i] for l, i in zip(loc_ps_val, i_choose_list)]
            labels_val = [l[i] for l, i in zip(labels_val, i_choose_list)]

        ## return
        return loc_ps_val, labels_val

        
    def set(self, x):
        with tc.no_grad():
            return self.forward(x)        

        
    def membership(self, x, y):
        x = [x_i for x_i in x]
        y = [y_i for y_i in y]
        device = x[0].device
        with tc.no_grad():
            loc_gt, label_gt = [y_i['boxes'] for y_i in y], [y_i['labels'] for y_i in y]
            if np.sum([l.shape[0] for l in label_gt]) == 0:
                # no labels
                return tc.zeros(0, device=device).bool()
                            
            # print('loc_gt:', loc_gt)
            # print('label_gt:', label_gt)
            # print()
            
            loc_ps, label_ps = self.set(x)
            
            # print('loc_ps:', loc_ps)
            # print('label_ps:', label_ps)
            # print()

            ## for each (loc_gt, label_gt), check whether loc_ps includes loc_gt and label_ps and label_gt are the same
            m = []
            for loc_gt_b, label_gt_b, loc_ps_b, label_ps_b in zip(loc_gt, label_gt, loc_ps, label_ps):
                assert(len(loc_gt_b) == len(label_gt_b))
                m_b = []
                for loc_gt_i, label_gt_i in zip(loc_gt_b, label_gt_b):
                    m_i = False
                    for loc_ps_i, label_ps_i in zip(loc_ps_b, label_ps_b):
                        if box_inclusion(loc_gt_i, loc_ps_i) and label_gt_i == label_ps_i:
                            m_i = True
                            break
                    m_b.append(m_i)
                m.append(tc.tensor(m_b, device=device))
            m = tc.cat(m).bool()
            #print(f'membership: {m.sum()} / {len(m)} = {m.float().mean()}')
            return m
        
            
    def size(self, x, y=None):
        x = [x_i for x_i in x]
        ## return dummy size
        return tc.zeros(1, device=x[0].device)

