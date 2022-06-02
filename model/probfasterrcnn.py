import os, sys
import warnings

from .util import *

import torchvision
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.detection.roi_heads import *
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator, MultiScaleRoIAlign, RPNHead, TwoMLPHead, GeneralizedRCNNTransform, load_state_dict_from_url, model_urls
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes



def get_conservative_box(mu, logvar):
    """
    box convention: xyxy
    """
    n = mu.shape[0]
    mu, logvar = mu.reshape(n, -1, 4), logvar.reshape(n, -1, 4)
    # one sigmal box
    sigma = logvar.exp().sqrt()
    xy_cons_topleft = mu[:, :, :2] - sigma[:, :, :2]
    xy_cons_botright = mu[:, :, 2:] + sigma[:, :, 2:]
    box_cons = tc.cat([xy_cons_topleft, xy_cons_botright], 2)
    box_cons = box_cons.reshape(n, -1)

    return box_cons



##
## rpn
##
from torchvision.models.detection.rpn import *


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self,
                 anchor_generator,
                 head,
                 #
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 #
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction='sum',
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,       # type: ImageList
                features,     # type: Dict[str, Tensor]
                targets=None,  # type: Optional[List[Dict[str, Tensor]]]
                return_score=False,
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        if return_score:
            return boxes, losses, scores
        else:
            return boxes, losses

    
##
## RCNN
##
class CustomGeneralizedRCNN(GeneralizedRCNN):
    def __init__(self, backbone, rpn, roi_heads, transform):
        super().__init__(backbone, rpn, roi_heads, transform)

        
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses, scores = self.rpn(images, features, targets, return_score=True)
        detections, detector_losses = self.roi_heads(features, proposals, scores, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)


class ProbRoIHeads(RoIHeads):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes,     # type: List[Tuple[int, int]]
                               labels_gt, ##NEW
                               boxes_gt, ##NEW
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        
        pred_boxes = box_regression['bbox_deltas'] ##NEW
        pred_boxes_cons = get_conservative_box(box_regression['bbox_deltas'], box_regression['bbox_deltas_logvar']) ##NEW
        
        if boxes_gt is None:
            ## generate dummies
            gt_boxes = tc.zeros_like(pred_boxes) ##NEW
        else:
            gt_boxes = tc.cat([b.unsqueeze(1).expand(-1, num_classes, -1) for b in boxes_gt], 0).view(pred_boxes.shape[0], -1)

        pred_boxes_in_feat, pred_boxes_cons_in_feat, gt_boxes_in_feat = pred_boxes, pred_boxes_cons, gt_boxes ##NEW
        pred_boxes = self.box_coder.decode(pred_boxes, proposals) ##NEW
        pred_boxes_cons = self.box_coder.decode(pred_boxes_cons, proposals) ##NEW
        gt_boxes = self.box_coder.decode(gt_boxes, proposals) ##NEW

        pred_boxes_in_feat = pred_boxes_in_feat.reshape(pred_boxes_in_feat.shape[0], -1, 4) ##NEW
        pred_boxes_cons_in_feat = pred_boxes_cons_in_feat.reshape(pred_boxes_cons_in_feat.shape[0], -1, 4) ##NEW
        gt_boxes_in_feat = gt_boxes_in_feat.reshape(gt_boxes_in_feat.shape[0], -1, 4) ##NEW
        
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_boxes_cons_list = pred_boxes_cons.split(boxes_per_image, 0) ##NEW
        gt_boxes_list = gt_boxes.split(boxes_per_image, 0) ##NEW
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_boxes_in_feat_list = pred_boxes_in_feat.split(boxes_per_image, 0) ##NEW
        pred_boxes_cons_in_feat_list = pred_boxes_cons_in_feat.split(boxes_per_image, 0) ##NEW
        gt_boxes_in_feat_list = gt_boxes_in_feat.split(boxes_per_image, 0) ##NEW
        

        ##NEW
        if labels_gt is None:
            labels_gt_list = [tc.zeros(s.shape[0], dtype=tc.long) for s in pred_scores_list]
        else:
            labels_gt_list = labels_gt
            
        ## duplicate gt labels, like one hot vector
        labels_gt_list = [F.one_hot(l, s.shape[-1]).mul(l.unsqueeze(1)) for l, s in zip(labels_gt_list, pred_scores_list)]
        
        all_boxes = []
        all_boxes_cons = [] ##NEW
        all_scores = []
        all_labels = []
        all_labels_gt = [] ##NEW
        all_boxes_gt = [] ##NEW
        all_boxes_in_feat, all_boxes_cons_in_feat, all_boxes_gt_in_feat = [], [], []
        for boxes, boxes_cons, scores, image_shape, labels_gt, boxes_gt, boxes_in_feat, boxes_cons_in_feat, gt_boxes_in_feat in zip(
                pred_boxes_list, pred_boxes_cons_list, pred_scores_list, image_shapes, labels_gt_list, gt_boxes_list,
                pred_boxes_in_feat_list, pred_boxes_cons_in_feat_list, gt_boxes_in_feat_list
        ): ##NEW

            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            boxes_cons = box_ops.clip_boxes_to_image(boxes_cons, image_shape) ##NEW
            boxes_gt = box_ops.clip_boxes_to_image(boxes_gt, image_shape) ##NEW
            ##no clip for boxes in feature space

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            boxes_cons = boxes_cons[:, 1:] ##NEW
            boxes_gt = boxes_gt[:, 1:] ##NEW
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            labels_gt = labels_gt[:, 1:] ##NEW
            boxes_in_feat = boxes_in_feat[:, 1:] ##NEW
            boxes_cons_in_feat = boxes_cons_in_feat[:, 1:] ##NEW
            gt_boxes_in_feat = gt_boxes_in_feat[:, 1:] ##NEW
            
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            boxes_cons = boxes_cons.reshape(-1, 4) ##NEW
            boxes_gt = boxes_gt.reshape(-1, 4) ##NEW
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            labels_gt = labels_gt.reshape(-1) ##NEW
            boxes_in_feat = boxes_in_feat.reshape(-1, 4)##NEW
            boxes_cons_in_feat = boxes_cons_in_feat.reshape(-1, 4)##NEW
            gt_boxes_in_feat = gt_boxes_in_feat.reshape(-1, 4)##NEW
            
            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, boxes_cons, boxes_gt, scores, labels, labels_gt = boxes[inds], boxes_cons[inds], boxes_gt[inds], scores[inds], labels[inds], labels_gt[inds] ##NEW
            boxes_in_feat, boxes_cons_in_feat, gt_boxes_in_feat = boxes_in_feat[inds], boxes_cons_in_feat[inds], gt_boxes_in_feat[inds] ##NEW

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, boxes_cons, boxes_gt, scores, labels, labels_gt = boxes[keep], boxes_cons[keep], boxes_gt[keep], scores[keep], labels[keep], labels_gt[keep] ##NEW
            boxes_in_feat, boxes_cons_in_feat, gt_boxes_in_feat = boxes_in_feat[keep], boxes_cons_in_feat[keep], gt_boxes_in_feat[keep] ##NEW

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, boxes_cons, boxes_gt, scores, labels, labels_gt = boxes[keep], boxes_cons[keep], boxes_gt[keep], scores[keep], labels[keep], labels_gt[keep] ##NEW
            boxes_in_feat, boxes_cons_in_feat, gt_boxes_in_feat = boxes_in_feat[keep], boxes_cons_in_feat[keep], gt_boxes_in_feat[keep] ##NEW

            all_boxes.append(boxes)
            all_boxes_cons.append(boxes_cons) ##NEW
            all_scores.append(scores)
            all_labels.append(labels)
            all_labels_gt.append(labels_gt)
            all_boxes_gt.append(boxes_gt)

            all_boxes_in_feat.append(boxes_in_feat) ##NEW
            all_boxes_cons_in_feat.append(boxes_cons_in_feat) ##NEW
            all_boxes_gt_in_feat.append(gt_boxes_in_feat) ##NEW


        return all_boxes, all_boxes_cons, all_scores, all_labels, all_labels_gt, all_boxes_gt, all_boxes_in_feat, all_boxes_cons_in_feat, all_boxes_gt_in_feat ##NEW

    
    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                proposal_scores,
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            if targets is not None:
                proposals, matched_idxs, labels_gt, regression_targets = self.select_training_samples(proposals, targets) ##NEW
            else:
                labels = None
                regression_targets = None
                matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            ##NEW
            if targets is None:
                boxes, boxes_cons, scores, labels, _, _, boxes_in_feat, boxes_cons_in_feat, _ = self.postprocess_detections(
                    class_logits, box_regression, proposals, image_shapes, None, regression_targets) ##NEW
                
            else:
                boxes, boxes_cons, scores, labels, labels_gt, boxes_gt, boxes_in_feat, boxes_cons_in_feat, boxes_gt_in_feat = self.postprocess_detections(
                    class_logits, box_regression, proposals, image_shapes, labels_gt, regression_targets) ##NEW
            num_images = len(boxes)
            for i in range(num_images):
                if targets is None:
                    result.append(
                        {
                            "boxes": boxes[i],
                            "boxes_cons": boxes_cons[i], ##NEW
                            "labels": labels[i],
                            "scores": scores[i],
                            "boxes_gt": tc.zeros_like(boxes[i]), ##NEW
                            "labels_gt": tc.zeros_like(labels[i]), ##NEW
                            'proposals': proposals[i], ##NEW
                            'proposal_scores': proposal_scores[i], ##NEW
                            ## return boxes in feature space
                            "boxes_in_feat": boxes_in_feat[i], ##NEW
                            "boxes_cons_in_feat": boxes_cons_in_feat[i], ##NEW 
                            "boxes_gt_in_feat": tc.zeros_like(boxes[i]), ##NEW
                            'proposals_in_feat': proposals[i], ##NEW                            
                        }
                    )
                else:
                    result.append(
                        {
                            "boxes": boxes[i],
                            "boxes_cons": boxes_cons[i], ##NEW
                            "boxes_gt": boxes_gt[i], ##NEW
                            "labels": labels[i],
                            "labels_gt": labels_gt[i], ##NEW
                            "scores": scores[i],
                            'proposals': proposals[i], ##NEW
                            'proposal_scores': proposal_scores[i], ##NEW
                            ## return boxes in feature space
                            "boxes_in_feat": boxes_in_feat[i], ##NEW
                            "boxes_cons_in_feat": boxes_cons_in_feat[i], ##NEW 
                            "boxes_gt_in_feat": boxes_gt_in_feat[i], ##NEW
                            'proposals_in_feat': proposals[i], ##NEW
                        }
                    )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses

    

class ProbFastRCNNPredictor(FastRCNNPredictor):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
        self.bbox_pred_logvar = nn.Linear(in_channels, num_classes * 4)
        

    def forward(self, x):
        scores, bbox_deltas = super().forward(x)
        bbox_delta_logvar = self.bbox_pred_logvar(x)
        return scores, {'bbox_deltas': bbox_deltas, 'bbox_deltas_logvar': bbox_delta_logvar}

    
class GeneralizedProbRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__(min_size, max_size, image_mean, image_std)

        
    def postprocess(self,
                    result,               # type: List[Dict[str, Tensor]]
                    image_shapes,         # type: List[Tuple[int, int]]
                    original_image_sizes  # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes, boxes_cons, boxes_gt, proposals = pred["boxes"], pred['boxes_cons'], pred['boxes_gt'], pred['proposals'] ##NEW
            boxes  = resize_boxes(boxes, im_s, o_im_s) ##NEW
            boxes_cons = resize_boxes(boxes_cons, im_s, o_im_s) ##NEW
            boxes_gt = resize_boxes(boxes_gt, im_s, o_im_s) ##NEW
            proposals = resize_boxes(proposals, im_s, o_im_s) ##NEW

            result[i]["boxes"] = boxes
            result[i]['boxes_cons'] = boxes_cons ##NEW
            result[i]['boxes_gt'] = boxes_gt ##NEW
            result[i]['proposals'] = proposals ##NEW
            
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result
        

class ProbFasterRCNN(CustomGeneralizedRCNN):
    """
    Implements Faster R-CNN.
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
    Example::
        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FasterRCNN
        >>> from torchvision.models.detection.rpn import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # FasterRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> # put the pieces together inside a FasterRCNN model
        >>> model = FasterRCNN(backbone,
        >>>                    num_classes=2,
        >>>                    rpn_anchor_generator=anchor_generator,
        >>>                    box_roi_pool=roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = ProbFastRCNNPredictor( ##NEW
                representation_size,
                num_classes)

        roi_heads = ProbRoIHeads( ##NEW
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedProbRCNNTransform(min_size, max_size, image_mean, image_std)

        super(ProbFasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

    def _get_boxes_logvar(self, out, var_min=1.0, var_max=np.inf, var_feat_min=1e-1, var_feat_max=np.inf):
        out_new = []
        ## agument logvar
        for pred in out:
            ## boxes in image space
            boxes, boxes_sigma = pred['boxes'], pred['boxes_cons']
            sigma = (boxes - boxes_sigma).abs()
            sigma_topleft, sigma_botright = sigma[:, :2], sigma[:, 2:]
            sigma_worst = tc.max(sigma[:, :2], sigma[:, 2:])
            sigma = tc.cat((sigma_worst, sigma_worst), 1)
            var = sigma.pow(2)
            var = tc.maximum(tc.tensor(var_min, device=var.device), var)
            var = tc.minimum(tc.tensor(var_max, device=var.device), var)
            pred['boxes_logvar'] = var.log()

            ## boxes in feature space
            boxes, boxes_sigma = pred['boxes_in_feat'], pred['boxes_cons_in_feat']
            var = (boxes - boxes_sigma).abs().pow(2)
            var = tc.maximum(tc.tensor(var_feat_min, device=var.device), var)
            var = tc.minimum(tc.tensor(var_feat_max, device=var.device), var)
            pred['boxes_logvar_in_feat'] = var.log()

            out_new.append(pred)
        return out
        
        
    def forward(self, images, targets=None):
        out = super().forward(images, targets)

        if not self.training:
            out_new = self._get_boxes_logvar(out)
        else:
            out_new = out
                                
        return out_new


def probfasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
    Example::
        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        num_classes (int): number of output classes of the model (including the background)
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = ProbFasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def ProbFasterRCNN_resnet50_fpn(
        #pretrained=False,
        path_pretrained=None,
        progress=True,
        n_labels=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    pretrained = True if path_pretrained else False
    
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = ProbFasterRCNN(backbone, n_labels, **kwargs)

    if path_pretrained:
        model.load_state_dict(torch.load(path_pretrained))
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict, strict=False)
    return model


class DetectionDatasetLoader:
    def __init__(self, mdl_viewer, dsld, device, ideal):
        self.train = DetectionDataLoader(mdl_viewer, dsld.train, device, ideal)
        self.val = DetectionDataLoader(mdl_viewer, dsld.val, device, ideal)
        self.test = DetectionDataLoader(mdl_viewer, dsld.test, device, ideal)
        

        
class DetectionDataLoader:
    def __init__(self, mdl_viewer, ld, device, ideal):
        self.mdl_viewer = mdl_viewer.eval() # always in eval mode
        self.ld = ld
        self.device = device
        self.ideal = ideal

        
    def __iter__(self):
        self.iter = iter(self.ld)
        return self


    def __next__(self):
        x, y = next(self.iter)
        x = [x_b.to(self.device) for x_b in x]
        y = [{k: v.to(self.device) for k, v in y_b.items()} for y_b in y]
        with tc.no_grad():
            yh, y = self.mdl_viewer(x, y, ideal=self.ideal)
            # if self.ideal:
            #     ## proposal network contains the groud-truth
            #     yh, y = self.mdl_viewer(x, y, ideal)
            # else:
            #     yh, y = self.mdl_viewer(x), y
        return yh, y

    
class DetectionVisLoader:
    def __init__(self, ld_x, ld_loc, mdl_loc_ps):
        warnings.warn('loader is training mode')
        self.ld_x = ld_x
        self.ld_loc = ld_loc
        self.mdl_loc_ps = mdl_loc_ps


    def __iter__(self):
        self.iter_x = iter(self.ld_x)
        self.iter_loc = iter(self.ld_loc)
        return self


    def __next__(self):
        x_img, _ = next(self.iter_x)
        x_loc, y_loc = next(self.iter_loc)
        x_loc_cc = {k: tc.cat(v, 0) for k, v in x_loc.items()}
        loc_lb, loc_ub = self.mdl_loc_ps.set_boxapprox(x_loc_cc)
        ps_loc = tc.cat([loc_lb[:, :2], loc_ub[:, 2:]], 1)
        ps_loc = tc.split(ps_loc, [x.shape[0] for x in x_loc['mu']])
        assert(len(x_img) == len(y_loc) == len(x_loc['mu']) == len(ps_loc))

        return x_img, y_loc, x_loc['mu'], ps_loc
        

    #     print(pred)
    #     obj_valid = predset_obj[:, 1]
    #     pred_loc_valid = pred[0]['boxes'][obj_valid] 
    #     predset_loc_valid = tc.cat([predset_loc_lb[obj_valid, :2], predset_loc_ub[obj_valid, 2:]], 1)


        
    
class ProposalView(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl.eval()


    def forward(self, images, targets=None, ideal=False):
        assert(targets is not None)
        assert(ideal is False)

        with tc.no_grad():
            pred_list = self.mdl.forward(images) ## do not pass targets
        pred_ret = {'proposal': [p['proposals'] for p in pred_list], 
                    'score': [p['proposal_scores'] for p in pred_list],
                    'image': images}
        targets_ret = [t['boxes'] for t in targets]
        return pred_ret, targets_ret
        


class ObjectnessView(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl.eval()

        
    def forward(self, images, targets=None, ideal=True):
        # only used in construct a prediction set
        assert(ideal)
        assert(targets is not None)
        
        with tc.no_grad():
            pred_list = self.mdl.forward(images, targets) ## pass targets such that the proposal network is ideal

        labels = tc.cat([p['labels'] for p in pred_list])
        scores = tc.cat([p['scores'] for p in pred_list])
        device = scores.device
        if targets is not None:
            labels_gt = tc.cat([p['labels_gt'] for p in pred_list])
            val = labels_gt > 0

            ## ignore background predictions
            scores = scores[val] 
            labels_gt = labels_gt[val]
            labels = labels[val]
            
            #print(scores.shape, labels.shape, labels_gt.shape)
            
            ph = tc.where((labels == labels_gt).unsqueeze(1), tc.stack([1-scores, scores], 1), tc.stack([scores, 1-scores], 1)) ## good approximation
        else:
            ph = tc.stack([1-scores, scores], 1)
        
        if len(ph) == 0:
            dummy = tc.zeros(0, device=device)
            pred = {'ph': dummy.unsqueeze(1), 'logph': dummy.unsqueeze(1), 'yh_top': dummy, 'ph_top': dummy, 'logph_y': dummy}
            y = dummy
        else:
            pred = {'ph': ph, 'logph': ph.log(), 'yh_top': ph.argmax(-1), 'ph_top': ph.max(-1)[0], 'logph_y': ph[:, 1].log()}
            y = tc.ones(len(ph), device=device).long() # always one
        
        if targets is None:
            return pred
        else:
            return pred, y
        
            
        # scores = tc.cat([p['scores'] for p in pred_list])
        # labels = tc.cat([p['labels'] for p in pred_list])
        # #ph = tc.where(labels.unsqueeze(1)>0, tc.stack([1-scores, scores], 1), tc.stack([scores, 1-scores], 1))

        # scores = scores[labels>0] ## ignore background
        # ph = tc.stack([1-scores, scores], 1)
        
        # pred = {'ph': ph, 'logph': ph.log(), 'yh_top': ph.argmax(-1), 'ph_top': ph.max(-1)[0]}
        
        # y = tc.cat([p['labels_gt'] for p in pred_list])
        # y = (y >= 1).long() ## if there is an object other than background, otherwise zero

        # pred['logph_y'] = ph.log().gather(1, y.view(-1, 1)).squeeze(1)
        # return pred, y

        
class LocationView(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl.eval()

        
    def forward(self, images, targets=None, ideal=True):
        assert(ideal)
        assert(targets is not None)

        with tc.no_grad():
            pred_list = self.mdl.forward(images, targets)
        
        mu = tc.cat([pred['boxes'] for pred in pred_list], 0)
        logvar = tc.cat([pred['boxes_logvar'] for pred in pred_list], 0)
        pred = {'mu': mu, 'logvar': logvar}
        
        if targets is None:
            return pred
        else:
            obj_gt = tc.cat([pred['labels_gt'] for pred in pred_list], 0)
            y = tc.cat([pred['boxes_gt'] for pred in pred_list], 0)
                        
            # only consider foreground examples since a bounding box is valid if it's foreground
            valid = obj_gt > 0
            pred['mu'] = pred['mu'][valid]
            pred['logvar'] = pred['logvar'][valid]
            y = y[valid]
            
            pred['logph_y'] = - neg_log_prob(pred['mu'], pred['logvar'], y)

            return pred, y


class FilterLabel(nn.Module):
    def __init__(self, mdl, target_labels):
        super().__init__()
        self.mdl = mdl
        self.target_labels = target_labels


    def get_mdl(self):
        return self.mdl
    
        
    def forward(self, images, targets=None):
        pred = self.mdl(images, targets)

        if self.target_labels is None:
            return pred
        
        pred_new = []
        for p in pred:
            # # ground-truth handling
            # labels_gt = pred[i]['labels_gt']
            # print(labels_gt)
            # idx_valid = tc.zeros_like(labels_gt).bool()
            # for l in self.target_labels:
            #     idx_valid = idx_valid | (labels_gt==l)
            # pred[i]['labels_gt'] = pred[i]['labels_gt'][idx_valid]
            # pred[i]['boxes_gt'] = pred[i]['boxes_gt'][idx_valid]
            # pred[i]['boxes_gt_in_feat'] = pred[i]['boxes_gt_in_feat'][idx_valid]

            ## prediction handling
            labels = p['labels']
            idx_valid = tc.zeros_like(labels).bool()
            for l in self.target_labels:
                idx_valid = idx_valid | (labels==l)
            p_new = {
                'labels': p['labels'][idx_valid],
                'scores': p['scores'][idx_valid],
                'boxes': p['boxes'][idx_valid],
                'boxes_cons': p['boxes_cons'][idx_valid],
                'boxes_logvar': p['boxes_logvar'][idx_valid],
                'boxes_in_feat': p['boxes_in_feat'][idx_valid],
                'boxes_cons_in_feat': p['boxes_cons_in_feat'][idx_valid],
                'boxes_logvar_in_feat': p['boxes_logvar_in_feat'][idx_valid],
                
                'proposals': p['proposals'],
                'proposal_scores': p['proposal_scores'],
                'proposals_in_feat': p['proposals_in_feat'], 
            }

            if targets is not None:
                p_new['labels_gt'] = p['labels_gt'][idx_valid]
                p_new['boxes_gt'] = p['boxes_gt'][idx_valid]
                p_new['boxes_gt_in_feat'] = p['boxes_gt_in_feat'][idx_valid]
            
            pred_new.append(p_new)
        return pred_new
