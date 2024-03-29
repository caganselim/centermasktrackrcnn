# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import Boxes, Instances, pairwise_iou, ImageList
from detectron2.utils.events import get_event_storage
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.layers import ShapeSpec

from .mask_head import build_mask_head, mask_rcnn_loss, mask_rcnn_inference
from .maskiou_head import build_maskiou_head, mask_iou_loss, mask_iou_inference
from .proposal_utils import add_ground_truth_to_proposals
from .pooler import ROIPooler

__all__ = ["CenterROIHeads"]


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks

class ROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # ywlee for using targets.gt_classes
        # in add_ground_truth_to_proposal()
        # gt_boxes = [x.gt_boxes for x in targets]

        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).


        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)


        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []

        # Process each image one by one.
        for proposals_per_image, targets_per_image in zip(proposals, targets):

            has_gt = len(targets_per_image) > 0

            # Calculates pairwise IoU. Gets N and M bounding boxes, return NxM Tensor.
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)

            # Returns N best matched ground truth index
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)


            # Sample proposals from the networks output.
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals.
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:

                # Get ground truth
                sampled_targets = matched_idxs[sampled_idxs]

                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.

                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:

                # Why is that?
                gt_boxes = Boxes(targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4)))
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads (TensorBoard)
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
            detected instances. Returned during inference only; may be [] during training.

            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()

@ROI_HEADS_REGISTRY.register()
class CenterROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches  masks directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(CenterROIHeads, self).__init__(cfg, input_shape)
        self._init_mask_head(cfg)
        self._init_mask_iou_head(cfg)

        # pooled, logits
        self.mask_feature_src = cfg.MASKGNN.MASK_FEATURE_SRC

    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        assign_crit       = cfg.MODEL.ROI_MASK_HEAD.ASSIGN_CRITERION

        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            assign_crit=assign_crit,
        )

        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )


    def _init_mask_iou_head(self, cfg):
        # fmt: off
        self.maskiou_on     = cfg.MODEL.MASKIOU_ON
        if not self.maskiou_on:
            return
        in_channels         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        pooler_resolution   = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self.maskiou_weight = cfg.MODEL.MASKIOU_LOSS_WEIGHT

        # fmt : on

        self.maskiou_head = build_maskiou_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """

        del images

        if self.training:


            proposals = self.label_and_sample_proposals(proposals, targets)


        del targets

        if self.training:

            # Process masks
            losses, mask_features, selected_mask, labels, maskiou_targets, proposals = self._forward_mask(features, proposals)
            #losses, proposals = self._forward_mask(features, proposals)

            num_boxes_per_image = [len(i) for i in proposals]

            losses.update(self._forward_maskiou(mask_features, proposals, selected_mask, labels, maskiou_targets))


            return proposals, losses

        else:

            # During inference cascaded prediction is used: the mask heads are only
            # applied to the top scoring box detections.

            pred_instances = self.forward_with_given_boxes(features, proposals)

            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_maskiou(instances[0].get('mask_features'), instances)

        return instances

    def encode_gt_boxes(self, features, proposals):

        for idx, prop in enumerate(proposals):
            proposals[idx].set('pred_boxes', proposals[idx].gt_boxes)
            proposals[idx].set('pred_classes', proposals[idx].gt_classes)

        num_proposals_per_im = [len(proposals_per_im) for proposals_per_im in proposals]
        features = [features[f] for f in self.in_features]
        # mask_features => [K, 256, POOL_RESOLUTION, POOL_RESOLUTION]

        pooled_feats = self.mask_pooler(features, proposals, False)
        mask_feats = torch.split(pooled_feats, num_proposals_per_im)

        for idx, feats in enumerate(mask_feats):
            proposals[idx].set('obj_features', feats)

        return proposals, num_proposals_per_im


    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        # features are received from p3, p4, p5
        features = [features[f] for f in self.in_features]

        if self.training:

            # The loss is only defined on positive proposals.
            # Select fg proposals selects where a gt is assigned to an instance.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)

            # From now on, proposals are fg.
            # proposal_boxes = [x.proposal_boxes for x in proposals]
            num_proposals_per_im = [len(proposals_per_im) for proposals_per_im in proposals]

            # mask_features => [K, 256, POOL_RESOLUTION, POOL_RESOLUTION]
            pooled_feats = self.mask_pooler(features, proposals, self.training)

            # mask logits => [K, C, POOL_RESOLUTION*2, POOL_RESOLUTION*2] - K: #masks, C: #classes
            mask_logits = self.mask_head(pooled_feats)

            if self.mask_feature_src == "pooled":
                mask_feats = torch.split(pooled_feats, num_proposals_per_im)
            elif self.mask_feature_src == "logits":
                mask_feats = torch.split(mask_logits, num_proposals_per_im)

            for idx, feats in enumerate(mask_feats):
                proposals[idx].set('obj_features', feats)

            loss, selected_mask, labels, maskiou_targets = mask_rcnn_loss(mask_logits, proposals, True)

            mask_rcnn_inference(mask_logits, proposals) # ASK

            # now proposals become pred_instances.

            return {"loss_mask": loss}, pooled_feats, selected_mask, labels, maskiou_targets, proposals

        else:

            # pred_boxes = [x.pred_boxes for x in instances]

            mask_features = self.mask_pooler(features, instances)

            mask_logits = self.mask_head(mask_features)
            instances[0].set('mask_features', mask_features)

            if self.mask_feature_src == "pooled":
                instances[0].set('obj_features', mask_features)
            elif self.mask_feature_src == "logits":
                instances[0].set('obj_features', mask_logits)

            mask_rcnn_inference(mask_logits, instances)

            return instances



    def _forward_maskiou(self, mask_features, instances, selected_mask=None, labels=None, maskiou_targets=None):
        """
        Forward logic of the mask iou prediction branch.
        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, calibrate instances' scores.
        """

        if not self.maskiou_on:

            return {} if self.training else instances

        if self.training:

            pred_maskiou = self.maskiou_head(mask_features, selected_mask)

            return {"loss_maskiou": mask_iou_loss(labels, pred_maskiou, maskiou_targets, self.maskiou_weight)}

        else:
            selected_mask = torch.cat([i.pred_masks for i in instances], 0)
            if selected_mask.shape[0] == 0:
                return instances
            pred_maskiou = self.maskiou_head(mask_features, selected_mask)

            mask_iou_inference(instances, pred_maskiou)


            return instances
