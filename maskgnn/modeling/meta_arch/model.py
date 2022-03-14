# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torchvision
from detectron2.utils.events import get_event_storage

from torch import nn
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads

# New imports
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from maskgnn.modeling.matcher.build import build_matcher
from maskgnn.modeling.obj_encoder.build import build_obj_encoder

__all__ = ["CenterMaskGNN"]


def prepare_freeze_settings(cfg):
    freeze_settings = {"backbone": cfg.MODEL.FREEZE.BACKBONE,
                       "proposal_generator": cfg.MODEL.FREEZE.PROPOSAL_GENERATOR,
                       "roi_heads": cfg.MODEL.FREEZE.ROI_HEADS,
                       "tracker_net" : cfg.MODEL.FREEZE.TRACKER_NET,
                       "gnn": cfg.MODEL.FREEZE.GNN,
                       "obj_encoder": cfg.MODEL.FREEZE.OBJ_ENCODER}
    return freeze_settings

import torch


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (n, k) if is_aligned == False else shape (n, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


@META_ARCH_REGISTRY.register()
class CenterMaskGNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            matcher: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            freeze_settings: {},
            tracker_mode: str,
            trn_double_loader: bool,
            val_double_loader: bool,
            debug_return_gt: bool,
            fully_unsupervised: bool,
            train_gnn_only: bool,
            use_pooled_obj_feats : bool,
            roi_pooler_resolution: int,
            mask_feature_src : str,
            freeze_cls_heads: bool,
            losses_cfg,
            num_classes
    ):
        """
        NOTE: this interface is experimental.
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()

        self.losses_cfg = losses_cfg
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.freeze_settings = freeze_settings
        self.trn_double_loader = trn_double_loader
        self.val_double_loader = val_double_loader

        # New
        self.matcher = matcher

        # Tracking related params
        self.tracker_mode = tracker_mode

        # Init prevs.
        self.prev_bboxes = None
        self.prev_roi_feats = None
        self.prev_det_labels = None
        self.prev_masks = None
        self.prev_scores = None

        self.processing_video = "-1"
        self.is_first = False

        # If this setting is set to True, no forward-propagation will be done. GT instances are returned.
        self.debug_return_gt = debug_return_gt
        self.fully_unsupervised = fully_unsupervised
        self.train_gnn_only = train_gnn_only
        self.use_pooled_obj_feats = use_pooled_obj_feats
        self.roi_pooler_resolution = roi_pooler_resolution
        self.mask_feature_src = mask_feature_src
        self.freeze_cls_heads = freeze_cls_heads

        # Freeze params
        logger = logging.getLogger(__name__)
        for module_name in freeze_settings:
            if freeze_settings[module_name]:
                logger.warning(f"[MaskGNN] Freezing module: {module_name}")
                # Tracker net is a small net in fcos
                if module_name == "tracker_net":
                    continue
                    module = self.proposal_generator.fcos_head.tracking
                    for param in module.parameters():
                        param.requires_grad = False
                else:
                    module = self.__getattr__(module_name)
                    for param in module.parameters():
                        param.requires_grad = False


        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # Matcher to assign box proposals to gt boxes
        self.num_classes = num_classes

        assert (self.pixel_mean.shape == self.pixel_std.shape), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "matcher": build_matcher(cfg),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "freeze_settings": prepare_freeze_settings(cfg),
            "trn_double_loader": cfg.DATASETS.TRN_DOUBLE_LOADER,
            "val_double_loader": cfg.DATASETS.VAL_DOUBLE_LOADER,
            "tracker_mode": cfg.MASKGNN.TRACKER_MODE,
            "debug_return_gt": cfg.DEBUG.RETURN_GT,
            "fully_unsupervised": cfg.MASKGNN.FULLY_UNSUPERVISED,
            "train_gnn_only" : cfg.MASKGNN.TRAIN_GNN_ONLY,
            "use_pooled_obj_feats": cfg.MODEL.MATCHER.OBJ_ENCODER.NAME == "EncoderCNN",
            "roi_pooler_resolution": cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
            "mask_feature_src" : cfg.MASKGNN.MASK_FEATURE_SRC,
            "losses_cfg": cfg.MASKGNN.LOSSES,
            "freeze_cls_heads" : cfg.MASKGNN.FREEZE_CLS_HEADS,
            "num_classes" : cfg.MODEL.FCOS.NUM_CLASSES
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_key="image"):
        """
        Normalize, pad and batch the input images.
        """
        images = [x[image_key].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        #instances = instance_level_nms(instances)

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            #print(len(results_per_image))
            if len(results_per_image) != 0:

                r = detector_postprocess(results_per_image, height, width)


                keep_idxs = torchvision.ops.batched_nms(boxes=r.pred_boxes.tensor, scores= r.scores, idxs=r.pred_classes, iou_threshold=0.6)
                r = r[keep_idxs]
                if len(r) > 10:
                    print("FILTER")
                    scores  = r.scores
                    idxs = torch.argsort(scores,descending=True)[:10]
                    r = r[idxs]


                processed_results.append({"instances": r})
            else:
                print("No dets!")
                processed_results.append({"instances": results_per_image})

        return processed_results


    def _tracker_update(self, dets):

        assert len(dets) == 1  # The batch size should be 1
        det = dets[0]
        det_bboxes = det.pred_boxes.tensor
        det_roi_feats = det.obj_features
        det_labels = det.pred_classes
        det_masks = det.pred_masks.to("cpu")
        det_scores = det.scores

        if len(det) == 0:
            det.set("tracking_id", torch.tensor([]))
            return dets, None


        if self.is_first or (not self.is_first and self.prev_bboxes is None):


            # Inject tracking ids

            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = det_roi_feats
            self.prev_det_labels = det_labels
            self.prev_masks = det_masks

            det_obj_ids = np.arange(len(det_bboxes), dtype=np.int32)

            debug_dict = None

        else:

            assert self.prev_roi_feats is not None

            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)]
            prev_bbox_img_n = [self.prev_roi_feats.size(0)]

            match_score = self.matcher(det_roi_feats, self.prev_roi_feats, bbox_img_n, prev_bbox_img_n)[0]
            match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)
            label_delta = (self.prev_det_labels == det_labels.view(-1,1)).float()
            bbox_ious = bbox_overlaps(det_bboxes[:,:4], self.prev_bboxes[:,:4])

            # compute comprehensive score
            comp_scores, debug_dict = self.matcher.compute_comp_scores(match_logprob,
                det_scores[:,None],
                bbox_ious,
                label_delta,
                add_bbox_dummy=True)

            match_likelihood, match_ids = torch.max(comp_scores, dim =1)

            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of existing object,
            # add tracking features/bboxes of new object
            match_ids = match_ids.cpu().numpy().astype(np.int32)

            det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1)

            old_idx_keeper = {}

            best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)

            for idx, match_id in enumerate(match_ids):
                if match_id == 0:

                    # add new object
                    det_obj_ids[idx] = self.prev_roi_feats.size(0)
                    self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                    self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                    self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                    self.prev_masks = torch.cat((self.prev_masks, det_masks[idx][None]), dim=0)

                else:

                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score

                    obj_id = match_id - 1
                    match_score = comp_scores[idx, match_id]

                    if match_score > best_match_scores[obj_id]:

                        # Find old index.
                        if obj_id in old_idx_keeper.keys():
                            old_idx = old_idx_keeper[obj_id]
                            det_obj_ids[old_idx] = -1

                        det_obj_ids[idx] = obj_id

                        # Keep the old idx again.
                        old_idx_keeper[obj_id] = idx

                        best_match_scores[obj_id] = match_score


                        # update feature
                        self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                        self.prev_bboxes[obj_id] = det_bboxes[idx]
                        self.prev_masks[obj_id] = det_masks[idx]

            debug_dict["det_obj_ids"] = det_obj_ids
            debug_dict["match_ids"] = match_ids
            debug_dict["n_curr_dets"] = bbox_img_n[0]
            debug_dict["n_prev_dets"] = prev_bbox_img_n[0]

        # Assemble results.
        #print("det_obj_ids: ", len(det_obj_ids) , " - ", det_obj_ids)
        #print("dets: " , len(det))


        remove_idxs = det_obj_ids != -1
        det.set("tracking_id", torch.tensor(det_obj_ids))
        det = det.to("cpu")[remove_idxs]

        return [det], debug_dict

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"

        """

        if not self.training:

            if "file_name_0" in batched_inputs[0]:

                # THIS IS A STANDARD VIDEO DATASET, E.G. YT-VOS, DAVIS.
                current_video_name = batched_inputs[0]['file_name_0'].split('/')[-2]
                width = batched_inputs[0]["width"]
                height = batched_inputs[0]["height"]

                if self.processing_video != current_video_name and self.tracker_mode != "off":
                    print("Resetting tracker... - ", current_video_name)
                    self.is_first = False
                    self.prev_bboxes = None
                    self.prev_roi_feats = None
                    self.prev_det_labels = None
                    self.processing_video = current_video_name

            else:

                # Two frame detection case
                width = batched_inputs[0]["width"]
                height = batched_inputs[0]["height"]

                print("Resetting tracker...")

            if self.debug_return_gt:
                logger = logging.getLogger(__name__)
                logger.warning("[MaskGNN] - Returning GroundTruth dets.")
                return self.return_gt(batched_inputs)

            if self.val_double_loader:
                # This takes care of the tracker during the inference.

                # End of tracker handling.
                if "image_1" in batched_inputs[0]:
                    return self.inference_double(batched_inputs)
                else:
                    return self.inference_single(batched_inputs)
            else:

                return self.inference_single(batched_inputs)

            # END OF INFERENCE PART
        else:


            # Training redirection.
            if self.train_gnn_only:
                return self.forward_gnn(batched_inputs)
            if self.trn_double_loader:
                return self.forward_double(batched_inputs)
            else:
                return self.forward_single(batched_inputs)

    def forward_single(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):

        """

        :param batched_inputs:
        :return: losses (dict)
        """

        images_0 = self.preprocess_image(batched_inputs, image_key="image_0")
        gt_instances_0 = [x["instances_0"].to(self.device) for x in batched_inputs]

        features_0 = self.backbone(images_0.tensor)
        proposals, proposal_losses = self.proposal_generator(images_0=images_0,
                                                             features_0=features_0,
                                                             gt_instances_0=gt_instances_0)

        _, detector_losses = self.roi_heads(images_0, features_0, proposals, gt_instances_0)

        losses = {}
        for key in proposal_losses.keys():
            losses[key] = proposal_losses[key]
        for key in detector_losses.keys():
            losses[key] = detector_losses[key]

        return losses

    def forward_double(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):

        """
        Use this fn during training.
        :param batched_inputs:
        :return: loss_dict
        """

        losses = {}
        images_0 = self.preprocess_image(batched_inputs, image_key="image_0")
        images_1 = self.preprocess_image(batched_inputs, image_key="image_1")
        gt_instances_0 = [x["instances_0"].to(self.device) for x in batched_inputs]
        gt_instances_1 = [x["instances_1"].to(self.device) for x in batched_inputs]

        # Run VoVNet
        features_0, features_1 = self.backbone(images_0.tensor), self.backbone(images_1.tensor)

        proposals_1, proposal_losses = self.proposal_generator(images_0=images_1,
                                                               features_0=features_1,
                                                               gt_instances_0=gt_instances_1)

        # Proposal losses (dict) has the following keys:
        # ['loss_fcos_cls', 'loss_fcos_loc', 'loss_fcos_ctr', 'tracking_loss']
        for key in proposal_losses.keys():
            save = False
            if key == "loss_fcos_cls" and self.losses_cfg.FCOS_CLS:
                save = True
            elif key == "loss_fcos_loc" and self.losses_cfg.FCOS_LOC:
                save = True
            elif key == "loss_fcos_ctr" and self.losses_cfg.FCOS_CTR:
                save = True
            # Save in the end if cond satisfied.
            if save:
                losses[key] = proposal_losses[key]


        # Run SAM
        dets_1, detector_losses_1 = self.roi_heads(images_1, features_1, proposals_1, gt_instances_1)
        losses["loss_mask"] = detector_losses_1["loss_mask"]

        # dict_keys(['scores', 'pred_classes', 'locations', 'proposal_boxes', 'deltas', 'gt_classes',
        # 'gt_boxes', 'gt_track_id', 'gt_masks', 'gt_deltas', 'obj_features', 'pred_masks'])

        dets_0, ref_x_n = self.roi_heads.encode_gt_boxes(features_0, gt_instances_0)

        ids = []

        # Process each image ref_det & det pairs one by one.
        for ref_det, det in zip(dets_0,dets_1):
            ref_track_ids = ref_det.gt_track_id.cpu().numpy()
            track_ids = det.gt_track_id.cpu().numpy()

            lut = {}
            for idx, id in enumerate(ref_track_ids):
                lut[id] = idx

            tracks = []
            for track_id in track_ids:
                if track_id in lut.keys():
                    # add one, because we leave the first index for new objects.
                    tracks.append(lut[track_id] + 1)
                else:
                    tracks.append(0)
            ids.append(torch.tensor(tracks).to(self.device))


        x_n = [len(det) for det in dets_1]

        x = torch.cat([det.obj_features for det in dets_1],dim=0)
        ref_x = torch.cat([det.obj_features for det in dets_0], dim=0)


        match_score = self.matcher.forward(x, ref_x, x_n, ref_x_n)

        matching_loss, acc = self.matcher.matching_loss(match_score, ids)
        storage = get_event_storage()
        storage.put_scalar("matcher/accuracy", acc)

        losses.update(matching_loss)

        return losses

    """
    Inference Functions.
    """

    def inference_single(
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]]
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs, image_key="image_0")
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        dets, _ = self.roi_heads(images, features, proposals, None)

        dets, debug_dict = self._tracker_update(dets)
        dets = CenterMaskGNN._postprocess(dets, batched_inputs, images.image_sizes)

        dets[0]["debug_dict"] = debug_dict



        return dets


    def inference_double(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):

        """
        Use this function to evaluate the tracking performance.
        BATCH SIZE IS ONE
        This fn oes not instantiate Tracker.
        :param batched_inputs:
        :return:
        """

        assert not self.training

        # Preprocess images
        images_0 = self.preprocess_image(batched_inputs, image_key="image_0")
        images_1 = self.preprocess_image(batched_inputs, image_key="image_1")

        # Run VoVNet
        features_0 = self.backbone(images_0.tensor)
        features_1 = self.backbone(images_1.tensor)

        # Run FCOS
        proposals_0, proposals_1, _ = self.proposal_generator(images_0=images_0,
                                                              images_1=images_1,
                                                              features_0=features_0,
                                                              features_1=features_1)

        # Run SAM
        dets_0, _ = self.roi_heads(images_0, features_0, proposals_0)


        if self.tracker_mode == "off":

            # Calculate second frame dets
            dets_1, _ = self.roi_heads(images_1, features_1, proposals_1)
            dets_0 = self._postprocess(dets_0, batched_inputs, images_0.image_sizes)
            dets_1 = self._postprocess(dets_1, batched_inputs, images_1.image_sizes)

            return dets_0, dets_1


        elif self.tracker_mode == "gnn":
            num_objs_0 = len(dets_0[0])

            # Zero object case.
            if num_objs_0 == 0:
                dets_0[0].current_states = torch.zeros(0, 12).cuda()

            else:
                obj_features = dets_0[0].obj_features

                objs = self.obj_encoder(obj_features)

                # Copy the bbox, then normalize
                height, width = dets_0[0].image_size
                bboxes = dets_0[0].pred_boxes.tensor.clone()
                bboxes[:, 0] /= width
                bboxes[:, 1] /= height
                bboxes[:, 2] /= width
                bboxes[:, 3] /= height

                # Merge an object vector.
                objs = torch.cat([objs, bboxes], dim=1)

                dets_0[0].current_states = objs

                # Predict the transition delta
                tracking_delta = dets_0[0].pred_deltas

                # Predict a delta.
                pred_delta = self.gnn.message_passing(objs, [objs.shape[0]], tracking_delta)
                objs = objs + pred_delta

                dets_0[0].next_states = objs
        else:
            print("Invalid tracking mode!")

        # Do postprocess
        dets_0 = self._postprocess(dets_0, batched_inputs, images_0.image_sizes)
        dets_0 = self._tracker_update(dets_0)

        return dets_0