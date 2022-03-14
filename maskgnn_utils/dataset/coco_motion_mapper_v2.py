import copy
import logging
from typing import List, Optional, Union
import torch
import numpy as np

from detectron2.config import configurable
import maskgnn_utils.dataset.map_utils as utils
from detectron2.data import transforms as T
from maskgnn_utils.dataset.imgaug_backend import *

__all__ = ["COCOMotionMapper"]

class COCOMotionMapper:

    """
    The callable currently does the following:

    1. Read the images from "file_name_1" and "file_name_2
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        coco_motion_params,
    ):

        """

        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """

        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"

        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.coco_motion_params     = coco_motion_params
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):

        augs = utils.build_augmentation(cfg, is_train)
        print("INSIDE FROM CONFIG: AUGS =>>>>>>", augs)

        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        coco_motion_params = cfg.COCO_MOTION_AUG


        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "coco_motion_params": coco_motion_params,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        # Create the augmenter.
        augmenter = ImageToSeqAugmenter(perspective=self.coco_motion_params.PERSPECTIVE,
                                        affine=self.coco_motion_params.AFFINE,
                                        motion_blur=self.coco_motion_params.MOTION_BLUR,
                                        brightness_range=self.coco_motion_params.BRIGHTNESS_RANGE,
                                        hue_saturation_range=self.coco_motion_params.HUE_SATURATION_RANGE,
                                        perspective_magnitude=self.coco_motion_params.PERSPECTIVE_MAGNITUDE,
                                        scale_range=self.coco_motion_params.SCALE_RANGE,
                                        translate_range={"x": self.coco_motion_params.TRANSLATE_RANGE_X,
                                                         "y": self.coco_motion_params.TRANSLATE_RANGE_Y},
                                        rotation_range=self.coco_motion_params.ROTATION_RANGE,
                                        motion_blur_kernel_sizes=self.coco_motion_params.MOTION_BLUR_KERNEL_SIZES,
                                        motion_blur_prob=self.coco_motion_params.MOTION_BLUR_PROB,
                                        identity_mode=self.coco_motion_params.IDENTITY_MODE,
                                        seed_override=self.coco_motion_params.SEED_OVERRIDE)

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        annos = dataset_dict['annotations']

        annos_0, annos_1 = [], []
        id_counter = 0
        for obj in annos:
            if obj.get("iscrowd", 0) == 0:
                obj["track_id"] = id_counter + 1
                annos_0.append(copy.deepcopy(obj))
                annos_1.append(copy.deepcopy(obj))
                id_counter += 1

        # Load image
        image_0 = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image_0)

        # Process the first image
        aug_input_0 = T.AugInput(image_0, sem_seg=None)
        transforms_0 = self.augmentations(aug_input_0)
        image_0 = aug_input_0.image
        image_1 = image_0.copy()

        # Make a copy.
        image_shape = image_0.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        if "annotations" in dataset_dict:

            # Process the first one.
            annos_0 = [utils.transform_instance_annotations(obj, transforms_0, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices)
                       for obj in annos_0
                       if obj.get("iscrowd", 0) == 0]

            instances_0 = utils.annotations_to_instances(annos_0, image_shape, mask_format=self.instance_mask_format)

            # Process the second one.
            annos_1 = [utils.transform_instance_annotations(obj, transforms_0, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices)
                for obj in annos_1
                if obj.get("iscrowd", 0) == 0
            ]

            # Hallucinate motion here.
            image_1, annos_1 = augmenter(image=image_1, annotations=annos_1)
            instances_1 = utils.annotations_to_instances(annos_1, image_shape, mask_format=self.instance_mask_format)

            dataset_dict["instances_0"] = instances_0
            dataset_dict["instances_1"] = instances_1

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.

            instances_0 = utils.filter_empty_instances(instances_0)
            instances_1 = utils.filter_empty_instances(instances_1)

            # After we're done, calculate matchings etc.
            track_ids_0 = instances_0.gt_track_id
            track_ids_1 = instances_1.gt_track_id

            # Match the targets based on the index.
            matches_0, matches_1 = [], []
            gt_bboxes_0, gt_bboxes_1 = [], []

            if len(instances_0.gt_boxes) != 0:

                for i, id_0 in enumerate(track_ids_0):
                    is_matched = False

                    for j, id_1 in enumerate(track_ids_1):
                        if id_0 == id_1:
                            matches_0.append(i)
                            matches_1.append(j)
                            gt_bboxes_0.append(instances_0.gt_boxes.tensor[i, :])
                            gt_bboxes_1.append(instances_1.gt_boxes.tensor[j, :])
                            is_matched = True

                    if not is_matched:

                        gt_bboxes_0.append(torch.zeros_like(instances_0.gt_boxes.tensor[i, :]))
                        gt_bboxes_1.append(torch.zeros_like(instances_0.gt_boxes.tensor[i, :]))

                gt_bboxes_0 = torch.stack(gt_bboxes_0, dim=0)
                gt_bboxes_1 = torch.stack(gt_bboxes_1, dim=0)

                # Calculate loss target.
                gt_w0 = gt_bboxes_0[:, 2] - gt_bboxes_0[:, 0] + 0.0001
                gt_h0 = gt_bboxes_0[:, 3] - gt_bboxes_0[:, 1] + 0.0001
                gt_w1 = gt_bboxes_1[:, 2] - gt_bboxes_1[:, 0] + 0.0001
                gt_h1 = gt_bboxes_1[:, 3] - gt_bboxes_1[:, 1] + 0.0001

                gt_delta_x = (gt_bboxes_1[:, 0] - gt_bboxes_0[:, 0]) / gt_w0
                gt_delta_y = (gt_bboxes_1[:, 1] - gt_bboxes_0[:, 1]) / gt_h0
                gt_delta_w = torch.log(gt_w1 / gt_w0)
                gt_delta_h = torch.log(gt_h1 / gt_h0)


                assert not gt_delta_x.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)

                assert not gt_delta_y.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)

                assert not gt_delta_w.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)

                assert not gt_delta_h.isnan()[0], print("gt_bboxes_0:", gt_bboxes_0,
                                                        "gt_bboxes_1:", gt_bboxes_1,
                                                        "gt_delta_x: ", gt_delta_x,
                                                        "gt_delta_y: ", gt_delta_y,
                                                        "gt_delta_w: ", gt_delta_w,
                                                        "gt_delta_h: ", gt_delta_h,
                                                        "gt_w0: ", gt_w0,
                                                        "gt_h0: ", gt_h0)


                gt_deltas = torch.stack([gt_delta_x, gt_delta_y, gt_delta_w, gt_delta_h], dim=1)


                if len(gt_deltas) > 0:
                    instances_0.gt_deltas = gt_deltas

            # Saving instances
            dataset_dict["instances_0"] = instances_0
            dataset_dict["instances_1"] = instances_1
            dataset_dict["matches_0"] = matches_0
            dataset_dict["matches_1"] = matches_1

        else:

            image_1, annots = augmenter(image=image_1, annotations=None)

        dataset_dict["image_0"] = torch.as_tensor(np.ascontiguousarray(image_0.transpose(2, 0, 1)))/255.
        dataset_dict["image_1"] = torch.as_tensor(np.ascontiguousarray(image_1.transpose(2, 0, 1)))/255.

        return dataset_dict

