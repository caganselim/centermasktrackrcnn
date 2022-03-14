import copy
import logging
from typing import List, Optional, Union
import torch
import numpy as np

from detectron2.config import configurable
from detectron2.structures import BoxMode
import maskgnn_utils.dataset.map_utils as utils

from skimage import transform

from detectron2.data import transforms as T

__all__ = ["COCOMotionMapper"]

from maskgnn_utils.dataset.coco_motion_utils import get_affine_transform, _get_aug_param, warp_affine, \
    post_process_annotations_v2


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
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        # CenterTrack
        self.scale = 0.05
        self.shift = 0.05

        self.canvas_width = 800
        self.canvas_height = 600

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):

        augs = [T.RandomFlip(horizontal=True, vertical=False)]

        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
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

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # Load image
        image_0 = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image_0)
        image_shape = image_0.shape[:2]  # h, w

        aug_input_0 = T.AugInput(image_0, sem_seg=None)
        transforms_0 = self.augmentations(aug_input_0)
        image_0 = aug_input_0.image

        # Make a copy.
        image_1 = image_0.copy()

        # Parameters are ca
        c = np.array([image_0.shape[1] / 2., image_0.shape[0] / 2.], dtype=np.float32)
        s = max(image_0.shape[0], image_0.shape[1])
        height, width = image_0.shape[0], image_0.shape[1]
        c, aug_s, rot = _get_aug_param(c, s, width, height, scale=self.scale, shift=self.shift)
        s = s * aug_s

        # Prepare the transformation matrix for the second frame.
        trans_1 = get_affine_transform(c, s, 0, [self.canvas_height, self.canvas_width])
        trans_1 = np.asarray(trans_1)
        pad = np.array([[0, 0, 1]], dtype=np.float32)
        trans_1 = np.concatenate((trans_1, pad))
        trans_1 = transform.ProjectiveTransform(trans_1)

        image_1 = warp_affine(image=image_1, matrix=trans_1, cval=(0, 0, 0),
                              interpolation=1, mode=0, output_shape=(self.canvas_height, self.canvas_width))
        image_1 = (image_1.astype(np.float32)/255.)

        # This time, disturb center & scale
        c_pre, aug_s_pre, _ = _get_aug_param(c, s, width, height, disturb=True, scale=self.scale, shift=self.shift)
        s_pre = s * aug_s_pre

        # Prepare the transformation matrix for the first frame.
        trans_0 = get_affine_transform(c_pre, s_pre, 0, [self.canvas_height, self.canvas_width])
        trans_0 = np.asarray(trans_0)
        pad = np.array([[0, 0, 1]], dtype=np.float32)
        trans_0 = np.concatenate((trans_0, pad))
        trans_0 = transform.ProjectiveTransform(trans_0)

        image_0 = warp_affine(image=image_0, matrix=trans_0, cval=(0, 0, 0),
                              interpolation=1, mode=0, output_shape=(self.canvas_height, self.canvas_width))

        image_0 = (image_0.astype(np.float32) / 255.)


        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        dataset_dict["image_0"] = torch.as_tensor(np.ascontiguousarray(image_0.transpose(2, 0, 1)))
        dataset_dict["image_1"] = torch.as_tensor(np.ascontiguousarray(image_1.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:

            annos = [utils.transform_instance_annotations(obj,
                                                          transforms_0,
                                                          image_shape,
                                                          keypoint_hflip_indices=self.keypoint_hflip_indices)
                       for obj in dataset_dict.pop("annotations")
                       if obj.get("iscrowd", 0) == 0]

            for id, obj in enumerate(annos):
                obj["track_id"] = id

            annos_0 = copy.deepcopy(annos)
            annos_1 = copy.deepcopy(annos)

            # Apply affine transformation on the annotations.
            annos_0 = post_process_annotations_v2(annos_0, trans_0, canvas_height=self.canvas_height, canvas_width=self.canvas_width)
            annos_1 = post_process_annotations_v2(annos_1, trans_1, canvas_height=self.canvas_height, canvas_width=self.canvas_width)

            # Convert annotations to detectrons Instance format.
            instances_0 = utils.annotations_to_instances(annos_0, (self.canvas_height, self.canvas_width))
            instances_1 = utils.annotations_to_instances(annos_1, (self.canvas_height, self.canvas_width))

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

        return dataset_dict

