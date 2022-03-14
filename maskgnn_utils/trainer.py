import logging
import os
from collections import OrderedDict

import torch
import time
import numpy as np

import matplotlib.pyplot as plt
import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import get_event_storage


from detectron2.evaluation import DatasetEvaluators, inference_on_dataset, print_csv_format, DatasetEvaluator

from maskgnn_utils import InstanceVisualizer
from maskgnn_utils.evaluators import TrackingDeltaEvaluator, UVOSWriter, COCOEvaluatorVID, DebugWriter

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog

from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)

from maskgnn_utils.evaluators.kitti_mots_writer import KITTIMOTSWriter
from maskgnn_utils.evaluators.ytvis_writer import YTVISWriter


class MaskGNNTrainer(DefaultTrainer):

    """
    This is the same Trainer except that we rewrite the
    `build_train_loader` method.

    # self.iter => trainer base current iteration.

    """

    @classmethod
    def build_train_loader(cls, cfg):

        # CenterMaskGNN
        # Overrided this function to decide a custom dataset mapper for our setup.
        # To pretrain our model with static images only, we use the default DatasetMapper
        # of detectron2. For the trainings that we require GNN, we use two_frame_mapper.

        print("Building train loader!")

        if cfg.DATASETS.TRN_DOUBLE_LOADER:

            if cfg.DATASETS.DATASET_NAME == "coco":

                from maskgnn_utils.dataset.coco_motion_mapper import COCOMotionMapper as DatasetMapper

            else:

                from maskgnn_utils.dataset.two_frame_mapper import TwoFrameDatasetMapperTrain as DatasetMapper

        else:
            from maskgnn_utils.dataset.one_frame_mapper import DatasetMapper

        mapper = DatasetMapper(cfg, is_train=True)

        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):

        print("Building test loader!")
        if cfg.DATASETS.VAL_DOUBLE_LOADER:

            if cfg.DATASETS.DATASET_NAME == "coco":

                from maskgnn_utils.dataset.coco_motion_mapper import COCOMotionMapper as DatasetMapper

            else:

                from maskgnn_utils.dataset.two_frame_mapper import TwoFrameDatasetMapperTest as DatasetMapper
        else:
            from maskgnn_utils.dataset.one_frame_mapper import OneFrameDatasetMapper as DatasetMapper

        mapper = DatasetMapper(cfg, is_train=False)

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """

        # Prepare the output folder.
        if output_folder is None:

            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            print("Evaluating last checkpoint: " , output_folder)

            # If the output folder doesn't exist, create.
            os.makedirs(output_folder, exist_ok=True)

        evaluator_list = []

        # Get the evaluator type from our dataset settings.
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["coco_vid"]:
            evaluator_list.append(COCOEvaluatorVID(dataset_name, output_dir=output_folder))

        # MaskGNN - Add two frame tracking evaluator type.
        if evaluator_type in ["two_frame_tracking"]:
            evaluator_list.append(TrackingDeltaEvaluator(dataset_name, output_dir=output_folder))

        if evaluator_type in ["uvos_writer"]:
            evaluator_list.append(UVOSWriter(dataset_name, output_dir=output_folder))

        if evaluator_type in ["ytvis_writer"]:
            evaluator_list.append(YTVISWriter(dataset_name, output_dir=output_folder))

        if evaluator_type in ["debug_writer"]:
            evaluator_list.append(DebugWriter(dataset_name, output_dir=output_folder))

        if evaluator_type in ["kitti_mots_writer"]:
            evaluator_list.append(KITTIMOTSWriter(dataset_name, output_dir=output_folder))

        # Return the evaluator if there is only one otherwise return DatasetEvaluators Object.
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)



    

