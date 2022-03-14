import os.path

from detectron2.engine import default_setup
from maskgnn.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from maskgnn_utils.evaluators.coco_helper import load_coco_json


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
        
    # Set score_threshold for builtin models
    thresh = cfg.MASKGNN.SCORE_THRESH_TEST
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = thresh

    cfg.freeze()
    default_setup(cfg, args)

    return cfg

def set_datasets(cfg):

    dataset_root = cfg.DATASETS.DATASET_DIR
    trn_json_file = cfg.DATASETS.TRN_JSON_PATH
    val_json_file = cfg.DATASETS.VAL_JSON_PATH
    padded_training = cfg.MASKGNN.PADDED_TRAINING_ON

    # Register dataset
    if cfg.DATASETS.DATASET_NAME == "coco":

        DatasetCatalog.register(f"{cfg.DATASETS.DATASET_NAME}_train",
                                lambda: load_coco_json(trn_json_file,
                                                       os.path.join(dataset_root,"train2017"),
                                                       f"{cfg.DATASETS.DATASET_NAME}_train"))
        DatasetCatalog.register(f"{cfg.DATASETS.DATASET_NAME}_test",
                                lambda: load_coco_json(val_json_file,
                                                       os.path.join(dataset_root, "val2017"),
                                                       f"{cfg.DATASETS.DATASET_NAME}_test"))

    else:

        if cfg.DATASETS.TRN_DOUBLE_LOADER:
            if padded_training:
                from maskgnn_utils.dataset.prep_double_padded_dataset import get_padded_dset_dict_double as get_dset_dict_trn
            else:
                from maskgnn_utils.dataset.prep_double_dataset import get_dset_dict_double as get_dset_dict_trn
        else:
            from maskgnn_utils.dataset.prep_single_dataset import get_dset_dict_single as get_dset_dict_trn


        if cfg.DATASETS.VAL_DOUBLE_LOADER:
            if padded_training:
                from maskgnn_utils.dataset.prep_double_padded_dataset import get_padded_dset_dict_double as get_dset_dict_val
            else:
                from maskgnn_utils.dataset.prep_double_dataset import get_dset_dict_double as get_dset_dict_val
        else:
            from maskgnn_utils.dataset.prep_single_dataset import get_dset_dict_single as get_dset_dict_val

        # Register datasets
        f_trn = "train"
        dataset_name = cfg.DATASETS.DATASET_NAME
        DatasetCatalog.register(f"{cfg.DATASETS.DATASET_NAME}_train",
                                lambda f_trn=f_trn: get_dset_dict_trn(dataset_root=dataset_root,
                                                                  dataset_name=dataset_name,
                                                                  json_file=trn_json_file,
                                                                  include_last=False,
                                                                  is_train=True))

        f_val = "test"
        DatasetCatalog.register(f"{cfg.DATASETS.DATASET_NAME}_test",
                                lambda f_val=f_val: get_dset_dict_val(dataset_root=dataset_root,
                                                                  dataset_name=dataset_name,
                                                                  json_file=val_json_file,
                                                                  include_last=True,
                                                                  is_train=False))

    # Set classes.
    if cfg.MODEL.FCOS.NUM_CLASSES == 1:
        MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_train").set(thing_classes=["object"])
        MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_test").set(thing_classes=["object"])
    else:

        if cfg.DATASETS.DATASET_NAME == "ytvis":
            thing_classes = ["person", "giant_panda", "lizard", "parrot", "skateboard", "sedan",
                             "ape", "dog","snake", "monkey", "hand", "rabbit", "duck", "cat",
                             "cow", "fish", "train", "horse", "turtle", "bear", "motorbike",
                             "giraffe", "leopard", "fox", "deer", "owl", "surfboard", "airplane",
                             "truck", "zebra", "tiger", "elephant","snowboard", "boat", "shark",
                             "mouse", "frog", "eagle", "earless_seal", "tennis_racket"]

            MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_train").set(thing_classes=thing_classes)
            MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_test").set(thing_classes=thing_classes)


        elif cfg.DATASETS.DATASET_NAME == "kitti_mots":
            thing_classes = ["person", "car"]
            MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_train").set(thing_classes=thing_classes)
            MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_test").set(thing_classes=thing_classes)


    # Eval type: coco, two_frame_tracking, uvos_writer
    MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_train").set(evaluator_type=cfg.MASKGNN.EVAL_TYPE)
    MetadataCatalog.get(f"{cfg.DATASETS.DATASET_NAME}_test").set(evaluator_type=cfg.MASKGNN.EVAL_TYPE)