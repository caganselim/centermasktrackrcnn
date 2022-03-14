# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# NUM CLASSES TRICK
_C.MODEL.ROI_HEADS.NUM_CLASSES = 40
_C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 40
_C.MODEL.RETINANET.NUM_CLASSES = 40
_C.MODEL.FCOS.NUM_CLASSES = 40

_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'


# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.VOVNET = CN()

_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256
_C.MODEL.VOVNET.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.VOVNET.WITH_MODULATED_DCN = False
_C.MODEL.VOVNET.DEFORMABLE_GROUPS = 1

# ---------------------------------------------------------------------------- #
# CenterMask
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD.ASSIGN_CRITERION = "area"
_C.MODEL.MASKIOU_ON = False
_C.MODEL.MASKIOU_LOSS_WEIGHT = 1.0

_C.MODEL.ROI_MASKIOU_HEAD = CN()
_C.MODEL.ROI_MASKIOU_HEAD.NAME = "MaskIoUHead"
_C.MODEL.ROI_MASKIOU_HEAD.CONV_DIM = 256
_C.MODEL.ROI_MASKIOU_HEAD.NUM_CONV = 4

# ---------------------------------------------------------------------------- #
# Keypoint Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_KEYPOINT_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.ROI_KEYPOINT_HEAD.ASSIGN_CRITERION = "ratio"

# -----------------------------<----------------------------------------------- #
# MATCHER
# ---------------------------------------------------------------------------- #
_C.MODEL.MATCHER = CN()
_C.MODEL.MATCHER.NAME = "ObjectMatcher"
_C.MODEL.MATCHER.IS_GNN_ON = True

_C.MODEL.MATCHER.GNN = CN()
_C.MODEL.MATCHER.GNN.INPUT_DIM = 12
_C.MODEL.MATCHER.GNN.HIDDEN_DIM = 128
_C.MODEL.MATCHER.GNN.ACTION_DIM = 4
_C.MODEL.MATCHER.OBJ_ENCODER = CN()
_C.MODEL.MATCHER.OBJ_ENCODER.NAME = "EncoderMLP"
_C.MODEL.MATCHER.OBJ_ENCODER.ENCODER_MLP = CN()
_C.MODEL.MATCHER.OBJ_ENCODER.ENCODER_MLP.INPUT_DIM = 28*28
_C.MODEL.MATCHER.OBJ_ENCODER.ENCODER_MLP.HIDDEN_DIM = 128
_C.MODEL.MATCHER.OBJ_ENCODER.ENCODER_MLP.OUTPUT_DIM = 8
_C.MODEL.MATCHER.OBJ_ENCODER.ENCODER_CNN = CN()
_C.MODEL.MATCHER.OBJ_ENCODER.ENCODER_CNN.INPUT_DIM = 256
_C.MODEL.MATCHER.OBJ_ENCODER.ENCODER_CNN.HIDDEN_DIM = 64
_C.MODEL.MATCHER.OBJ_ENCODER.ENCODER_CNN.OUTPUT_DIM = 8

# ---------------------------------------------------------------------------- #
# FREEZE - to be modified.
# ---------------------------------------------------------------------------- #
_C.MODEL.FREEZE = CN()
_C.MODEL.FREEZE.BACKBONE = False
_C.MODEL.FREEZE.PROPOSAL_GENERATOR = False
_C.MODEL.FREEZE.ROI_HEADS = True
_C.MODEL.FREEZE.GNN = False
_C.MODEL.FREEZE.OBJ_ENCODER = False

_C.MASKGNN = CN()
_C.MASKGNN.PADDED_TRAINING_ON = False
_C.MASKGNN.TRAIN_GNN_ONLY = False
_C.MASKGNN.FREEZE_CLS_HEADS = False
_C.MASKGNN.TRACKER_MODE= "maskiou"
_C.MASKGNN.EVAL_TYPE= "uvos_writer"
_C.MASKGNN.FLOW_ENABLED = False
_C.MASKGNN.FULLY_UNSUPERVISED= False
_C.MASKGNN.VIS_TRAIN= False
_C.MASKGNN.SCORE_THRESH_TEST= 0.2
_C.MASKGNN.MASK_FEATURE_SRC = "logits"

_C.MASKGNN.LOSSES = CN()
_C.MASKGNN.LOSSES.FCOS_CLS = True
_C.MASKGNN.LOSSES.FCOS_LOC = True
_C.MASKGNN.LOSSES.FCOS_CTR = True
_C.MASKGNN.LOSSES.TRACKING = True
_C.MASKGNN.LOSSES.MASK = True
_C.MASKGNN.LOSSES.MASKIOU = True
_C.MASKGNN.LOSSES.CONTRASTIVE = True

#Coco motion params
_C.COCO_MOTION_AUG = CN()
_C.COCO_MOTION_AUG.PERSPECTIVE = False
_C.COCO_MOTION_AUG.AFFINE = True
_C.COCO_MOTION_AUG.BRIGHTNESS_RANGE = (-50, 50)
_C.COCO_MOTION_AUG.HUE_SATURATION_RANGE = (-15, 15)
_C.COCO_MOTION_AUG.PERSPECTIVE_MAGNITUDE = 0.0
_C.COCO_MOTION_AUG.SCALE_RANGE = 1.0
_C.COCO_MOTION_AUG.TRANSLATE_RANGE_X = (-0.15, 0.15)
_C.COCO_MOTION_AUG.TRANSLATE_RANGE_Y = (-0.15, 0.15)
_C.COCO_MOTION_AUG.ROTATION_RANGE = (-20, 20)
_C.COCO_MOTION_AUG.MOTION_BLUR = True
_C.COCO_MOTION_AUG.MOTION_BLUR_KERNEL_SIZES = (7,9)
_C.COCO_MOTION_AUG.MOTION_BLUR_PROB = 0.5
_C.COCO_MOTION_AUG.IDENTITY_MODE = False
_C.COCO_MOTION_AUG.SEED_OVERRIDE = None


