from __future__ import division

import numpy as np
import cv2
import skimage
import skimage.transform
from detectron2.structures import BoxMode

import skimage.transform
import numpy as np
import math
from functools import wraps
from warnings import warn
from itertools import product
import cv2
import numpy as np
import skimage
from typing import Sequence, Optional, Union


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


def _get_aug_param(c, s, width, height, disturb=False, scale=0.05, shift=0.05):

    if not disturb:

        aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = _get_border(128, width)
        h_border = _get_border(128, height)  # If smaller than 256. div by 2
        c[0] = np.random.randint(low=w_border, high=width - w_border)
        c[1] = np.random.randint(low=h_border, high=height - h_border)

    else:

        sf = scale
        cf = shift

        c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

    # Disabled rotation thing.
    rot = 0

    return c, aug_s, rot


def post_process_annotations_v2(annotations, trans_output, canvas_height, canvas_width):
    annos = []
    for idx, annotation in enumerate(annotations):

        if annotation.get("iscrowd", 0) == 0:

            bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)

            bbox_tl_in = np.array([[bbox[0], bbox[1]]])
            bbox_br_in = np.array([[bbox[2], bbox[3]]])

            box_tl = skimage.transform.matrix_transform(bbox_tl_in, trans_output.params)[0]
            box_br = skimage.transform.matrix_transform(bbox_br_in, trans_output.params)[0]

            xmin, ymin = box_tl[0], box_tl[1]
            xmax, ymax = box_br[0], box_br[1]

            xmin = np.clip(xmin, 0, canvas_width)
            ymin = np.clip(ymin, 0, canvas_height)
            xmax = np.clip(xmax, 0, canvas_width)
            ymax = np.clip(ymax, 0, canvas_height)

            annotation["bbox"] = [xmin, ymin, xmax, ymax]
            annotation["bbox_mode"] = BoxMode.XYXY_ABS

            if "segmentation" in annotation:
                # each instance contains 1 or more polygons
                segm = annotation["segmentation"]
                if isinstance(segm, list):
                    # polygons
                    polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                    tmp = []
                    for p in polygons:
                        points = skimage.transform.matrix_transform(p, trans_output.params)

                        # Clip?
                        points[:, 0] = np.clip(points[:, 0], 0, canvas_width)
                        points[:, 1] = np.clip(points[:, 1], 0, canvas_height)

                        tmp.append(points.reshape(-1))

                    annotation["segmentation"] = tmp

            annos.append(annotation)

    return annos


def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1


def _is_identity_matrix(matrix: skimage.transform.ProjectiveTransform) -> bool:
    return np.allclose(matrix.params, np.eye(3, dtype=np.float32))

def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    @wraps(process_fn)
    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn



def warp_affine(
    image: np.ndarray,
    matrix: skimage.transform.ProjectiveTransform,
    interpolation: int,
    cval: Union[int, float, Sequence[int], Sequence[float]],
    mode: int,
    output_shape: Sequence[int],
) -> np.ndarray:
    if _is_identity_matrix(matrix):
        return image

    dsize = int(np.round(output_shape[1])), int(np.round(output_shape[0]))
    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix.params[:2], dsize=dsize, flags=interpolation, borderMode=mode, borderValue=cval
    )
    tmp = warp_fn(image)


    return tmp
