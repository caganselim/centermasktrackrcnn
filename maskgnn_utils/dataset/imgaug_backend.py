import copy
import imgaug
import imgaug.augmenters as iaa
import numpy as np
from datetime import datetime
from detectron2.structures import BoxMode


def process_polygons(annotations_dict, img_shape):
    polygon_list = []

    for idx, annot in enumerate(copy.deepcopy(annotations_dict)):
        # Process each annotation one by one. Get the polygon first
        polygon_in = annot['segmentation']

        polygon_np = [imgaug.augmentables.polys.Polygon(np.asarray(p).reshape(-1, 2), label=idx + 1) for p in
                      polygon_in]
        # Save the polygon input
        polygon_list.extend(polygon_np)

    polygons_out = imgaug.augmentables.PolygonsOnImage(polygon_list, shape=img_shape)

    return polygons_out


def preprocess_annotations(annotations_dict, img_shape):
    # Containers
    polygon_list = []
    bboxes_list = []

    for idx, annot in enumerate(annotations_dict):

        polygon_in = annot['segmentation']
        box_in = annot['bbox']

        # Save bounding-boxes
        box = imgaug.augmentables.BoundingBox(x1=box_in[0], y1=box_in[1],
                                              x2=box_in[2],
                                              y2=box_in[3], label=idx)

        # Save polygons
        polygons = []
        for p in polygon_in:
            p_in = np.asarray(p).reshape(-1, 2)

            pp = imgaug.augmentables.polys.Polygon(p_in, label=idx)
            polygons.append(pp)

        # We have a valid
        bboxes_list.append(box)
        polygon_list.extend(polygons)

    # Merge annotations & filter them.
    polygons_out = imgaug.augmentables.PolygonsOnImage(polygon_list, shape=img_shape)
    bboxes_out = imgaug.augmentables.BoundingBoxesOnImage(bboxes_list, shape=img_shape)

    return polygons_out, bboxes_out


class ImageToSeqAugmenter(object):
    def __init__(self, perspective=False, affine=True, motion_blur=True,
                 brightness_range=(-50, 50), hue_saturation_range=(-15, 15), perspective_magnitude=0.0,
                 scale_range=1.0, translate_range={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, rotation_range=(-20, 20),
                 motion_blur_kernel_sizes=(7, 9), motion_blur_prob=0.5, identity_mode=False, seed_override=None):

        self.identity_mode = identity_mode
        self.seed_override = seed_override

        if self.seed_override is None:
            seed = int(datetime.now().strftime('%M%S%f')[-8:])
        else:
            seed = self.seed_override

        imgaug.seed(seed)

        self.basic_augmenter = iaa.SomeOf((1, None), [
            iaa.Add(brightness_range),
            iaa.AddToHueAndSaturation(hue_saturation_range)
        ])

        transforms = []
        if perspective:
            transforms.append(iaa.PerspectiveTransform(perspective_magnitude))
        if affine:
            transforms.append(iaa.Affine(scale=scale_range,
                                         translate_percent=translate_range,
                                         rotate=rotation_range,
                                         order=1,  # cv2.INTER_LINEAR
                                         backend='auto'))
        transforms = iaa.Sequential(transforms)
        transforms = [transforms]

        if motion_blur:
            blur = iaa.Sometimes(motion_blur_prob, iaa.OneOf(
                [
                    iaa.MotionBlur(ksize)
                    for ksize in motion_blur_kernel_sizes
                ]
            ))
            transforms.append(blur)

        self.frame_shift_augmenter = iaa.Sequential(transforms)

    def __call__(self, image, annotations=None):

        if self.identity_mode:

            if annotations is None:

                return image

            else:

                return image, annotations

        det_augmenter = self.frame_shift_augmenter.to_deterministic()
        height, width, _ = image.shape

        polygons_out, bboxes_out = preprocess_annotations(annotations, (height, width))

        if annotations:

            # to keep track of which points in the augmented image are padded zeros, we augment an additional all-ones
            # array. The problem is that imgaug will apply one augmentation to the image and associated mask, but a
            # different augmentation to this array. To prevent this, we manually seed the rng in imgaug before both
            # function calls to ensure that the same augmentation is applied to both.

            aug_image, aug_polygons, aug_bboxes = det_augmenter(image=self.basic_augmenter(image=image),
                                                                polygons=polygons_out,
                                                                bounding_boxes=bboxes_out)

            try:
                bboxes_out_removed = aug_bboxes.remove_out_of_image(fully=True, partly=False)
                poly_aug_removed = aug_polygons.remove_out_of_image(fully=True, partly=False)
            except:
                print("=> rare error while removing out!")
                bboxes_out_removed = aug_bboxes
                poly_aug_removed = aug_polygons

            # filter out invalid polygons (< 3 points)
            assert len(bboxes_out_removed) != poly_aug_removed, "Bbox vs poly aug is not equal!"
            for idx, annot in enumerate(annotations):
                annotations[idx]["segmentation"] = []

            for bbox in bboxes_out_removed:
                # Recover the index back.
                idx = bbox.label
                box_as_a_list = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
                annotations[idx]["bbox"] = box_as_a_list
                annotations[idx]["bbox_mode"] = BoxMode.XYXY_ABS

            invalid_polygons_indices = set()
            for poly in poly_aug_removed:
                # Recover the index back.
                idx = poly.label
                poly_list = []
                for pt in poly:
                    poly_list.append(pt[0].item())
                    poly_list.append(pt[1].item())

                if len(poly_list) >= 6 and len(poly_list) % 2 == 0:
                    annotations[idx]["segmentation"].append(poly_list)
                else:
                    print(f"[ImgAugBackend] Invalid polygon after data augmentation at index {idx}")
                    invalid_polygons_indices.add(idx)

            # Cleanup
            annotations = [v for i, v in enumerate(annotations) if i not in invalid_polygons_indices]

            return aug_image, annotations

        else:

            aug_image = det_augmenter(image=image)

            return aug_image, annotations
