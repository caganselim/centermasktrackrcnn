import os
import json
from detectron2.structures import BoxMode
import time

def process_objects(frame_annotations):

    objs = []
    for obj in frame_annotations:
        obj = {
            "bbox": obj["bbox"],
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": obj["segmentation"],
            "category_id": 0,
            "track_id" : obj["id"]
        }
        objs.append(obj)
    return objs

def get_padded_dset_dict_double(dataset_root, json_file, only_n_frames=None,
                                include_last=False, is_davis=False, pad_range=3):

    print("CALLED GET DSET DICT DOUBLE. include last: " , include_last, "is_davis: ", is_davis)

    # Initial setup
    t0 = time.time()
    dataset_dicts = []
    image_id = 0

    with open(json_file) as f:

        print("Dataset root: ", dataset_root)
        print("Opened json file: ", json_file)

        # Load YoutubeVIS dict.
        dset_json = json.load(f)

        # Videos
        for idx, vid_dict in enumerate(dset_json["videos"]):

            # if idx % 500 == 0:
            #     print("Processing video: ", idx)

            seq_len = len(vid_dict["frames"])
            seq_len = seq_len if include_last else seq_len-1

            # say seq_len => 20 frames at most. We drop the last frame during training.
            # then, we end-up with 19 frames as the src. Indexing for the next frames
            # is valid for up-to [0-19] (new_seq_len)

            for i in range(seq_len):

                # Handle the last frame. This mapping is useful while performing evaluation.
                # Not used while training with two frames.
                if include_last and i == (seq_len - 1):

                    frame_dict_0 = vid_dict["frames"][i]
                    frame_name_0 = frame_dict_0["frame_name"]
                    frame_annotations_0 = frame_dict_0["annotations"]

                    # Create a record dict to store info for the consecutive frames
                    if is_davis:

                        record = {"width": vid_dict["width"], "height": vid_dict["height"],
                                  "file_name_0": os.path.join(dataset_root, "JPEGImages/480p", frame_name_0),
                                  "image_id": image_id}
                    else:

                        record = {"width": vid_dict["width"], "height": vid_dict["height"],
                                  "file_name_0": os.path.join(dataset_root, "JPEGImages", frame_name_0),
                                  "image_id": image_id}

                    record["annotations_0"] = process_objects(frame_annotations_0)


                else:

                    # Process frame dictionaries. First; save the current one.
                    frame_dict_0 = vid_dict["frames"][i]
                    frame_name_0 = frame_dict_0["frame_name"]
                    frame_annotations_0 = frame_dict_0["annotations"]

                    # Then, perform the following frames. Check the following statements
                    num_frames_to_be_saved = pad_range if seq_len - i > pad_range else seq_len - i
                    next_frame_infos = []

                    assert num_frames_to_be_saved > 0, "num_frames_to_be_saved must be higher than 0"
                    assert num_frames_to_be_saved <= pad_range, "num_frames_to_be_saved must be less than or equal to pad_range"

                    for k in range(num_frames_to_be_saved):

                        # This will be appended. Prepare an inner record.
                        frame_dict_next = vid_dict["frames"][i+k+1]

                        # Create a record dict to store info for the consecutive frames
                        annotations_next = process_objects(frame_dict_next["annotations"])

                        if is_davis:
                            inner_rec = {"width": vid_dict["width"], "height": vid_dict["height"],
                                         "file_name_1": os.path.join(dataset_root, "JPEGImages/480p", frame_dict_next["frame_name"]),
                                         "annotations_1": annotations_next}
                        else:
                            inner_rec = {"width": vid_dict["width"], "height": vid_dict["height"],
                                         "file_name_1": os.path.join(dataset_root, "JPEGImages", frame_dict_next["frame_name"]),
                                         "annotations_1": annotations_next}

                        # Save in the end
                        next_frame_infos.append(inner_rec)

                    # Process objects and save it
                    record = {"file_name_0": os.path.join(dataset_root, "JPEGImages", frame_name_0),
                              "annotations_0": process_objects(frame_annotations_0),
                              "next_frame_infos": next_frame_infos,
                              "image_id": image_id}

                # Save the record
                dataset_dicts.append(record)

                # Increment image id.
                image_id += 1

    print(f"Elapsed: {time.time() - t0} seconds")

    if only_n_frames is not None:

        return dataset_dicts[:only_n_frames]

    return dataset_dicts


if __name__ == "__main__":

    c = get_padded_dset_dict_double(dataset_root="./YoutubeVIS", json_file="./datasets/jsons/ytvos/ytvos_trn.json")
    #c = get_dset_dict(dataset_root="./datasets/blk30k-100", json_file="./jsons/blk30k/blk30k_trn.json")

    print(len(c))