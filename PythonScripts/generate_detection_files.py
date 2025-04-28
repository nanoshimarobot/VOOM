import numpy as np
import os
import cv2
import json
import math
import glob
import tqdm
from ultralytics import YOLO

from typing import List, Tuple, Dict
from ultralytics.engine.results import Results

def get_image(ori_path, new_path=None, is_color=True, is_tum=False):
    head, filename = os.path.split(ori_path)
    if is_color:
        if is_tum:
            return cv2.imread(ori_path)
            # return cv2.imread(new_path + filename[:-4] + '.png')
        return cv2.imread(new_path + filename)
    else:
        if is_tum:
            return cv2.imread(ori_path, -1)
        im_depth = cv2.imread(new_path + filename[:-4] + '.png', -1)
        return im_depth

def estimate_mask_contour(mask_box):
    mask_box = (mask_box * 255).astype(np.uint8)
    canny = cv2.Canny(mask_box, 100, 150)
    contours, hierarchies = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    return contour
    # return contours[/]

# filenames = json.load(open('support_files/diamond_dji.json'))

def generate(ds_name: str):
    rgb_files = glob.glob(f"/home/toyozoshimada/sandbox_ws/tum_dataset/{ds_name}/rgb/*.png")
    # rgb_files = glob.glob("/home/toyozoshimada/Downloads/rgbd_dataset_freiburg2_desk/rgb/*.png")
    rgb_files = sorted(rgb_files, key=lambda x: float(x.split('/')[-1].rsplit('.', 1)[0]))

    predictor = YOLO('/home/toyozoshimada/Downloads/yolov8x-seg.pt')
    list_to_save = []
    count = 0
    for rgb_file in tqdm.tqdm(rgb_files):
        count += 1
        #if count == 10:
            #break
        try:
            head, filename = os.path.split(rgb_file)
            dict_per_im = dict()
            dict_per_im["file_name"] = filename
            dict_per_im["detections"] = []
            im_rgb = cv2.imread(rgb_file)
            orig_h, orig_w = im_rgb.shape[:2]
            im_canvas = im_rgb.copy()
            im_rgb = cv2.resize(im_rgb, (640, 480))
            results: List[Results] = predictor(
                im_rgb, save=False #, conf=0.1, device=0, visualize=False, show=False
            )
        except Exception as err:
            print(f"error {err} while processing {rgb_file}")
            raise NotImplementedError("unko")
        if results[0].masks is None:
            list_to_save.append(dict_per_im)
            continue
        boxes = results[0].boxes.to("cpu").numpy()
        masks = results[0].masks.to("cpu").numpy()

        scale_x = orig_w / 640.0
        scale_y = orig_h / 480.0

        for box_, cls, conf, mask in zip(boxes.xyxy, boxes.cls, boxes.conf, masks.data):
            x1, y1, x2, y2 = box_
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y
            box = np.array([x1, y1, x2, y2], dtype=np.float64)

            contour = estimate_mask_contour(mask)
            if contour is None:
                continue
            if len(contour) < 10:
                continue
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (w, h), angle = ellipse
            cx *= scale_x
            cy *= scale_y
            w *= scale_x
            h *= scale_y

            theta = ellipse[2] * math.pi / 180
            ellipse_data = np.array([cx, cy, w, h, angle],
                                    dtype=np.float64)
            category_id = int(cls)
            det = dict()
            det["category_id"] = category_id
            det["detection_score"] = np.float64(conf)
            det["bbox"] = list(box)
            det["ellipse"] = list(ellipse_data)
            dict_per_im["detections"].append(det)

            cv2.rectangle(im_canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0))
            cv2.ellipse(im_canvas, ((cx,cy),(w,h), angle), (0, 0, 255))

        cv2.imwrite(f"/home/toyozoshimada/sandbox_ws/tum_dataset/{ds_name}/test/{rgb_file.split("/")[-1]}", im_canvas)
        list_to_save.append(dict_per_im)
    with open(f'/home/toyozoshimada/sandbox_ws/tum_dataset/{ds_name}/detections_yolov8x_seg_aist_lunchroom_with_ellipse.json', 'w') as outfile:
        json.dump(list_to_save, outfile)

if __name__=="__main__":
    target=[
        # "aist_hallway1_round_with_2person_raw202503131119",
        # "aist_hallway1_round_with_3person_raw202503131123",
        # "aist_hallway1_round_with_4person_raw202503141724",
        "aist_hallway2_round_with_2person_raw202503131131",
        "aist_hallway2_round_with_3person_raw202503131129",
        "aist_hallway2_round_with_4person_raw202503141729",
        "aist_hallway3_round_with_2person_raw202503131134",
        "aist_hallway3_round_with_3person_raw202503141735",
        "aist_hallway3_round_with_4person_raw202503141732"
    ]

    for tag in target:
        print(f"==== Process {tag} ====")
        generate(tag)
        # print(f"=======================")