#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------------------- make2explore.com -------------------------------------------------------#
# Project           - Real-Time Object Detection on Kria KR260 using a Webcam
# Created By        - info@make2explore.com
# Last Modified     - 07/03/2026 17:36:00 @admin
# Software          - Python, PYNQ Framework, Vitis AI Runtime 3.5, DPU-PYNQ Python, OpenCV, YOLOx
# Hardware          - AMD Kria KR260 Robotics Starter Kit.     
# Sensors Used      - External USB WebCam - Logitech C270
# Source Repo       - github.com/make2explore
# ===========================================================================================================#

print(" ")
print("make2explore-Kria_KR260_Object_Detection")
print(" ")

# ***********************************************************************
# Import Packages
# ***********************************************************************
import os
import time
import numpy as np
import cv2
import random
import colorsys
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt

# ***********************************************************************
# input file names
# ***********************************************************************
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "yolox_nano_pt.xmodel")
OVERLAY_PATH = os.path.join(PROJECT_DIR, "overlays", "dpu", "dpu.bit")
LABELS_PATH = os.path.join(PROJECT_DIR, "labels", "coco2017_classes.txt")

# ***********************************************************************
# Prepare the Overlay and load the "cnn.xmodel"
# ***********************************************************************
from pynq_dpu import DpuOverlay
from pynq import Overlay
from pynq.lib import AxiGPIO

print("Loading DPU overlay...")
overlay = DpuOverlay(OVERLAY_PATH)
overlay.load_model(MODEL_PATH)
ol = overlay

dpu = overlay.runner

# ***********************************************************************
# Utility Functions
# ***********************************************************************
def preprocess(image, input_size, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_image = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_image = np.ones(input_size, dtype=np.uint8) * 114

    ratio = min(input_size[0] / image.shape[0],
                input_size[1] / image.shape[1])
    resized_image = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_image = resized_image.astype(np.uint8)

    padded_image[:int(image.shape[0] * ratio), :int(image.shape[1] *
                                                    ratio)] = resized_image
    #padded_image = padded_image.transpose(swap)

    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image, ratio

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def postprocess(
    outputs,
    img_size,
    ratio,
    nms_th,
    nms_score_th,
    max_width,
    max_height,
    p6=False,
):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    predictions = outputs[0]
    boxes = predictions[:, :4]
    scores = sigmoid(predictions[:, 4:5]) * softmax(predictions[:, 5:])
    #scores = predictions[:, 4:5] * predictions[:, 5:]
    
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(
        boxes_xyxy,
        scores,
        nms_thr=nms_th,
        score_thr=nms_score_th,
    )

    bboxes, scores, class_ids = [], [], []
    if dets is not None:
        bboxes, scores, class_ids = dets[:, :4], dets[:, 4], dets[:, 5]
        for bbox in bboxes:
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(bbox[2], max_width)
            bbox[3] = min(bbox[3], max_height)

    return bboxes, scores, class_ids


def nms(boxes, scores, nms_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(
    boxes,
    scores,
    nms_thr,
    score_thr,
    class_agnostic=True,
):
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware

    return nms_method(boxes, scores, nms_thr, score_thr)

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    final_dets = []
    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr

        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = self._nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [
                        valid_boxes[keep], valid_scores[keep, None],
                        cls_inds
                    ],
                    1,
                )
                final_dets.append(dets)

    if len(final_dets) == 0:
        return None

    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr,
                                    score_thr):
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr

    if valid_score_mask.sum() == 0:
        return None

    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)

    dets = None
    if keep:
        dets = np.concatenate([
            valid_boxes[keep],
            valid_scores[keep, None],
            valid_cls_inds[keep, None],
        ], 1)

    return dets

'''Get model classification information'''	
def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
    
class_names = get_class(LABELS_PATH)
num_classes = len(class_names)


# Create color scheme with specific colors for certain object types
colors = [(0, 0, 0)] * num_classes  # Initialize with black

# Dangerous/sharp items - RED
colors[43] = (0, 0, 255)      # knife - BRIGHT RED
colors[44] = (0, 0, 200)      # scissors (if exists)
colors[34] = (0, 50, 255)     # baseball bat - DARK RED

# People - BLUE
colors[0] = (255, 150, 0)     # person - BRIGHT BLUE

# Vehicles - GREEN
colors[2] = (0, 255, 0)       # car - BRIGHT GREEN
colors[3] = (0, 200, 0)       # motorcycle - GREEN
colors[5] = (0, 180, 0)       # bus - DARK GREEN
colors[7] = (0, 160, 0)       # truck - DARK GREEN

# Electronics - CYAN
colors[63] = (255, 255, 0)    # laptop - CYAN
colors[64] = (255, 200, 0)    # mouse - LIGHT CYAN
colors[65] = (255, 150, 0)    # remote - DARK CYAN
colors[66] = (200, 255, 0)    # keyboard - CYAN
colors[67] = (255, 200, 100)  # cell phone - LIGHT CYAN

# Animals - ORANGE
colors[15] = (0, 140, 255)    # bird - ORANGE
colors[16] = (0, 120, 255)    # cat - ORANGE
colors[17] = (0, 100, 255)    # dog - ORANGE

# Sports items - YELLOW
colors[32] = (0, 0, 0)        # sports ball - BLACK
colors[35] = (0, 200, 255)    # baseball glove - YELLOW
colors[36] = (0, 180, 255)    # skateboard - YELLOW

# Furniture - MAGENTA
colors[56] = (255, 0, 255)    # chair - MAGENTA
colors[57] = (200, 0, 200)    # couch - DARK MAGENTA
colors[60] = (255, 100, 255)  # dining table - LIGHT MAGENTA

# Food/drinks - PINK
colors[39] = (180, 0, 255)    # bottle - PINK
colors[41] = (150, 0, 255)    # cup - PINK
colors[46] = (200, 100, 255)  # banana - LIGHT PINK

# For any remaining classes, generate random bright colors
for i in range(num_classes):
    if colors[i] == (0, 0, 0):  # If still black (unassigned)
        # Generate bright, high-contrast color
        import random
        colors[i] = (random.randint(100, 255), 
                     random.randint(100, 255), 
                     random.randint(100, 255))



'''Draw detection frame with labels'''
def draw_bbox(image, bboxes, classes):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    image_h, image_w, _ = image.shape

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.6
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(1.8 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        
        # Draw rectangle
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        
        # Draw label with background
        class_name = classes[class_ind]
        label = f"{class_name}: {score:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (coor[0], coor[1] - text_height - 10),
            (coor[0] + text_width, coor[1]),
            bbox_color,
            -1
        )
        
        # Draw text (white color for visibility)
        cv2.putText(
            image,
            label,
            (coor[0], coor[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            (255, 255, 255),
            2
        )
        
    return image



# ***********************************************************************
# Use VART APIs
# ***********************************************************************

dpu = overlay.runner
inputTensors = dpu.get_input_tensors()

outputTensors = dpu.get_output_tensors()
shapeIn = tuple(inputTensors[0].dims)

shapeOut0 = (tuple(outputTensors[0].dims)) # (1, 52, 52, 85)
shapeOut1 = (tuple(outputTensors[1].dims)) # (1, 26, 26, 85)
shapeOut2 = (tuple(outputTensors[2].dims)) # (1, 13, 13, 85)

outputSize0 = int(outputTensors[0].get_data_size() / shapeIn[0]) # 229840
outputSize1 = int(outputTensors[1].get_data_size() / shapeIn[0]) # 57460
outputSize2 = int(outputTensors[2].get_data_size() / shapeIn[0]) # 14365

input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
output_data = [np.empty(shapeOut0, dtype=np.float32, order="C"), 
               np.empty(shapeOut1, dtype=np.float32, order="C"),
               np.empty(shapeOut2, dtype=np.float32, order="C")]
image = input_data[0]




def run(input_image, section_i, display=False):
    input_shape=(416, 416)
    class_score_th=0.3
    nms_th=0.45
    nms_score_th=0.1

    # Pre-processing
    # print(input_image.shape)
    image_size = input_image.shape[:2]
    image_height, image_width = input_image.shape[0], input_image.shape[1]
    image_data, ratio = preprocess(input_image, input_shape)
    
    # Fetch data to DPU and trigger it
    image[0,...] = image_data.reshape(shapeIn[1:])
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)

    # Decode output from YOLOX-nano
    outputs = np.concatenate([output.reshape(1, -1, output.shape[-1]) for output in output_data], axis=1)
    bboxes, scores, class_ids = postprocess(
        outputs,
        input_shape,
        ratio,
        nms_th,
        nms_score_th,
        image_width,
        image_height,
    )
    
    #Draw boxes into image
    bboxes_with_scores_and_classes = []
    for i in range(len(bboxes)):
        bbox = bboxes[i].tolist() + [scores[i], class_ids[i]]
        bboxes_with_scores_and_classes.append(bbox)
    bboxes_with_scores_and_classes = np.array(bboxes_with_scores_and_classes)
    display = draw_bbox(input_image, bboxes_with_scores_and_classes, class_names)
    
    return class_ids




# Initialize the VideoCapture object using OpenCV directly
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Failed to open the camera.")
else:
    print("The camera opened successfully.")

# Display True/False
display=True

# LED(GPIO)_set
gpio_0_ip = ol.ip_dict['axi_gpio_0']
gpio_out = AxiGPIO(gpio_0_ip).channel1
mask = 0xffffffff

# Initialize variables for Avg_FPS calculation
frame_count = 0
avg_start_time = time.time()

print("Object Detection Started! Press 'q' to quit.")
print(" ")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    # Get the height and width of the image
    height, width, _ = frame.shape

    # No Split the image 640:480
    sections = [
        frame                  #front
    ]

    # Display each section   
    for i, section in enumerate(sections):
        class_ids = run(section, i+1, display)

        if display:
            cv2.imshow(f"Object Detection - KR260", section)
    
    # Sports ball(32)-Led(GPIO)
    if 32 in class_ids:
        gpio_out.write(0x1C,mask) #All_led_on
    else:
        gpio_out.write(0x00,mask) #All_led_off

    if cv2.waitKey(1) & 0xFF == ord('q'):
        gpio_out.write(0x00,mask) #All_GPIO_off
        break
    
    end_time = time.time()
    fps = 1/(end_time - start_time)
    print(f"FPS: {fps:.2f}")

    # Update frame count
    frame_count += 1

    # Check the time every 100 frames
    if frame_count % 100 == 0:
        avg_end_time = time.time()
        elapsed_time = avg_end_time - avg_start_time
        avg_fps = frame_count / elapsed_time
        print(" ")
        print(f"Average FPS over 100 frames: {avg_fps:.2f}")
        print(" ")
        # Reset the timer and frame count
        frame_count = 0
        avg_start_time = time.time()

cap.release()
cv2.destroyAllWindows()

# ***********************************************************************
# Clean up
# ***********************************************************************
del overlay
del dpu
