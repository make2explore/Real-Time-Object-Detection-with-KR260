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
# Vehicle Counter Code

print("KR260 Vehicle Counter")
print(" ")

import os
import cv2
import time
import numpy as np
from collections import OrderedDict
from pynq_dpu import DpuOverlay

# ==========================================================
# Load DPU Overlay + Model
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

OVERLAY_PATH = os.path.join(PROJECT_DIR, "overlays", "dpu", "dpu.bit")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "yolox_nano_pt.xmodel")

overlay = DpuOverlay(OVERLAY_PATH)
overlay.load_model(MODEL_PATH)
dpu = overlay.runner

inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()

shapeIn = tuple(inputTensors[0].dims)
shapeOut0 = tuple(outputTensors[0].dims)
shapeOut1 = tuple(outputTensors[1].dims)
shapeOut2 = tuple(outputTensors[2].dims)

input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
output_data = [
    np.empty(shapeOut0, dtype=np.float32, order="C"),
    np.empty(shapeOut1, dtype=np.float32, order="C"),
    np.empty(shapeOut2, dtype=np.float32, order="C"),
]

image = input_data[0]

# ==========================================================
# Preprocess
# ==========================================================
def preprocess(frame, input_size):
    padded = np.ones((input_size[0], input_size[1], 3),
                     dtype=np.uint8) * 114

    ratio = min(input_size[0] / frame.shape[0],
                input_size[1] / frame.shape[1])

    resized = cv2.resize(
        frame,
        (int(frame.shape[1] * ratio),
         int(frame.shape[0] * ratio))
    )

    padded[:resized.shape[0], :resized.shape[1]] = resized
    padded = np.ascontiguousarray(padded, dtype=np.float32)

    return padded, ratio

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ==========================================================
# YOLOX Postprocess (Correct Decode)
# ==========================================================
def postprocess(outputs, img_size, ratio, score_th):

    strides = [8, 16, 32]
    grids = []
    expanded_strides = []

    hsizes = [img_size[0] // s for s in strides]
    wsizes = [img_size[1] // s for s in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(
            np.full((*grid.shape[:2], 1), stride)
        )

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    predictions = outputs[0]

    boxes = predictions[:, :4]
    obj = sigmoid(predictions[:, 4:5])
    cls = sigmoid(predictions[:, 5:])

    scores = obj * cls

    cls_ids = scores.argmax(axis=1)
    cls_scores = scores[np.arange(len(scores)), cls_ids]

    mask = cls_scores > score_th

    boxes = boxes[mask]
    cls_ids = cls_ids[mask]

    boxes_xyxy = np.zeros_like(boxes)

    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    boxes_xyxy /= ratio

    return boxes_xyxy, cls_ids

# ==========================================================
# Inference
# ==========================================================
def run(frame):

    input_shape = (416, 416)
    score_th = 0.25

    image_data, ratio = preprocess(frame, input_shape)

    image[0, ...] = image_data.reshape(shapeIn[1:])
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)

    outputs = np.concatenate(
        [out.reshape(1, -1, out.shape[-1])
         for out in output_data],
        axis=1
    )

    bboxes, class_ids = postprocess(
        outputs, input_shape, ratio, score_th
    )

    return bboxes, class_ids

# ==========================================================
# Stable Centroid Tracker
# ==========================================================
class CentroidTracker:

    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):

        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.linalg.norm(
                np.array(objectCentroids)[:, None] - inputCentroids,
                axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

# ==========================================================
# MAIN
# ==========================================================
vehicle_classes = [2, 3, 5, 7]
VIDEO_PATH = os.path.join(PROJECT_DIR, "media", "highway.mp4")

cap = cv2.VideoCapture(VIDEO_PATH)
ct = CentroidTracker()

totalCount = 0
objectMemory = {}
countedIDs = set()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    H = frame.shape[0]
    lineA = int(H * 0.45)
    lineB = 260

    start = time.time()

    bboxes, class_ids = run(frame)

    rects = []

    for i in range(len(bboxes)):
        if int(class_ids[i]) in vehicle_classes:
            x1, y1, x2, y2 = bboxes[i].astype(int)
            rects.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

    objects = ct.update(rects)

    for objectID, centroid in objects.items():

        cX, cY = centroid
        prevY = objectMemory.get(objectID)

        if prevY is not None:

            if prevY is not None:
                # Count when vehicle crosses RED line downward
                if prevY < lineB and cY >= lineB:
                    if objectID not in countedIDs:
                        totalCount += 1
                        countedIDs.add(objectID)

        objectMemory[objectID] = cY

    cv2.line(frame, (0, lineA), (640, lineA),
             (255, 0, 0), 2)
    cv2.line(frame, (0, lineB), (640, lineB),
             (0, 0, 255), 2)

    cv2.putText(frame,
                f"Total Vehicles: {totalCount}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    fps = 1 / (time.time() - start)
    cv2.putText(frame,
                f"FPS: {fps:.2f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2)

    cv2.imshow("Vehicle Counter - KR260", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()