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
# USB Camera Test
import cv2
import numpy as np
import time

# Open camera using OpenCV directly (not GStreamer)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Failed to open the camera.")
    exit()
else:
    print("The camera opened successfully.")

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive the frame.")
        break
    
    # Display the full frame
    cv2.imshow("Full Frame", frame)
    
    # Count frames
    frame_count += 1
    
    # Calculate and display FPS every 30 frames
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"Frames: {frame_count}, FPS: {fps:.2f}")
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and cleanup
cap.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
print(f"\nTotal frames: {frame_count}")
print(f"Average FPS: {frame_count / total_time:.2f}")
