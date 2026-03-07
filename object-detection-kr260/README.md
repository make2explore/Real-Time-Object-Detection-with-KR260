# Real-Time Object Detection on KR260 using a Webcam

This project demonstrates **hardware-accelerated real-time object detection using the DPU on the AMD Kria KR260 Robotics Starter Kit**.

The system uses a **USB webcam as input** and performs object detection using the **YOLOX Nano model running on the DPU accelerator**.

Pipeline overview:

```
Webcam → Frame Capture → Pre-processing → DPU Inference → Object Detection Output
```

---

# Hardware Requirements

* AMD Kria KR260 Robotics Starter Kit
* USB webcam (Logitech C270 tested)
* HDMI monitor
* USB keyboard and mouse
* 32GB or larger microSD card

---

# Software Requirements

* Ubuntu 22.04 for Kria
* PYNQ framework
* Vitis AI Runtime 3.5
* Python 3
* OpenCV

---

# Step 1 — Install the Operating System

Download the official Ubuntu image for Kria boards.

Official download page:

https://ubuntu.com/download/amd-xilinx

Download the **Ubuntu 22.04 image compatible with the KR260**.

## Flash the Image

1. Insert the microSD card into your computer
2. Flash the image using:

   * Balena Etcher
   * Raspberry Pi Imager
3. Insert the microSD card into the KR260
4. Power on the board

Default login:

```
username: ubuntu
password: ubuntu
```

---

# Step 2 — Install PYNQ and Vitis AI Runtime

Create the installation script:

```
scripts/install_ai_stack.sh
```

Run the installation:

```
chmod +x scripts/install_ai_stack.sh
sudo ./scripts/install_ai_stack.sh
```

The script installs:

* PYNQ framework
* Vitis AI Runtime
* DPU-PYNQ interface
* Example notebooks

Installation typically takes **20–30 minutes**.

---

# Step 3 — Verify DPU Installation

After installation, reboot the board.

Check the DPU runtime:

```
xdputil query
```

If successful, information about the DPU architecture will be displayed.

---

# Step 4 — Test the Webcam

Before running AI inference, verify the webcam works.

Run:

```
python3 examples/usb-camera-test-opencv.py
```

If successful, a window showing the webcam feed will appear.

Press **q** to exit.

---

# Step 5 — Run Real-Time Object Detection

Run the main object detection script:

```
python3 examples/Kria-KR260-ObjDet.py
```

The program will:

1. Capture frames from the webcam
2. Run inference on the DPU
3. Display detected objects with bounding boxes
4. Print FPS in the terminal

Typical performance:

```
12 – 17 FPS
```

---

# Step 6 — Vehicle Counting Example

This project also includes a **vehicle counting application**.

Run:

```
python3 examples/vehicle_counter.py
```

Input video:

```
media/highway.mp4
```

The script performs:

* vehicle detection
* object tracking
* counting vehicles crossing a line
* real-time FPS display

Vehicle classes counted:

* Car
* Motorcycle
* Bus
* Truck

---

# Project Structure

```
object-detection-kr260
│
├── scripts
│   └── install_ai_stack.sh
│
├── overlays
│   └── dpu
│       ├── dpu.bit
│       ├── dpu.hwh
│       └── dpu.xclbin
│
├── models
│   └── yolox_nano_pt.xmodel
│
├── examples
│   ├── usb-camera-test-opencv.py
│   ├── Kria-KR260-ObjDet.py
│   └── vehicle_counter.py
│
└── media
    └── highway.mp4
```

---

# Important Note

This implementation uses **OpenCV camera capture instead of GStreamer pipelines** for improved compatibility with standard OpenCV installations.

---

# Troubleshooting

### Camera device busy

Check which process is using the camera:

```
sudo lsof /dev/video0
```

Kill the process if necessary.

---

# Performance

Typical performance observed:

```
YOLOX Nano + DPU
≈ 12–17 FPS
```

Performance may vary depending on:

* camera resolution
* system load
* model configuration

---
