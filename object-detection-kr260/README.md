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
* HDMI monitor (Optional), you can use [NoMachine Headless Remote Desktop Setup](https://github.com/make2explore/AMD-Kria-KR260-Robotics-Starter-Kit/tree/main/Installing-NoMachine)
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

# Step 2 — Update Base System

After booting the KR260 for the first time, update the system repositories and install the required AMD/Xilinx package sources.

Run the following commands:

```id="a2sys1"
sudo add-apt-repository ppa:xilinx-apps --yes &&
sudo add-apt-repository ppa:ubuntu-xilinx/default --yes &&
sudo add-apt-repository ppa:xilinx-apps/xilinx-drivers --yes &&
sudo add-apt-repository ppa:lely/ppa --yes &&
sudo apt update --yes &&
sudo apt upgrade --yes
```

This step installs and updates the required package repositories used by the Kria platform.

These repositories provide:

* Xilinx platform applications
* Kria-specific drivers
* updated hardware support packages
* additional system dependencies required by PYNQ and Vitis AI

This process may take several minutes depending on your internet connection.

After the update finishes, continue to the next step to install the AI runtime environment.


# Step 3 — Install PYNQ and Vitis AI Runtime

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

# Step 4 — Verify DPU Installation

After installation, reboot the board.

Check the DPU runtime:

```
xdputil query
```

If successful, information about the DPU architecture will be displayed.

---

# Step 5 — Test the Webcam

Before running AI inference, verify the webcam works.

Run:

```
python3 examples/usb-camera-test-opencv.py
```

If successful, a window showing the webcam feed will appear.

Press **q** to exit.

---

# Step 6 — Run Real-Time Object Detection

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

# Step 7 — Vehicle Counting Example

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
