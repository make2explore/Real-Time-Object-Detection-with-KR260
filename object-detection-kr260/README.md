# Real-Time Object Detection on KR260 using a Webcam

This project demonstrates **hardware-accelerated real-time object detection using the DPU on the AMD Kria KR260 Robotics Starter Kit**.

The system uses a **USB webcam as input** and performs object detection using the **YOLOX Nano model running on the DPU accelerator**.

Pipeline overview:   
<p align="center">
<img src="/assets/m2e-KR260-Project-pipeline.png" height="400">    
</p>  

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

```bash
sudo add-apt-repository ppa:xilinx-apps --yes &&
sudo add-apt-repository ppa:ubuntu-xilinx/default --yes &&
sudo add-apt-repository ppa:xilinx-apps/xilinx-drivers --yes &&
sudo add-apt-repository ppa:lely/ppa --yes &&
sudo apt update &&
sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y
```

This step installs and updates the required repositories used by the Kria platform.

These repositories provide:

* Xilinx platform applications
* Kria hardware drivers
* updated platform packages
* additional dependencies required for PYNQ and Vitis AI

The upgrade process may take several minutes depending on network speed.

Once the system update finishes, continue with the next step to install the AI runtime environment.  


# Step 3 — Install PYNQ and Vitis AI Runtime

Clone this repository to the KR260 system.

```bash
git clone https://github.com/make2explore/Real-Time-Object-Detection-with-KR260
```

Navigate to the project directory:

```bash
cd Real-Time-Object-Detection-with-KR260
```

Navigate to the scripts folder:

```bash
cd object-detection-kr260/scripts
```

Run the installation script:

```bash
chmod +x install_ai_stack.sh
sudo ./install_ai_stack.sh
```

The script installs:

* PYNQ framework
* Vitis AI Runtime 3.5
* DPU-PYNQ interface
* runtime patches required for KR260

Installation typically takes **20–30 minutes** depending on network speed.

---

# Verify Installation

Activate the PYNQ environment:

```bash
source /etc/profile.d/pynq_venv.sh
```

Verify PYNQ installation:

```bash
python3 -c "import pynq; print('PYNQ installed successfully')"
```

Check the DPU runtime:

```bash
python3 -c "import pynq_dpu; print('DPU runtime available')"
```

Example output:

```
DPU runtime available
```

If this command displays DPU runtime available, the installation is successful.

---

# Step 4 — Test the Webcam

Before running AI inference, verify that the webcam works.

Navigate to the project folder:

```bash
cd Real-Time-Object-Detection-with-KR260/object-detection-kr260/examples
```

Run the webcam test script:

```bash
python3 usb-camera-test-opencv.py
```

If successful, a window showing the webcam feed will appear.

Press **q** to exit.

---

# Step 5 — Run Real-Time Object Detection

Activate the PYNQ environment if it is not already active:

```bash
source /etc/profile.d/pynq_venv.sh
```

Run the object detection script:

```bash
sudo -E $(which python3) Kria-KR260-ObjDet.py
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

This repository also includes a **vehicle counting application**.

Run:

```bash
sudo -E $(which python3) vehicle_counter.py
```

Input video file:

```
media/highway.mp4
```

The script performs:

* vehicle detection
* object tracking
* counting vehicles crossing a line
* displaying FPS and vehicle count

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

This implementation uses **OpenCV camera capture instead of GStreamer pipelines** to improve compatibility with standard OpenCV installations.

---

# Troubleshooting

### Camera device busy

Check which process is using the camera:

```bash
sudo lsof /dev/video0
```

Kill the process if necessary.

---

# Performance

Typical observed performance:

YOLOX Nano + DPU
≈ **12–17 FPS**

Performance may vary depending on:

* camera resolution
* system load
* model configuration
