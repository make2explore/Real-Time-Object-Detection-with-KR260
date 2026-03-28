# Real-Time-Object-Detection-with-KR260
Real-Time Object Detection on the Kria KR260 using a Webcam

<img src="/assets/m2e-Kria-object-thumb.jpg" height="200">  

This repository contains **AI and Computer Vision projects built using the AMD Kria KR260 Robotics Starter Kit**.

These projects demonstrate how to use the **DPU (Deep Processing Unit)** on the KR260 for **hardware-accelerated AI inference**.  
  
**The AMD Kria™ KR260 Robotics Starter Kit** is a high-performance development platform designed to bridge the gap between traditional software development and hardware-accelerated robotics. Centered around the Kria K26 System-on-Module (SOM), it features a Zynq™ UltraScale+™ MPSoC that combines a quad-core ARM processor with programmable logic. This architecture allows roboticists to achieve low-latency, deterministic control and high-speed sensor fusion that standard CPUs cannot match. A standout feature is its native support for ROS 2 (Robot Operating System), which enables developers to implement hardware-accelerated "blocks" for vision and communication without needing deep FPGA expertise.

The board is heavily optimized for industrial networking and machine vision, equipped with four Gigabit Ethernet ports and a 10G SFP+ cage for high-bandwidth 10GigE Vision cameras. For expansion, the kit includes a Raspberry Pi HAT header and four Pmod connectors, allowing for easy integration with a vast ecosystem of sensors and peripherals. Whether you are developing autonomous mobile robots (AMRs) or industrial automation systems, the KR260 provides a production-ready path to move from a desktop prototype to a deployed industrial solution using the same K26 module. 
  
<img src="/assets/kr260-product.jpg" height="200">  

---

# Hardware Requirements

* AMD Kria KR260 Robotics Starter Kit
* USB Webcam (Logitech C270 tested)
* 32GB or larger microSD card
* HDMI monitor
* USB keyboard and mouse
* Ethernet connection or Wi-Fi adapter

---

# Software Requirements

* Ubuntu 22.04 for Kria boards
* PYNQ framework
* Vitis AI Runtime 3.5
* Python 3
* OpenCV

---

# Available Projects

## Real-Time Object Detection using Webcam

This project demonstrates **real-time object detection on the KR260 using a USB webcam and the DPU accelerator**.

Features:

* YOLOX Nano model running on DPU
* Real-time webcam detection
* Object bounding boxes with class labels
* Example vehicle counting application

📂 Project guide available here:

```
object-detection-kr260/
```

---

# KR260 Platform

These projects are designed for the **AMD Kria KR260 Robotics Starter Kit**, which combines:

* ARM Cortex-A53 processor
* Programmable FPGA logic
* Dedicated AI accelerator (DPU)

This architecture enables **high-performance edge AI applications** such as robotics vision and smart cameras.

---

# Future Projects

Additional KR260 projects will be added to this repository.

Planned examples include:

* Multi-camera vision systems
* Fall detection using edge AI
* ROS2-based robotics perception
* Human behavior analysis with AI

------------------------------------------------------------------------------------------------------

📕 **YouTube Video Links**  

▶️ Real-Time Object Detection on the Kria KR260 using a Webcam  🔗  https://youtu.be/  
  
▶️ KR260 Robotics Starter Kit Unboxing - ROS 2 + FPGA Power 🔗  https://youtu.be/dSLpAk4iOcQ  

▶️ KR260 Robotics Starter Kit Headless Setup 🔗  https://youtu.be/S-gdlXVWeXY  
  
▶️ AMD Kria KR260 : ROS2, FPGA Hardware Acceleration #ai #edgeai #ros2 🔗 https://youtube.com/shorts/HO7OIZ3Ha8A  

-------------------------------------------------------------------------------------------------------
📒 **Important Links**  
 
📖 KR260 User Guide :🔗 https://docs.amd.com/r/en-US/ug1092-kr260-starter-kit  
💾 KR260 Applications : 🔗 https://xilinx.github.io/kria-apps-docs/kr260/build/html/index.html  

📌 Kria SOM (System on Modules)  :  🔗 https://www.amd.com/en/products/system-on-modules/kria.html  

🛒  Purchase  -   
AMD Official Website : 🔗 https://www.amd.com/en/products/system-on-modules/kria/k26/kr260-robotics-starter-kit.html  

Product page :  🔗 https://www.amd.com/en/products/system-on-modules/kria/k26/robotics.html  


------------------------------------------------------------------------------------------------------

📜 Source Code, Circuit Diagrams and Documentation : 

🌐 GitHub Repository - 🔗 https://github.com/make2explore/Real-Time-Object-Detection-with-KR260      
  
🌐 Hackster Blog - 🔗 https://www.hackster.io/make2explore  
  
🌐 Instructable Blog - 🔗 https://www.instructables.com/make2explore  
  

------------------------------------------------------------------------------------------  

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg