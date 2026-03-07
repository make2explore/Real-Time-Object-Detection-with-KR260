#!/bin/bash

# ==========================================================
# KR260 AI Stack Installation Script
# Installs:
#  - PYNQ Framework
#  - Vitis AI Runtime 3.5
#  - DPU-PYNQ Python Interface
#  - Required runtime patches
#
# Tested on:
#  Ubuntu 22.04 (Kria image)
# ==========================================================

set -e

echo "=========================================="
echo "KR260 AI Environment Setup"
echo "=========================================="

cd /home/ubuntu

# ==========================================================
# STEP 1 — Download Vitis AI Runtime Package
# ==========================================================

echo " "
echo "STEP 1: Download Vitis AI Runtime Package"
echo " "

wget -O vai3.5_kr260.zip \
https://www.xilinx.com/bin/public/openDownload?filename=vai3.5_kr260.zip


# ==========================================================
# STEP 2 — Install PYNQ Framework
# ==========================================================

echo " "
echo "STEP 2: Install PYNQ Framework"
echo " "

git clone https://github.com/Xilinx/Kria-PYNQ.git

cd Kria-PYNQ

# KR260 and KV260 share the same K26 SOM
# therefore the KV260 PYNQ stack is used
bash install.sh -b KV260


# ==========================================================
# STEP 3 — Install Vitis AI Runtime Packages
# ==========================================================

echo " "
echo "STEP 3: Install Vitis AI Runtime"
echo " "

cd /home/ubuntu

unzip vai3.5_kr260.zip

pushd vai3.5_kr260/target/runtime_deb/

bash setup.sh

cd ..

tar -xzf lack_lib.tar.gz

sudo cp -r lack_lib/* /usr/lib

popd


# ==========================================================
# STEP 4 — Install DPU-PYNQ Python Package
# ==========================================================

echo " "
echo "STEP 4: Install DPU-PYNQ Python Interface"
echo " "

cd /home/ubuntu

git clone https://github.com/Xilinx/DPU-PYNQ -b design_contest_3.5

cd DPU-PYNQ

# Activate PYNQ virtual environment
source /etc/profile.d/pynq_venv.sh

python3 -m pip install . --no-build-isolation


# ==========================================================
# STEP 5 — Install Example Notebooks
# ==========================================================

echo " "
echo "STEP 5: Install PYNQ DPU Example Notebooks"
echo " "

cd /home/root/jupyter_notebooks

rm -rf pynq-dpu

pynq get-notebooks pynq-dpu -p . --force


# ==========================================================
# STEP 6 — Apply Runtime Fixes
# ==========================================================

echo " "
echo "STEP 6: Apply Runtime Fixes"
echo " "

# Ensure VART can locate required libraries
sudo sed -i -e '$aexport LD_LIBRARY_PATH=/usr/lib' \
/etc/profile.d/pynq_venv.sh

# Allow xdputil to work inside pynq environment
sudo sed -i "s/\/usr\/bin\///g" /usr/bin/xdputil


# ==========================================================
# STEP 7 — Installation Complete
# ==========================================================

echo " "
echo "=========================================="
echo "Installation Completed Successfully"
echo "=========================================="
echo "Please reboot the system before continuing."
echo " "
