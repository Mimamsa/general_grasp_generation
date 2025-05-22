#!/bin/bash

# Install torch first
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Install other dependency
pip install -r requirements.txt

# Install the dependency for REGDNet
cd REGNetv2/REGNetv2/multi_model/utils/pn2_utils 
python3 setup.py install
cd functions
python3 setup.py install
cd ../../../../../..


# Excute script for SAM, FastSAM, DinoV2
cd MatchAnything/SAM6D/SAM6D/Instance_Segmentation_Model
python download_sam.py
python download_fastsam.py
python download_dinov2.py

# copy the checkpoints folder to root.
cp -r checkpoints ../../../../checkpoints
cd ../../../..
