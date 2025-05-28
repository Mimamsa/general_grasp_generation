#!/bin/bash

PROJECT_ROOT=$(pwd)
DOWNLOAD_SCRIPTS_PATH='MatchAnything/SAM6D/SAM6D/Instance_Segmentation_Model'

function install_dependencies {
    # Install torch first
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

    # Install other dependency
    pip install -r requirements.txt

    # Install the dependency for REGDNet
    cd REGNetv2/REGNetv2/multi_model/utils/pn2_utils 
    python3 setup.py install --user
    cd functions
    python3 setup.py install --user
    cd $PROJECT_ROOT
}

function download_checkpoints {
    # Excute script for SAM, FastSAM, DinoV2
    if test -f $DOWNLOAD_SCRIPTS_PATH/checkpoints/segment-anything/sam_vit_h_4b8939.pth; then
        echo 'File `sam_vit_h_4b8939.pth` exists.'
    else
        python3 $DOWNLOAD_SCRIPTS_PATH/download_sam.py
    fi

    if test -f $DOWNLOAD_SCRIPTS_PATH/checkpoints/FastSAM/FastSAM-x.pt; then
        echo 'File `FastSAM-x.pt` exists.'
    else
        python3 $DOWNLOAD_SCRIPTS_PATH/download_fastsam.py
    fi

    if test -f $DOWNLOAD_SCRIPTS_PATH/checkpoints/dinov2/dinov2_vitl14_pretrain.pth; then
        echo 'File `dinov2_vitl14_pretrain.pth` exists.'
    else
        python3 $DOWNLOAD_SCRIPTS_PATH/download_dinov2.py
    fi

    # link the checkpoints folder to root.
    ln -s $DOWNLOAD_SCRIPTS_PATH/checkpoints $PROJECT_ROOT/checkpoints
    echo "Make a static link: $DOWNLOAD_SCRIPTS_PATH/checkpoints -> $PROJECT_ROOT/checkpoints"
}

install_dependencies
download_checkpoints
