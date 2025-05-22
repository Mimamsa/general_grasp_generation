# general_grasp_generation
This is the integration version combined https://github.com/haha20331/Match-Anything and https://github.com/haha20331/REGNetv2
Using SAM6D as the object recognition and segmentation modules for cropping the ROI point-cloud, REGDv2 as the grasping generation module given the point-cloud.
We also support using only point-cloud as input to generate the grasping pose.

## Installation
Environment: Tested under python3.10.12 && torch==2.4.1+cuda12.4
Firstly, create the environment using virtual-env.
```
python3 -m venv venv
source venv/bin/activiate
```
Then install all dependencies and download the models, directly execute the install.sh.
```
bash install.sh
```

## Quick-Start
We offer some scripts for quickly see the effect of the methods.
For users whose GPU VRAM>=8GB, it is recommanded to run under efficient way.
- RGB+Depth frame as input:
```
python3 run_efficiency_workflow_rgbd.py
```
- Point-cloud as input:
```
python3 run_efficiency_workflow_pcd.py
```

For users whose GPU VRAM<8GB:
- RGB+Depth frame as input:
```
python3 run_memory_workflow_rgbd.py
```
- Point-cloud as input:
```
python3 run_memory_workflow_pcd.py
```

## Configuration
WIP

## TODO
- mechanism for mult-target matching.
- collision-free grasping filter
