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
We mainly have three config files, `MA_config.yaml`, `config.yaml` and `workflow.yaml`. 
`MA_config.yaml` is for Macthing+Segment.
`config.yaml` is for grasp pose generation, including some camera settings, i.e., extrinsic parameters.
`workflow.yaml` is for workflow settings, you will see there are some duplicate parameters which appear in the above two files, Note that only `method` and `fx`, `fy`, `cx` and `cy` will affect.

In the future, we can make it all in one.


## TODO
- mechanism for mult-target matching.
- collision-free grasping filter
- mask all configs in one.
