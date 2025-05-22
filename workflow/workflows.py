import os, sys
sys.path.append(os.getcwd())
from utils.io_utils import glb2pcd, rgbd2pcd_mask, load_workflow_config
from MatchAnything.SAM6D.SAM6D.Instance_Segmentation_Model.match_anything import ObjectQueryHelper
from MatchAnything.SAM6D.SAM6D.Instance_Segmentation_Model.match_anything import load_config as load_MA_config
from REGNetv2.REGNetv2.grasp_detect_from_file_multiobjects import GraspDetector
from REGNetv2.REGNetv2.grasp_detect_from_file_multiobjects import load_config as load_GD_config
import numpy as np
import torch

class WorkflowBase():
    def __init__(
            self,
            ma_config,
            gd_config,
            workflow_config,
            device=None
    ):
        self.ma_config = ma_config
        self.gd_config = gd_config
        self.workflow_config = workflow_config
        self.method = workflow_config.method
        if self.method not in ["match", "dust3r"]:
            raise Exception(f"The arg 'method' accepted only 'match' or 'dust3r' so far, but got {self.method}.")
        self.intrinsic_param = workflow_config.cx, workflow_config.cy, workflow_config.fx, workflow_config.fy
        self.device = device

    def _load_model(self, func_name:str):
        if func_name == "match":
            model = ObjectQueryHelper(self.ma_config)
        
        elif func_name == "grasp":
            model = GraspDetector(self.gd_config)
        return model
    
    def run(self):
        raise NotImplementedError
    
class EfficiencyWorkflow(WorkflowBase):
    def __init__(
            self,
            ma_config,
            gd_config,
            workflow_config,
            device=None
    ):
        super().__init__(ma_config, gd_config, workflow_config, device)
        self.match_helper = self._load_model("match")
        self.grasp_helper = self._load_model("grasp")

    def run(self, rgb:np.ndarray=None, depth:np.ndarray=None, pcd:np.ndarray=None) -> tuple:
        """ Conducting the workflow, sequentially doing the following process.
            1. match and get mask for the queries.
            2. get the roi pointcloud.
            3. generate the grasping proposal given pointcloud. 
            If the pcd is given, only step 3 would be executed.
        Args:
            rgb: The rgb frame. Shape : (w, h, 3).
            depth: The depth frame. Shape: (w, h).
            pcd: The point-cloud. Shape (N, 6), with color information.
        Returns:
            grasps: The best grasping pose.
            grasp_info: The details of grasping info. containing point-cloud, score, top-k grasping.
            
        """
        if (self.method == "match") and ((rgb is None) or (depth is None)):
            raise Exception("Must provide rgb and depth when method is 'match'.")
        if (self.method == "dust3r") and (pcd is None):
            raise Exception("Must provide pcd when method is 'dust3r'.")
        
        if self.method == "match":
            query = self.match_helper.query(rgb)[0]
            # for query in queries:
            mask = query["mask"].astype(np.float32) * 255
            pcd = rgbd2pcd_mask(rgb, depth, mask, self.intrinsic_param)
            grasps, grasp_info = self.grasp_helper.generate_grasp(pcd)
            
        if self.method == "dust3r":
            grasps, grasp_info = self.grasp_helper.generate_grasp(pcd)
        return grasps, grasp_info
    
class MemoryWorkflow(WorkflowBase):
    def __init__(
            self,
            ma_config,
            gd_config,
            workflow_config,
            device=None
    ):
        super().__init__(ma_config, gd_config, workflow_config, device)
    
    def run(self, rgb:np.ndarray=None, depth:np.ndarray=None, pcd:np.ndarray=None):
        """ Conducting the workflow, sequentially doing the following process.
            1. match and get mask for the queries.
            2. get the roi pointcloud.
            3. generate the grasping proposal given pointcloud. 
            If the pcd is given, only step 3 would be executed.
        Args:
            rgb: The rgb frame. Shape : (w, h, 3).
            depth: The depth frame. Shape: (w, h).
            pcd: The point-cloud. Shape (N, 6), with color information.
        Returns:
            grasps: The best grasping pose.
            grasp_info: The details of grasping info. containing point-cloud, score, top-k grasping.
            
        """
        if (self.method == "match") and ((rgb is None) or (depth is None)):
            raise Exception("Must provide rgb and depth when method is 'match'.")
        if (self.method == "dust3r") and (pcd is None):
            raise Exception("Must provide pcd when method is 'dust3r'.")
        
        if self.method == "match":
            match_helper = self._load_model("match")
            queries = match_helper.query(rgb)
            del match_helper
            torch.cuda.empty_cache()

            grasp_helper = self._load_model("grasp")
            for query in queries:
                mask = query["mask"].astype(np.float32) * 255
                pcd = rgbd2pcd_mask(rgb, depth, mask, self.intrinsic_param)
                grasps, grasp_info = grasp_helper.generate_grasp(pcd)
            
            del grasp_helper
            torch.cuda.empty_cache()

        if self.method == "dust3r":
            grasp_helper = self._load_model("grasp")
            grasps, grasp_info = grasp_helper.generate_grasp(pcd)
            del grasp_helper
            torch.cuda.empty_cache()
        return grasps, grasp_info
   