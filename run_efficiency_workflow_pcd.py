import numpy as np
import open3d as o3d
from MatchAnything.SAM6D.SAM6D.Instance_Segmentation_Model.match_anything import load_config as load_MA_config
from REGNetv2.REGNetv2.grasp_detect_from_file_multiobjects import load_config as load_GD_config
from utils.io_utils import load_workflow_config, glb2pcd
from utils.viz_utils import create_box, draw_real_gripper
from workflow.workflows import EfficiencyWorkflow

visualize = True
if __name__ == "__main__":
    import cv2
    from PIL import Image
    import time
    ma_config = load_MA_config()
    gd_config = load_GD_config()
    workflow_config = load_workflow_config()

    workflow_config.method = "dust3r"
    glb_path = "Example/inputs/dust3r/scene/bottle.glb"
    pcd = glb2pcd(glb_path)

    workflow = EfficiencyWorkflow(ma_config, gd_config, workflow_config)
    while True:
        start_time = time.time()
        grasps, grasp_info = workflow.run(pcd=pcd.copy())
        if grasp_info is None:
            continue
        # ===== For visualization =====
        if visualize:
            points = grasp_info["points"][0]
            gpcd = o3d.geometry.PointCloud()
            gpcd.points = o3d.utility.Vector3dVector(points[:, :3])
            gpcd.colors = o3d.utility.Vector3dVector(points[:, 3:])
            geometries = [gpcd]
            grasp_top_k = grasp_info["grasp_top_k"]
            for i in range(len(grasp_top_k)):
                i_grasp = grasp_top_k[i]
            
                R = i_grasp[:-1, :3] # discard the homo term
                center = i_grasp[:-1, 3] # discard the homo term
                gripper = draw_real_gripper(center, R)
                geometries.extend(gripper)
            o3d.visualization.draw_geometries(geometries)
        # ==============================

        print(f"generate grasp pose costs: {time.time() - start_time:4f} seconds.")
        print("-------------------")