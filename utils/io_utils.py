import trimesh
import numpy as np
import argparse
import yaml

def glb2pcd(glb_path:str=None, scene_or_mesh=None) -> np.ndarray:
    """ Convert the data from .glb file to point-cloud with Shape (N, 6).
        Support the mesh input as well.
    Args:
        glb_path: The glb filepath.
        scene_or_path: The trimesh scene object.
    Returns:
        pcd: The point-cloud. Shape (N, 6).
    """
    if scene_or_mesh == None:
        scene_or_mesh = trimesh.load(glb_path)
    points = []
    colors = []

    # 處理 scene 或 mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        geometries = scene_or_mesh.geometry.values()
    else:
        geometries = [scene_or_mesh]

    for geom in geometries:
        verts = geom.vertices
        if geom.visual.kind == 'vertex' and hasattr(geom.visual, 'vertex_colors'):
            col = geom.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        else:
            col = np.ones_like(verts) * 0.5  # 預設灰色

        points.append(verts)
        colors.append(col)

    points = np.vstack(points)
    colors = np.vstack(colors)
    pcd = np.hstack((points, colors))
    return pcd

def rgbd2pcd_mask(rgb:np.ndarray, depth:np.ndarray, mask:np.ndarray, intrinsic_param:tuple) -> np.ndarray:
    """ Perfoming the conversion from rgbd to point-cloud with mask.
    Args:
        rgb: The RGB image. With Shape (W, H, 3).
        depth: The depth image. With Shape (W, H).
        mask: The ROI mask. With Shape (W, H).
        instrinsic_param: Camera's intrinsic parameter. With (cx, cy, fx, fy) order.
    Returns:
        pcd: The point-cloud. Shape (N, 6).
        
    """
    cx = intrinsic_param[0]
    cy = intrinsic_param[1]
    fx = intrinsic_param[2]
    fy = intrinsic_param[3]

    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))

    depth_m = depth.astype(np.float32) / 1000.0  # mm → m
    z = depth_m
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    points = np.stack((x, y, z), axis=2).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    mask_flat = (mask.reshape(-1) > 128)

    valid = (z > 0).reshape(-1) & mask_flat

    points = points[valid]
    colors = colors[valid]

    pcd = np.concatenate([points, colors], axis=1)
    return pcd

def load_workflow_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='workflow.yaml', help='Path to config file')
    cli_args = parser.parse_args()

    # 讀取 YAML
    with open(cli_args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 可以把 dict 轉成 Namespace，如果你想用 args.xxx 風格
    return argparse.Namespace(**config)

    
