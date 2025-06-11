"""A script to proceed the object grasping flow.
This is an obsolete script, DO NOT use this script to run the application.

References:
[Colored point cloud registration - open3d.org](https://www.open3d.org/docs/latest/tutorial/pipelines/colored_pointcloud_registration.html)

"""


import numpy as np
import open3d as o3d
import cv2
from PIL import Image
import time
from scipy.spatial.transform import Rotation as Rot
import yaml
import copy
import subprocess
import time
import shutil
import os

from REGNetv2.REGNetv2.grasp_detect_from_file_multiobjects import GraspDetector
from REGNetv2.REGNetv2.grasp_detect_from_file_multiobjects import load_config as load_GD_config
from utils.io_utils import load_workflow_config, glb2pcd
from utils.viz_utils import draw_real_gripper


visualize = True


#def rotate_z(angle):
#    c = np.cos(angle)
#    s = np.sin(angle)
#    return np.array([[c,-s, 0],
#                     [s, c, 0],
#                     [0, 0, 1]])

SWAP_AXIS = np.array(
    [[ 0.,  0.,  1., 0.],
     [-1.,  0.,  0., 0.],
     [ 0., -1.,  0., 0.],
     [ 0.,  0.,  0., 1.]])

GRIPPER_CENTER_TO_TCP = np.array(
    [[ 1.,  0.,  0., 0.],
     [ 0.,  1.,  0., 0.],
     [ 0.,  0.,  1., 0.0496],
     [ 0.,  0.,  0., 1.]])


class RobotState():
    def __init__(self, file_path):
 
        self.file_path = file_path

        with open(self.file_path, 'r') as stream:
            try:
                self.load_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.tcp_pos = np.array([self.load_dict['tcp'][0][k] for k in 'xyz'])  # in 'xyz' order, meter
        self.tcp_rot = np.array([self.load_dict['tcp'][0][k] for k in ['euler_x','euler_y','euler_z']])  # in degrees
        self.cam_pos = np.array([self.load_dict['camera'][0][k] for k in 'xyz'])  # in 'xyz' order, meter
        self.cam_rot = np.array([self.load_dict['camera'][0][k] for k in ['euler_x','euler_y','euler_z']])  # in degrees
        self.joints = np.array(self.load_dict['joints'])
        
        self.cam_tx = np.eye(4)
        self.cam_tx[:3,:3] = np.round(Rot.from_euler('xyz', self.cam_rot, degrees=True).as_matrix(), 5)
        self.cam_tx[:3,3] = np.round(self.cam_pos, 5)


def transform_numpy_pcd(tx, pcd):
    """
    Args
        tx (): 4x4 homogeneous array.
        pcd (): Nx6 colored pointcloud
    """
    rot = tx[:3,:3]  # (3x3)
    pos = tx[:3,3]  # (3,)
    
    #print(pcd[:,:3].T.shape)

    pcd[:,:3] = (rot @ pcd[:,:3].T).T + pos
    return pcd


def to_o3d_pcd(pcd):
    """Convert Nx6 pointcloud Open3d pointcloud object.
    Args
        pcd (): 
    """
    ret = o3d.geometry.PointCloud()
    ret.points = o3d.utility.Vector3dVector(pcd[:,:3])
    ret.colors = o3d.utility.Vector3dVector(pcd[:,3:])
    return ret


def to_numpy_pcd(o3d_pcd):
    """
    """
    return np.concatenate([np.asarray(o3d_pcd.points), np.asarray(o3d_pcd.colors)], axis=1)


def main():

    # Move to point #1: mid
    cmds = ['ros2', 'topic', 'pub', '--once', '/urscript_interface/script_command', 'std_msgs/msg/String', '{{data: \"movel(p[{}, {}, {}, {}, {}, {}], a=1.2, v=0.25, r=0)\"}}'.format(-0.4917, -0.1324, 0.5, 2.222, 2.224, 0.002)]
    normal = subprocess.run([*cmds],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
        text=True)
    print(normal.stdout)
    time.sleep(2)

    # Restart save count
    cmds = ['ros2', 'service', 'call', '/multiview_saver/restart_count', 'std_srvs/srv/Trigger', '{}']
    normal = subprocess.run([*cmds],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
        text=True)
    print(normal.stdout)
    time.sleep(1)

    # rename data
    #if os.path.exists('/home/yuhsienc/monoview_data/robot_state_1.yaml') and (not os.path.exists('/home/yuhsienc/monoview_data/demo/robot_state_1.yaml')):
    #    shutil.move('/home/yuhsienc/monoview_data/robot_state_1.yaml', '/home/yuhsienc/monoview_data/demo/robot_state_1.yaml')
    #    shutil.move('/home/yuhsienc/monoview_data/pcd_1.glb', '/home/yuhsienc/monoview_data/demo/pcd_1.glb')
    #    shutil.move('/home/yuhsienc/monoview_data/rgb_img_1.png', '/home/yuhsienc/monoview_data/demo/rgb_img_1.png')

    # Take a shot
    cmds = ['ros2', 'service', 'call', '/multiview_saver/capture_point', 'std_srvs/srv/Trigger', '{}']
    normal = subprocess.run([*cmds],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
        text=True)
    print(normal.stdout)
    time.sleep(2)

    # Move to point #2: left
    #cmds = ['ros2', 'topic', 'pub', '--once', '/urscript_interface/script_command', 'std_msgs/msg/String', '{{data: \"movel(p[{}, {}, {}, {}, {}, {}], a=1.2, v=0.25, r=0)\"}}'.format(-0.40337, -0.36476, 0.36938, -1.86123942, -1.86626076, -0.61672924)]
    #normal = subprocess.run([*cmds],
    #    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    #    check=True,
    #    text=True)
    #print(normal.stdout)
    #time.sleep(4)

    # Take a shot
    #cmds = ['ros2', 'service', 'call', '/multiview_saver/capture_point', 'std_srvs/srv/Trigger', '{}']
    #normal = subprocess.run([*cmds],
    #    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    #    check=True,
    #    text=True)
    #print(normal.stdout)
    #time.sleep(4)

    # Move to point #3: right
    #cmds = ['ros2', 'topic', 'pub', '--once', '/urscript_interface/script_command', 'std_msgs/msg/String', '{{data: \"movel(p[{}, {}, {}, {}, {}, {}], a=1.2, v=0.25, r=0)\"}}'.format(-0.40063, 0.15345, 0.34628, 1.85695284,  1.85690571, -0.60882747)]
    #normal = subprocess.run([*cmds],
    #    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    #    check=True,
    #    text=True)
    #print(normal.stdout)
    #time.sleep(4)

    # Take a shot
    #cmds = ['ros2', 'service', 'call', '/multiview_saver/capture_point', 'std_srvs/srv/Trigger', '{}']
    #normal = subprocess.run([*cmds],
    #    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    #    check=True,
    #    text=True)
    #print(normal.stdout)
    #time.sleep(4)

    # Move to point #1: mid
    #cmds = ['ros2', 'topic', 'pub', '--once', '/urscript_interface/script_command', 'std_msgs/msg/String', '{{data: \"movel(p[{}, {}, {}, {}, {}, {}], a=1.2, v=0.25, r=0)\"}}'.format(-0.4917, -0.1324, 0.5, 2.222, 2.224, 0.002)]
    #normal = subprocess.run([*cmds],
    #    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    #    check=True,
    #    text=True)
    #print(normal.stdout)
    #time.sleep(4)

    # load config
    gd_config = load_GD_config()
    robot_state_files = [
        '/home/yuhsienc/monoview_data/robot_state_1.yaml',
        #'/home/yuhsienc/monoview_data/robot_state_2.yaml',
        #'/home/yuhsienc/monoview_data/robot_state_3.yaml',
        #'/home/yuhsienc/multiview_data/cable_1/robot_state_1.yaml',
        #'/home/yuhsienc/multiview_data/cable_1/robot_state_2.yaml',
        #'/home/yuhsienc/multiview_data/cable_1/robot_state_3.yaml'
    ]
    pcd_files = [
        '/home/yuhsienc/monoview_data/pcd_1.glb',
        #'/home/yuhsienc/monoview_data/pcd_2.glb',
        #'/home/yuhsienc/monoview_data/pcd_3.glb',
        #'/home/yuhsienc/multiview_data/cable_1/pcd_1.glb',
        #'/home/yuhsienc/multiview_data/cable_1/pcd_2.glb',
        #'/home/yuhsienc/multiview_data/cable_1/pcd_3.glb'
    ]
    num_view = len(pcd_files)

    rs = []
    for f in robot_state_files:
        rs.append(RobotState(f))

    # modify camera_transform
    print('-'*20)
    gd_config.camera_transform['euler_deg'] = rs[0].cam_rot.tolist()
    gd_config.camera_transform['translation'] = rs[0].cam_pos.tolist()
    print(gd_config.camera_transform['euler_deg'])
    print(gd_config.camera_transform['translation'])
    print('-'*20)

    # load pointcloud as Open3D pcd object
    o3dPcds = []
    for f in pcd_files:
        numpy_pcd = glb2pcd(f)
        pcd = to_o3d_pcd(numpy_pcd)
        #pcd = pcd.voxel_down_sample(0.01)
        o3dPcds.append(pcd)

    # transform pointcloud (camera frame -> base frame)
    for i in range(num_view):
        o3dPcds[i].transform(rs[i].cam_tx)

    # Colored point cloud registration
    if num_view > 1:
        current_transformation = np.identity(4)

        # estimate normals
        for i in range(num_view):
            o3dPcds[i].estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.01 * 2, max_nn=30))

        result_icp = []
        for i in range(1, num_view):
            result_icp.append(o3d.pipelines.registration.registration_colored_icp(
                o3dPcds[i], o3dPcds[0], 0.01, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                  relative_rmse=1e-6,
                                                                  max_iteration=50))
            )
            o3dPcds[i].transform(result_icp[i-1].transformation)

    #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #o3d.visualization.draw_geometries([mesh, o3dPcds[0], o3dPcds[1], o3dPcds[2]])

    # Merge pointclouds
    o3dPcd = o3dPcds[0]
    if num_view>1:
        for i in range(1,num_view):
            o3dPcd += o3dPcds[i]

    # Create bounding box
    min_bound = (-0.85, -0.48, 0.)  #(-0.75, -0.38, 0.)
    max_bound = (0.10, 0.22, 0.5) #(0., 0.12, 0.5)
    box = o3dPcd.get_axis_aligned_bounding_box()
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    box.color = (1,0,0)
    
    # Crop bounding box
    o3dPcd = o3dPcd.crop(box)
    #o3dPcd = o3dPcd.voxel_down_sample(0.01)


    # Find the plane with the largest support in the point cloud
    plane_model, inliers = o3dPcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = o3dPcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = o3dPcd.select_by_index(inliers, invert=True)


    # visualize pcd
    #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #o3d.visualization.draw_geometries([mesh, o3dPcd])

    # get center
    center = o3dPcd.get_center()
    print('center: ', center)

    # transform pcd (base frame -> camera frame)
    o3dPcd.transform(np.linalg.inv(rs[0].cam_tx))

    # Convert back to numpy (N,6) array
    pcd = to_numpy_pcd(o3dPcd)

    # load grasp detector
    model = GraspDetector(gd_config)

    while True:
        start_time = time.time()

        # generate grasp
        grasps, grasp_info = model.generate_grasp(pcd)
        #del model
        #torch.cuda.empty_cache()

        if grasp_info is None:
            continue

        # select grasp (above table plane, near pointcloud center)
        select_grasps = []
        grasp_top_k = grasp_info["grasp_top_k"]
        for i in range(len(grasp_top_k)):
            if grasp_top_k[i][2,3] > 0.1196:  # z > 7+4.96 cm
                select_grasps.append(grasp_top_k[i])
        
        if len(select_grasps) == 0:
            continue
        
        #ds = []
        #for i in range(len(select_grasps)):
        #    pos = select_grasps[i][:3,3]
        #    d = np.linalg.norm((pos-center), 2)
        #    ds.append(d)
        #max_i = np.argmax(ds)
        #grasps = select_grasps[max_i]
        grasps = select_grasps[0]

        grasp_rot = grasps[:3,:3]
        grasp_pos = grasps[:3,3]
        #grasp_rotvec = Rot.from_matrix(grasp_rot).as_rotvec()
        #print('grasp_pos:\n', grasp_pos)
        #print('grasp_rot:\n', grasp_rot)
        #print('grasp_rotvec:\n', grasp_rotvec)

        tf_grasp_tx = grasps @ SWAP_AXIS @ GRIPPER_CENTER_TO_TCP
        tf_grasp_pos = tf_grasp_tx[:3,3]
        tf_grasp_rot = tf_grasp_tx[:3,:3]

        #tf_grasp_rot = rotate_z(np.pi) @ grasp_rot @ tr
        #tf_grasp_rot = grasp_rot @ SWAP_AXIS
        #tf_grasp_pos = np.array([*grasp_pos[:2], grasp_pos[2]-0.07])  # z-=7 cm
        

        tf_grasp_rotvec = Rot.from_matrix(tf_grasp_rot).as_rotvec()
        tf_grasp_pos = np.round(tf_grasp_pos, 5)
        tf_grasp_rotvec = np.round(tf_grasp_rotvec, 5)
        print('transformed grasp position:\n', tf_grasp_pos)
        print('transformed grasp rotation vector:\n', tf_grasp_rotvec)

        #instruction = 'ros2 topic pub --once /urscript_interface/script_command std_msgs/msg/String \'{{data: \"movel(p[{}, {}, {}, {}, {}, {}], a=1.2, v=0.25, r=0)\"}}\''.format(
        #    float(tf_grasp_pos[0]), float(tf_grasp_pos[1]+0.01), float(tf_grasp_pos[2]-0.08), float(tf_grasp_rotvec[0]), float(tf_grasp_rotvec[1]), float(tf_grasp_rotvec[2])
        #)
        #print(instruction)

        if visualize:
            points = grasp_info["points"][0]
            gpcd = o3d.geometry.PointCloud()
            gpcd.points = o3d.utility.Vector3dVector(points[:, :3])
            gpcd.colors = o3d.utility.Vector3dVector(points[:, 3:])
            geometries = [gpcd]

            # draw top grasps
            gripper = draw_real_gripper(grasp_pos, grasp_rot)
            geometries.extend(gripper)

            # Create bounding box
            min_bound = (-0.75, -0.38, 0.)
            max_bound = (0., 0.12, 0.5)
            box = gpcd.get_axis_aligned_bounding_box()
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            box.color = (1,0,0)
            #geometries.append(box)

            #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            #geometries.append(mesh)

            # draw top-k grasps
            #grasp_top_k = grasp_info["grasp_top_k"]  # (k,4,4)
            #for i in range(len(grasp_top_k)):
            #    i_grasp = grasp_top_k[i]
            #    R = i_grasp[:-1, :3] # discard the homo term
            #    center = i_grasp[:-1, 3] # discard the homo term
            #    gripper = draw_real_gripper(center, R)
            #    geometries.extend(gripper)
            o3d.visualization.draw_geometries(geometries)
        # ==============================

        if len(select_grasps) > 0:
            break


        print(f"generate grasp pose costs: {time.time() - start_time:4f} seconds.")
        print("-------------------")


    # Move to grip
    cmds = ['ros2', 'topic', 'pub', '--once', '/urscript_interface/script_command', 'std_msgs/msg/String', '{{data: \"movel(p[{}, {}, {}, {}, {}, {}], a=1.2, v=0.25, r=0)\"}}'.format(
        float(tf_grasp_pos[0]), float(tf_grasp_pos[1]+0.015), 0.2, float(tf_grasp_rotvec[0]), float(tf_grasp_rotvec[1]), float(tf_grasp_rotvec[2])
    )]
    normal = subprocess.run([*cmds],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
        text=True)
    print(normal.stdout)
    time.sleep(3)

    cmds = ['ros2', 'topic', 'pub', '--once', '/urscript_interface/script_command', 'std_msgs/msg/String', '{{data: \"movel(p[{}, {}, {}, {}, {}, {}], a=1.2, v=0.25, r=0)\"}}'.format(
        float(tf_grasp_pos[0]), float(tf_grasp_pos[1]+0.015), float(tf_grasp_pos[2]-0.085), float(tf_grasp_rotvec[0]), float(tf_grasp_rotvec[1]), float(tf_grasp_rotvec[2])
    )]
    normal = subprocess.run([*cmds],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
        text=True)
    print(normal.stdout)
    time.sleep(3)

    # close gripper
    cmds = ['ros2', 'topic', 'pub', '--once', '/gripper/cmd', 'robotiq_sock_driver_msgs/msg/GripperCmd', '{emergency_release: false, emergency_release_dir: true, stop: false, position: 0, speed: 100, force: 10}']
    normal = subprocess.run([*cmds],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
        text=True)
    print(normal.stdout)
    time.sleep(3)

    # Move to point #1: mid
    cmds = ['ros2', 'topic', 'pub', '--once', '/urscript_interface/script_command', 'std_msgs/msg/String', '{{data: \"movel(p[{}, {}, {}, {}, {}, {}], a=1.2, v=0.25, r=0)\"}}'.format(-0.4917, -0.1324, 0.5, 2.222, 2.224, 0.002)]
    normal = subprocess.run([*cmds],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
        text=True)
    print(normal.stdout)
    time.sleep(3)

    # Move to placement point
    cmds = ['ros2', 'topic', 'pub', '--once', '/urscript_interface/script_command', 'std_msgs/msg/String', '{{data: \"movel(p[{}, {}, {}, {}, {}, {}], a=1.2, v=0.25, r=0)\"}}'.format(-0.45, -0.5, -0.05, 2.222, 2.224, 0.002)]
    normal = subprocess.run([*cmds],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
        text=True)
    print(normal.stdout)
    time.sleep(3)

    # open gripper
    cmds = ['ros2', 'topic', 'pub', '--once', '/gripper/cmd', 'robotiq_sock_driver_msgs/msg/GripperCmd', '{emergency_release: false, emergency_release_dir: true, stop: false, position: 50, speed: 100, force: 10}']
    normal = subprocess.run([*cmds],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
        text=True)
    print(normal.stdout)
    time.sleep(3)


    # Move to point #1: mid
    cmds = ['ros2', 'topic', 'pub', '--once', '/urscript_interface/script_command', 'std_msgs/msg/String', '{{data: \"movel(p[{}, {}, {}, {}, {}, {}], a=1.2, v=0.25, r=0)\"}}'.format(-0.4917, -0.1324, 0.5, 2.222, 2.224, 0.002)]
    normal = subprocess.run([*cmds],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
        text=True)
    print(normal.stdout)


if __name__ == "__main__":
    main()
