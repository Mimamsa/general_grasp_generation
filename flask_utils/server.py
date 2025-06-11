""" Inintial Flask
Auther: Jason
Date: 2019/02/12
"""

import sys, os, json
from flask import Flask, request
from flask_utils.log import Log
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
import yaml
from glob import glob

from REGNetv2.REGNetv2.grasp_detect_from_file_multiobjects import GraspDetector
from REGNetv2.REGNetv2.grasp_detect_from_file_multiobjects import load_config as load_GD_config
from utils.io_utils import glb2pcd
from utils.viz_utils import draw_real_gripper


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




class Server:
    __app = Flask(__name__)
    __version = None
    __predicter = None
    __uploadDir = "./uploads/"
    __visualize = True

    @__app.route('/')
    def __index():
        return 'How do you turn this on'

    @__app.route('/dev/predict/test/', methods=['POST'])
    def __test():
        if request.headers['Content-Type']=='application/json':
            Log.info('{}'.format(request.json['name']))
        Log.info('Tested.\n')
        return 'Tested.\n'

    @__app.route('/dev/predict/result/', methods=['POST'])
    def __predict():
        """This function read pointclouds and configs from files, and outputs
        predicted grasps through HTTP POST response.
        """
        ## record start
        time_start = time.time()
        
        result = "No prediction"
        Log.info('__predict init.')

        root_dir = '/home/yuhsienc/monoview_data'

        robot_state_files = sorted(glob(root_dir + '/*.yaml'))
        #pcd_files = sorted(glob(root_dir + '/*.glb'))
        pcd_files = sorted(glob(root_dir + '/*.npy'))
        
        #num_view = len(pcd_files)
        num_view = int(request.json['view'])
        Log.debug('num_view: {}'.format(num_view))

        # load robot state        
        rs = []
        for f in robot_state_files:
            rs.append(RobotState(f))
        Log.debug('Robot states loaded.')

        # modify camera_transform
        Server._gd_config.camera_transform['euler_deg'] = rs[0].cam_rot.tolist()
        Server._gd_config.camera_transform['translation'] = rs[0].cam_pos.tolist()
        Log.debug('Camera_transform modified.')

        # load pointcloud as Open3D pcd object
        o3dPcds = []
        for f in pcd_files:
            #numpy_pcd = glb2pcd(f)
            with open(f, 'rb') as ff:
                numpy_pcd = np.load(ff)
            pcd = to_o3d_pcd(numpy_pcd)
            pcd = pcd.voxel_down_sample(0.01)
            o3dPcds.append(pcd)
        Log.debug('Pointcloud loaded.')

        # transform pointcloud (camera frame -> base frame)
        for i in range(num_view):
            o3dPcds[i].transform(rs[i].cam_tx)
        Log.debug('Pointcloud transformed. (camera frame -> base frame)')

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
            Log.debug('Pointcloud registration done.')

        # Merge pointclouds
        o3dPcd = o3dPcds[0]
        if num_view>1:
            for i in range(1, num_view):
                o3dPcd += o3dPcds[i]
        Log.debug('Pointcloud merge done.')

        # Create bounding box
        min_bound = (-0.85, -0.48, 0.)  #(-0.75, -0.38, 0.)
        max_bound = (0.10, 0.22, 0.5) #(0., 0.12, 0.5)
        box = o3dPcd.get_axis_aligned_bounding_box()
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        box.color = (1,0,0)
        
        # Crop bounding box
        o3dPcd = o3dPcd.crop(box)
        Log.debug('Pointcloud croped.')

        # transform pcd (base frame -> camera frame)
        o3dPcd.transform(np.linalg.inv(rs[0].cam_tx))
        Log.debug('Pointcloud transformed. (base frame -> camera frame)')

        # Convert back to numpy (N,6) array
        pcd = to_numpy_pcd(o3dPcd)

        while True:
            # generate grasp
            grasps, grasp_info = Server._predictor.generate_grasp(pcd)

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

            grasps = select_grasps[0]

            grasp_rot = grasps[:3,:3]
            grasp_pos = grasps[:3,3]

            tf_grasp_tx = grasps @ SWAP_AXIS @ GRIPPER_CENTER_TO_TCP
            tf_grasp_pos = tf_grasp_tx[:3,3] + np.array([0., 0.015, -0.085])
            tf_grasp_rot = tf_grasp_tx[:3,:3]
            tf_grasp_rotvec = Rot.from_matrix(tf_grasp_rot).as_rotvec()
            tf_grasp_pos = np.round(tf_grasp_pos, 5)
            tf_grasp_rotvec = np.round(tf_grasp_rotvec, 5)

            if Server.__visualize:
                points = grasp_info["points"][0]
                gpcd = o3d.geometry.PointCloud()
                gpcd.points = o3d.utility.Vector3dVector(points[:, :3])
                gpcd.colors = o3d.utility.Vector3dVector(points[:, 3:])
                geometries = [gpcd]

                # draw top grasps
                gripper = draw_real_gripper(grasp_pos, grasp_rot)
                geometries.extend(gripper)

                o3d.visualization.draw_geometries(geometries)

            if len(select_grasps) > 0:
                break

        ## record end
        cost = round(time.time()-time_start, 4)

        pred = np.concatenate([tf_grasp_pos, tf_grasp_rotvec])
        result = ','.join(list(map(str, pred)))  # pred[0],pred[1], ...
        Log.info("Result: {}".format(result), attrs=[])
        Log.info('Server time cost (per image): {} sec'.format(cost))

        return result


    def run():
        os.environ["FLASK_ENV"] = "development"

        # Initial GraspDetector
        Server._gd_config = load_GD_config()
        Server._predictor = GraspDetector(Server._gd_config)
        Log.info('GraspDetector is ready.')

        # Create default upload folder if not exist
        if not os.path.isdir(Server.__uploadDir):
            os.makedirs(Server.__uploadDir)

        # Start the server
        Server.__app.run('0.0.0.0', port=5566, debug=False)

        

        
