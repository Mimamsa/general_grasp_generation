import open3d as o3d
import numpy as np

def create_box(center, size, rot_mat, color):
    box = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
    box.paint_uniform_color(color)
    box.translate(-box.get_center())
    vertices = np.asarray(box.vertices)
    vertices -= np.mean(vertices, axis=0)
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.rotate(rot_mat, center=(0, 0, 0))
    box.translate(center)
    return box

def draw_real_gripper(center, R, base_color_offset=0.0):
    W = 0.050  # gripper width
    D = 0.0455  # gripper depth
    T = 0.0082  # gripper finger thickness
    H = 0.024  # gripper finger width
    offset_y = W / 2
    offset_x = D / 2 + T / 2

    gripper_parts = []
    base_color = [0.5 + base_color_offset, 0.5, 0.5]
    left_color = [0.0, 1.0 - base_color_offset, 0.0]
    right_color = [1.0 - base_color_offset, 0.0, 0.0]

    gripper_parts.append(create_box(center, size=(T, W, H), rot_mat=R, color=base_color))
    left_center = center + R @ np.array([offset_x, offset_y, 0])
    right_center = center + R @ np.array([offset_x, -offset_y, 0])
    gripper_parts.append(create_box(left_center, size=(D, T, H), rot_mat=R, color=left_color))
    gripper_parts.append(create_box(right_center, size=(D, T, H), rot_mat=R, color=right_color))
    return gripper_parts
