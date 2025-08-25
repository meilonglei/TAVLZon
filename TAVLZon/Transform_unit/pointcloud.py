import numpy as np
import open3d as o3d
from PIL import Image

def pixel_to_3d(depth, intrinsic,extrinsic):

    filter_z,filter_x = np.where(depth>0)
    depth_values = depth[filter_z,filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2])*depth_values / intrinsic[0][0]
    pixel_y = depth_values
    point_values = np.stack([pixel_x,pixel_z,-pixel_y],axis=-1)


    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]

    x = (filter_x- cx) * depth_values / fx 
    y = (filter_z - cy) * depth_values / fy
    z = depth_values

    points_camera = np.stack((x, y, -z), axis=-1)

    points_world = np.matmul(extrinsic,np.concatenate((point_values,np.ones((point_values.shape[0],1))),axis=-1).T).T
    return points_world[:,0:3]

def points_pointcloud(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    return pcd

def caculate(depth,intrinsic,extrinsic):
    points_world = pixel_to_3d(depth,intrinsic,extrinsic)
    pcd = points_pointcloud(points_world)
    return pcd