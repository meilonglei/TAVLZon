import habitat
from PIL import Image
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from omegaconf import DictConfig, OmegaConf,open_dict
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import LookUpActionConfig,LookDownActionConfig
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import cv2
import os
import inspect
from Transform_unit.camera_intrinsic import habitat_camera_intrinsic
current_dir = os.getcwd()
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from Transform_unit import camera_intrinsic
from Transform_unit import camera_extrinsic
from Transform_unit import pointcloud
from Transform_unit import transform
from Transform_unit import preprocess
from version_merge import *


class Instruct_Mapper:
    def __init__(self,
                 camera_intrinsic,
                 pcd_resolution=0.05,
                 grid_resolution=0.25,
                 floor_height_down=-0.2,
                 floor_height_up = 0.2,
                 ceiling_height=1.2,
                 translation_func=transform.habitat_translation,
                 rotation_func=transform.habitat_rotation,
                 rotate_axis=[0,1,0],
                 ):
        self.camera_intrinsic = camera_intrinsic
        self.pcd_resolution = pcd_resolution
        self.grid_resolution = grid_resolution
        self.floor_height_down = floor_height_down
        self.floor_height_up = floor_height_up
        self.ceiling_height = ceiling_height
        self.translation_func = translation_func
        self.rotation_func = rotation_func

    def reset(self,position,rotation):
        self.current_position = self.translation_func(position)
        self.current_rotation = self.rotation_func(rotation)
        self.scene_pcd = o3d.geometry.PointCloud()
        self.navigable_pcd = o3d.geometry.PointCloud()
        self.obstacle_pcd = o3d.geometry.PointCloud()
        self.height = 0.8
        self.frontier_points = np.empty((0, 3)) 
        self.sum_points = np.empty((0, 3))
        self.frontier_points_ed = np.empty((0, 3))
        self.current_navigable_pcd = o3d.geometry.PointCloud()
        self.current_obstacle_pcd = o3d.geometry.PointCloud()
        self.threshold_1 = 1.1
        self.threshold_2 = 1.1



    def update_single(self,rgb,depth,position,rotation):
        self.current_position = self.translation_func(position)
        self.current_rotation = self.rotation_func(rotation)
        depth = depth.squeeze()
        self.current_depth = preprocess.preprocess_depth(depth)   
        self.current_rgb = preprocess.preprocess_image(rgb)       

        extrinsic=camera_extrinsic.habitat_camera_extrinsic(self.current_position,self.current_rotation)
        self.current_pcd = pointcloud.caculate(self.current_depth,self.camera_intrinsic,extrinsic).voxel_down_sample(self.pcd_resolution)
        self.scene_pcd = self.current_pcd + self.scene_pcd
        self.scene_pcd = self.scene_pcd.voxel_down_sample(self.pcd_resolution)
        self.scene_points = np.asarray(self.scene_pcd.points)
        mask = (self.scene_points[:, 2] < self.ceiling_height ) & (self.floor_height_down  < self.scene_points[:, 2])
        self.useful_points = self.scene_points[mask]


    def update_object(self,depth,position,rotation):
        self.robot_position = self.translation_func(position)
        self.robot_rotation = self.rotation_func(rotation)
        depth = depth.squeeze()
        self.robot_depth = preprocess.preprocess_depth(depth)         
        extrinsic=camera_extrinsic.habitat_camera_extrinsic(self.robot_position,self.robot_rotation)
        self.robot_pcd = pointcloud.caculate(self.robot_depth,self.camera_intrinsic,extrinsic)
        return np.asarray(self.robot_pcd.points)

    def update_sum(self):
        mask = (self.useful_points[:, 2] < self.floor_height_up) & (self.floor_height_down < self.useful_points[:, 2])
        current_navigable_points = self.useful_points[mask]

        horizontal_distance_squared = (self.useful_points[:, 0] - self.current_position[0]) ** 2 + (self.useful_points[:, 1] - self.current_position[1]) ** 2
        mask = (horizontal_distance_squared > 0.3) & (horizontal_distance_squared < 5)
        interpolate_points = self.useful_points[mask] 
        interpolate_points[:,2] = self.floor_height_up-0.1   
        interpolate_nav_points = np.linspace(np.ones_like(interpolate_points)*[self.current_position[0],self.current_position[1],self.floor_height_up],interpolate_points,20).reshape(-1,3)
        current_navigable_points = np.concatenate((interpolate_nav_points, current_navigable_points), axis=0)  
        self.current_navigable_pcd.points = o3d.utility.Vector3dVector(current_navigable_points)
        self.current_navigable_pcd = self.current_navigable_pcd.voxel_down_sample(self.pcd_resolution)

        self.navigable_pcd = self.navigable_pcd + self.current_navigable_pcd
        self.navigable_pcd = self.navigable_pcd.voxel_down_sample(self.pcd_resolution)
        self.navigable_points = np.asarray(self.navigable_pcd.points)
    
        mask = (self.useful_points[:, 2] > self.floor_height_up) & (self.useful_points[:, 2] < self.ceiling_height)
        current_obstacle_points = self.useful_points[mask]
        horizontal_distance_squared = (current_obstacle_points[:, 0] - self.current_position[0]) ** 2 + (current_obstacle_points[:, 1] - self.current_position[1]) ** 2
        mask = (horizontal_distance_squared > 0.5)
        current_obstacle_points = current_obstacle_points[mask]
        self.current_obstacle_pcd = pointcloud.points_pointcloud(current_obstacle_points)

        self.obstacle_pcd = self.obstacle_pcd + self.current_obstacle_pcd
        self.obstacle_pcd = self.obstacle_pcd.voxel_down_sample(self.pcd_resolution)
        self.obstacle_points = np.asarray(self.obstacle_pcd.points)



        current_frontier_points = project_frontier(self.current_navigable_pcd,self.current_obstacle_pcd,self.floor_height_up)
        if len(current_frontier_points) != 0:
            current_frontier_points = aggregate_points(current_frontier_points,self.threshold_1)
            print("self.threshold: ",self.threshold_1)
        if len(current_frontier_points) !=0:

            distance_squared = (current_frontier_points[:, 0] - self.current_position[0]) ** 2 + (current_frontier_points[:, 1] - self.current_position[1]) ** 2
            mask = distance_squared > 1
            current_frontier_points = current_frontier_points[mask]
            current_frontier_points_real = np.empty((0, 3))

            if len(self.sum_points)==0:
                current_frontier_points_real = np.vstack((current_frontier_points_real,current_frontier_points))
            else:
                grid_resolution = 0.25
                max_bound = np.max(self.sum_points,axis=0)
                min_bound = np.min(self.sum_points,axis=0)
                grid_dimensions = np.ceil((max_bound - min_bound) / grid_resolution).astype(int)
                grid_map = np.zeros((grid_dimensions[0],grid_dimensions[1]),dtype=np.int32)
                sum_indices = np.ceil((self.sum_points - min_bound) / grid_resolution).astype(int)
                sum_indices[:,0] = np.clip(sum_indices[:,0],0,grid_dimensions[0]-1)
                sum_indices[:,1] = np.clip(sum_indices[:,1],0,grid_dimensions[1]-1)
                grid_map[sum_indices[:,0],sum_indices[:,1]] = 1

                frontier_indices = np.floor((current_frontier_points - min_bound) / grid_resolution).astype(int)
    
                max_x = np.max(self.sum_points[:,0])
                min_x = np.min(self.sum_points[:,0])
                max_y = np.max(self.sum_points[:,1])
                min_y = np.min(self.sum_points[:,1])

                for i in range(len(current_frontier_points)):
                    if current_frontier_points[i][0] >= max_x or current_frontier_points[i][0] <= min_x or current_frontier_points[i][1] >= max_y or current_frontier_points[i][1] <= min_y:
                        current_frontier_points_real = np.vstack((current_frontier_points_real,current_frontier_points[i]))
                    else:
                        if grid_map[frontier_indices[i][0],frontier_indices[i][1]]==0:
                            current_frontier_points_real = np.vstack((current_frontier_points_real,current_frontier_points[i]))

            self.frontier_points = np.vstack((self.frontier_points, current_frontier_points_real))  

        if len(self.frontier_points) != 0:
            num = len(self.frontier_points)
            for i in range(num-1,-1,-1):
                distance = np.sqrt(np.sum(np.square(self.frontier_points[i][:-1] - self.current_position[:-1])))
                if distance < 1.1:
                    self.frontier_points = np.delete(self.frontier_points,i,axis=0)


        self.frontier_points = filter_points_reverse(self.frontier_points,self.threshold_2)

        if (len(self.frontier_points) != 0) and (len(self.frontier_points_ed) != 0):
            for i in range(len(self.frontier_points_ed)):
                for x in range(len(self.frontier_points)-1,-1,-1):
                    distance = np.sqrt(np.sum(np.square(self.frontier_points_ed[i][:-1] - self.frontier_points[x][:-1])))
                    if distance < 1.1:
                        self.frontier_points = np.delete(self.frontier_points,x,axis=0)

        if len(self.frontier_points) != 0:
            self.frontier_points_ed = np.vstack((self.frontier_points_ed,self.frontier_points[-1]))
            
        self.sum_points = np.vstack((self.sum_points, self.navigable_points,self.obstacle_points))
 
        self.navigable_costmap_frontier,self.max_bound_frontier,self.min_bound_frontier,self.grid_dimensions= project_costmap_frontier(self.navigable_pcd,self.obstacle_pcd)


