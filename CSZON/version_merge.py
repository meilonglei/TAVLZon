import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.ndimage import binary_dilation



def project_frontier(navigable_pcd,obstacle_pcd,height,obstacle_height=1,grid_resolution=0.25):
    pcd_navigable = navigable_pcd
    pcd_obstacle  = obstacle_pcd
    points_obstacle =np.asarray(pcd_obstacle.points)
    points_navigable = np.asarray(pcd_navigable.points)
    np_all_points = np.concatenate((points_obstacle,points_navigable),axis=0)
    if len(np_all_points) != 0:
        max_bound = np.max(np_all_points,axis=0)  
        min_bound = np.min(np_all_points,axis=0)
        grid_dimensions = np.ceil((max_bound - min_bound) / grid_resolution).astype(int)
        grid_map = np.ones((grid_dimensions[0],grid_dimensions[1]),dtype=np.int32)

        navigable_indices = np.floor((points_navigable - min_bound) / grid_resolution).astype(int)
        navigable_indices[:,0] = np.clip(navigable_indices[:,0],0,grid_dimensions[0]-1)
        navigable_indices[:,1] = np.clip(navigable_indices[:,1],0,grid_dimensions[1]-1)
        navigable_indices[:,2] = np.clip(navigable_indices[:,2],0,grid_dimensions[2]-1)
        navigable_voxels = np.zeros(grid_dimensions,dtype=np.int32)
        navigable_voxels[navigable_indices[:,0],navigable_indices[:,1],navigable_indices[:,2]] = 1
        navigable_map = (navigable_voxels.sum(axis=2) > 0)
        grid_map[np.where(navigable_map>0)] = 3

        obstacle_indices = np.floor((points_obstacle - min_bound) / grid_resolution).astype(int)
        obstacle_indices[:,0] = np.clip(obstacle_indices[:,0],0,grid_dimensions[0]-1)
        obstacle_indices[:,1] = np.clip(obstacle_indices[:,1],0,grid_dimensions[1]-1)
        obstacle_indices[:,2] = np.clip(obstacle_indices[:,2],0,grid_dimensions[2]-1)
        obstacle_voxels = np.zeros(grid_dimensions,dtype=np.int32)
        obstacle_voxels[obstacle_indices[:,0],obstacle_indices[:,1],obstacle_indices[:,2]] = 1
        obstacle_map = (obstacle_voxels.sum(axis=2) > 0)
        grid_map[np.where(obstacle_map>0)] = 0

        outer_border_navigable = ((grid_map == 3)*255).astype(np.uint8)  
        contours,hierarchiy = cv2.findContours(outer_border_navigable,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        outer_border_navigable = cv2.drawContours(np.zeros((grid_map.shape[0],grid_map.shape[1])),contours,-1,(255,255,255),1).astype(np.float32)
        obstacles = ((grid_map == 0)*255).astype(np.float32)
        obstacles = cv2.dilate(obstacles.astype(np.uint8),np.ones((3,3)))
        outer_border_navigable = ((outer_border_navigable - obstacles) > 0)
        grid_map_x,grid_map_y = np.where(outer_border_navigable>0)
        grid_indexes = np.stack((grid_map_x,grid_map_y,obstacle_height*np.ones((grid_map_x.shape[0],))),axis=1)
        frontier_points = grid_indexes * grid_resolution + min_bound
        frontier_points[:,2] = height  
    else:
        frontier_points = np.empty((0,3))
    return frontier_points


def aggregate_points(points, threshold_1):  

    dist_matrix = squareform(pdist(points, 'euclidean'))
    
    n = len(points)
    visited = np.zeros(n, dtype=bool)  
    aggregated_points = []

    for i in range(n):
        if visited[i]:
            continue
        
        close_points_indices = np.where(dist_matrix[i] < threshold_1)[0]
        
        if len(close_points_indices) == 1:
            aggregated_points.append(points[i])
            visited[i] = True
        else:

            avg_point = np.mean(points[close_points_indices], axis=0)
            aggregated_points.append(avg_point)
            
            visited[close_points_indices] = True
    
    return np.array(aggregated_points)


def filter_points_reverse(points, min_distance):
    if len(points) == 0:
        return points
    
    filtered_indices = [len(points) - 1]  
    
    for i in range(len(points) - 2, -1, -1):
        current_point = points[i]
        

        tree = cKDTree(points[filtered_indices])
        distances, _ = tree.query(current_point, k=1)
        

        if distances >= min_distance:
            filtered_indices.append(i)

    filtered_indices.sort()
    return points[filtered_indices]


def project_costmap(navigable_pcd,obstacle_pcd,start_position,goal_position,grid_resolution=0.25):
    sum_pcd = navigable_pcd + obstacle_pcd
    points_navigable = navigable_pcd.points
    points_obstacle = obstacle_pcd.points
    sum_points = sum_pcd.points
    max_bound = np.max(sum_points,axis=0)
    min_bound = np.min(sum_points,axis=0)
    grid_dimensions = np.ceil((max_bound - min_bound) / grid_resolution).astype(int)
    grid_map = np.ones(grid_dimensions,dtype=np.float32) * 20

    start_index = np.floor((start_position - min_bound) / grid_resolution).astype(int)
    goal_index = np.floor((goal_position - min_bound) / grid_resolution).astype(int)

    goal_index[0] = np.clip(goal_index[0],0,grid_dimensions[0]-1)
    goal_index[1] = np.clip(goal_index[1],0,grid_dimensions[1]-1)
    goal_index[2] = np.clip(goal_index[2],0,grid_dimensions[2]-1)

    navigable_indices = np.floor((points_navigable - min_bound) / grid_resolution).astype(int)
    navigable_indices[:,0] = np.clip(navigable_indices[:,0],0,grid_dimensions[0]-1)
    navigable_indices[:,1] = np.clip(navigable_indices[:,1],0,grid_dimensions[1]-1)
    navigable_indices[:,2] = np.clip(navigable_indices[:,2],0,grid_dimensions[2]-1)
    navigable_voxels = np.zeros(grid_dimensions,dtype=np.int32)
    navigable_voxels[navigable_indices[:,0],navigable_indices[:,1],navigable_indices[:,2]] = 1
    navigable_map = (navigable_voxels.sum(axis=2) > 0)
    grid_map[np.where(navigable_map>0)] = 1
    obstacle_indices = np.floor((points_obstacle - min_bound) / grid_resolution).astype(int)
    obstacle_indices[:,0] = np.clip(obstacle_indices[:,0],0,grid_dimensions[0]-1)
    obstacle_indices[:,1] = np.clip(obstacle_indices[:,1],0,grid_dimensions[1]-1)
    obstacle_indices[:,2] = np.clip(obstacle_indices[:,2],0,grid_dimensions[2]-1)
    obstacle_voxels = np.zeros(grid_dimensions,dtype=np.int32)
    obstacle_voxels[obstacle_indices[:,0],obstacle_indices[:,1],obstacle_indices[:,2]] = 1
    obstacle_map = (obstacle_voxels.sum(axis=2) > 0)
    grid_map[np.where(obstacle_map>0)] = 1000
    navigable_costmap = grid_map.max(axis=2)

    color_navigable_costmap = cv2.applyColorMap((navigable_costmap).astype(np.uint8),cv2.COLORMAP_JET)
    return navigable_costmap,color_navigable_costmap,start_index[0:2],goal_index[0:2],min_bound



def project_costmap_frontier(navigable_pcd,obstacle_pcd,grid_resolution=0.10):
    sum_pcd = navigable_pcd + obstacle_pcd
    points_navigable = navigable_pcd.points
    points_obstacle = obstacle_pcd.points
    sum_points = sum_pcd.points
    max_bound = np.max(sum_points,axis=0)
    min_bound = np.min(sum_points,axis=0)
    grid_dimensions = np.ceil((max_bound - min_bound) / grid_resolution).astype(int)
    grid_map = np.ones(grid_dimensions,dtype=np.float32) * 20
    print(f"Shape of grid_map: {grid_map.shape}")

    navigable_indices = np.floor((points_navigable - min_bound) / grid_resolution).astype(int)
    navigable_indices[:,0] = np.clip(navigable_indices[:,0],0,grid_dimensions[0]-1)
    navigable_indices[:,1] = np.clip(navigable_indices[:,1],0,grid_dimensions[1]-1)
    navigable_indices[:,2] = np.clip(navigable_indices[:,2],0,grid_dimensions[2]-1)
    navigable_voxels = np.zeros(grid_dimensions,dtype=np.int32)
    navigable_voxels[navigable_indices[:,0],navigable_indices[:,1],navigable_indices[:,2]] = 1
    navigable_map = (navigable_voxels.sum(axis=2) > 0)
    grid_map[np.where(navigable_map>0)] = 1
    obstacle_indices = np.floor((points_obstacle - min_bound) / grid_resolution).astype(int)
    obstacle_indices[:,0] = np.clip(obstacle_indices[:,0],0,grid_dimensions[0]-1)
    obstacle_indices[:,1] = np.clip(obstacle_indices[:,1],0,grid_dimensions[1]-1)
    obstacle_indices[:,2] = np.clip(obstacle_indices[:,2],0,grid_dimensions[2]-1)
    obstacle_voxels = np.zeros(grid_dimensions,dtype=np.int32)
    obstacle_voxels[obstacle_indices[:,0],obstacle_indices[:,1],obstacle_indices[:,2]] = 1
    obstacle_map = (obstacle_voxels.sum(axis=2) > 0)
    grid_map[np.where(obstacle_map>0)] = 1000

    navigable_costmap = grid_map.max(axis=2)
    obstacle_mask = (navigable_costmap == 1000)
    dilated_mask = binary_dilation(obstacle_mask, iterations=1)
    navigable_costmap[dilated_mask] = 20

    return navigable_costmap,max_bound,min_bound,grid_dimensions


def path_planning(costmap,start_index,goal_index):
    planmap = costmap.copy()
    grid = Grid(matrix=(planmap).astype(np.int32))
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    start_index[1] = np.clip(start_index[1],0,costmap.shape[1]-1)
    start_index[0] = np.clip(start_index[0],0,costmap.shape[0]-1)
    goal_index[1] = np.clip(goal_index[1],0,costmap.shape[1]-1)
    goal_index[0] = np.clip(goal_index[0],0,costmap.shape[0]-1)
    start = grid.node(start_index[1],start_index[0])
    goal = grid.node(goal_index[1],goal_index[0])
    path,_ = finder.find_path(start,goal,grid)
    return path





