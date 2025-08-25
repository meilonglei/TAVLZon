import habitat
from PIL import Image
from PIL import ImageDraw
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from omegaconf import DictConfig, OmegaConf,open_dict
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import LookUpActionConfig,LookDownActionConfig
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut
import cv2
import os
import inspect
from Transform_unit.camera_intrinsic import habitat_camera_intrinsic
current_dir = os.getcwd()
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from QW_vl import *
from mapper import *
from version_merge import *
from demo import *

class HM3D_Objnav_Agent:
    def __init__(self,env:habitat.Env,mapper:Instruct_Mapper,config):
        self.env = env
        self.config = config
        self.mapper = mapper
        self.episode_samples = 0
        self.planner = ShortestPathFollower(env.sim,0.5,False)

    def transform_rgb_bgr(self,image):
        return image[:, :, [2, 1, 0]]
    
    def reset(self):
        self.episode_steps = 0
        self.obs = self.env.reset()
        self.arrived = 0
        self.key = 1
        self.double_obj = 0
        self.location_mean = np.empty((0,3))
        self.vis_frames = []
        self.image_around = []
        self.depth_around = []
        self.position_around = []
        self.rotation_around = []
        info = self.env.get_metrics()
        self.label = self.env.current_episode.object_category  
        frame = observations_to_image(self.obs, info)
        frame = overlay_frame(frame, info)
        self.vis_frames = [frame]
        self.strong_key = 0
        self.strong_time = 0
        self.concat_image = None


        self.mapper.reset(self.env.sim.get_agent_state().sensor_states['rgb'].position,self.env.sim.get_agent_state().sensor_states['rgb'].rotation) 
        self.mapper.floor_height_up = self.env.sim.get_agent_state().sensor_states['rgb'].position[1] - 0.875 + 0.2
        self.mapper.floor_height_down = self.env.sim.get_agent_state().sensor_states['rgb'].position[1] - 0.875 - 0.2
        self.mapper.ceiling_height = self.env.sim.get_agent_state().sensor_states['rgb'].position[1] - 0.875 + 1.2
        self.mapper.height = self.env.sim.get_agent_state().sensor_states['rgb'].position[1] - 0.875 + 0.4

    def rotate_panoramic(self,rotate_times = 12):
        self.mapper.threshold_1 = 1.1
        self.mapper.threshold_2 = 1.1
        self.mapper.current_frontier_array = []
        self.image_around = []
        self.depth_around = []
        self.position_around = []
        self.rotation_around = []
        for i in range(rotate_times):
            if self.env.episode_over:
                break
            if i%2==0:
                self.update_trajectory()
            self.image_around.append(self.obs['rgb'])    
            self.depth_around.append(self.obs['depth']) 
            self.position_around.append(self.env.sim.get_agent_state().sensor_states['rgb'].position)
            self.rotation_around.append(self.env.sim.get_agent_state().sensor_states['rgb'].rotation)
            self.obs = self.env.step(3)

            info = self.env.get_metrics()
            frame = observations_to_image(self.obs, info)
            frame = overlay_frame(frame, info)
            self.vis_frames.append(frame)

        self.concat_image = self.concat_panoramic(self.image_around)
        self.concat_image.save(r"/home/mll/pictures/obj_源.png")
        answer_ll = QW_LL(Text_1,self.simplified_label(self.label))

        print("association",answer_ll["association"])

        if answer_ll["association"] == "Strong association":
            print("Room type: ",answer_ll["room"])
            self.mapper.threshold_1 = 1.5
            self.mapper.threshold_2 = 1.5

            question_vl = "Is this a " + answer_ll["room"] + "?"
            answer_vl = QW_VL(r"/home/mll/pictures/obj_源.png",text_vl,question_vl)
            if answer_vl == "yes":
                self.mapper.threshold_1 = 0.8
                self.mapper.threshold_2 = 0.8


        self.mapper.update_sum()
        self.mapper.scene_pcd = o3d.geometry.PointCloud()


    def concat_panoramic(self,images):  
        height,width = 480,640
        concat_image = np.zeros((2*height + 3*20, 3*width + 4*20, 3),np.uint8)
        copy_images = np.array(images,dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
                row = ((i%4)//2)
                col = (i//4)
                concat_image[20*(row+1)+row*height:20*(row+1)+row*height+height:,col*width + col * 20+20:col*width+col*20+width+20,:] = copy_images[i]
        concat_image = Image.fromarray(concat_image)
        return concat_image


    def simplified_label(self,label):
        def_art = "the "
        dot = "."
        if label == "tv_monitor":
            label = "television"
        label_final = def_art + label + dot
        return label_final


    def detection_result(self,concat_image,label): 
        results = obj_detection(concat_image,label)
        if len(results[0]["boxes"]) != 0:
            for i, box in enumerate(results[0]['boxes']):
                if i==0:
                    xmin, ymin, xmax, ymax = map(int, box.tolist())  
                    label = results[0]['text_labels'][i] 
                    score = results[0]['scores'][i].item()  
                if results[0]['scores'][i].item() > results[0]['scores'][0].item():
                    xmin, ymin, xmax, ymax = map(int, box.tolist())  
                    label = results[0]['text_labels'][i]  
                    score = results[0]['scores'][i].item()  
            result_best = {"score":score,"label":label,"box":{"xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax}}
        else:
            result_best = []
        return result_best


    def update_trajectory(self):
        self.episode_steps += 2
        self.position = self.env.sim.get_agent_state().sensor_states['rgb'].position
        self.rotation = self.env.sim.get_agent_state().sensor_states['rgb'].rotation
        self.mapper.update_single(self.obs['rgb'],self.obs['depth'],self.position,self.rotation)


    def update_step(self):
        self.episode_steps += 1
        self.position = self.env.sim.get_agent_state().sensor_states['rgb'].position
        self.mapper.current_position = self.mapper.translation_func(self.position)
        self.rotation = self.env.sim.get_agent_state().sensor_states['rgb'].rotation


    def obj_location(self):
        self.location_mean = np.empty((0,3))
        label = self.simplified_label(self.label)


        Text = f"Is there a {self.label} in this picture?"
        answer = QW_VL(r"/home/mll/pictures/obj_源.png",text3,Text)

        if answer == "yes":
            result = self.detection_result(self.concat_image,label)
            if (len(result) != 0):

                box = result["box"]
                if box["xmin"] < 660 and box["xmax"] > 680:
                    if (660-box["xmin"]) > (box["xmax"]-680):
                        box["xmax"] = 660
                    if (660-box["xmin"]) < (box["xmax"]-680):
                        box["xmin"] = 680

                if box["xmin"] < 1320 and box["xmax"] > 1340:
                    if (1320-box["xmin"]) > (box["xmax"]-1340):
                        box["xmax"] = 1320
                    if (660-box["xmin"]) < (box["xmax"]-1340):
                        box["xmin"] = 1340               

                if (box["xmin"] < 650+10) and (box["ymin"] < 490+10):
                    number = 0
                if (650+10 <= box["xmin"] < 1300+20) and (box["ymin"] < 490+10):
                    number = 4
                if (1300+20 <= box["xmin"] < 1950+30) and (box["ymin"] < 490+10):
                    number = 8
                if (box["xmin"] < 650+10) and (490+10 <= box["ymin"] < 980+20):
                    number = 2
                if (650+10 <= box["xmin"] < 1300+20) and (490+10 <= box["ymin"] < 980+20):
                    number = 6
                if (1300+20 <= box["xmin"] < 1950+30) and (490+10 <= box["ymin"] < 980+20):
                    number = 10

                draw = ImageDraw.Draw(self.concat_image)  
                box = result["box"]
                label = result["label"]
                score = result["score"]
                xmin, ymin, xmax, ymax = box.values()
                draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
                draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="yellow",font_size = 20)
                self.concat_image.save(r"/home/mll/pictures/obj.png")

                row = ((number%4)//2)
                col = (number//4)
                box["xmin"] = box["xmin"] - 640*col - 20*(col+1)
                box["xmax"] = box["xmax"] - 640*col - 20*(col+1)
                box["ymin"] = box["ymin"] - 480*row - 20*(row+1)
                box["ymax"] = box["ymax"] - 480*row - 20*(row+1)
                obj_image = self.image_around[number]
                obj_depth = self.depth_around[number]

                image = Image.fromarray(obj_image) 
                image.save(r"/home/mll/pictures/obj_single.png")
                answer = QW_VL(r"/home/mll/pictures/obj_single.png",text3,Text)
                print("answer:",answer)
                if answer == "yes":
                    y = box["ymax"] - box["ymin"]
                    x = box["xmax"] - box["xmin"]
                    box["ymin"] = box["ymin"] + y*3//8
                    box["ymax"] = box["ymax"] - y*3//8
                    box["xmin"] = box["xmin"] + x*3//8
                    box["xmax"] = box["xmax"] - x*3//8
                    masked_depth = np.zeros_like(obj_depth)
                    masked_depth[box["ymin"]:box["ymax"],box["xmin"]:box["xmax"]] = obj_depth[box["ymin"]:box["ymax"],box["xmin"]:box["xmax"]]
                    posi = self.position_around[number]
                    posi_trans = posi[[0,2,1]]
                    rota = self.rotation_around[number]
                    location_sum = self.mapper.update_object(masked_depth,posi,rota)

                    if len(location_sum) != 0:
                            self.location_mean_mean = np.mean(location_sum,axis=0)
                            dis_mean = np.sqrt(np.sum(np.square(self.location_mean_mean[:-1] - posi_trans[:-1])))
                            for i in range(len(location_sum)):
                                dis = np.sqrt(np.sum(np.square(location_sum[i][:-1] - posi_trans[:-1])))
                                if i == 0:
                                    self.location_mean = location_sum[0]
                                    min_dis = dis
                                    max_dis = dis
                                if dis <= min_dis:
                                    min_dis = dis
                                    self.location_mean_min = location_sum[i]
                                if dis >= max_dis:
                                    max_dis = dis
                                    self.location_mean_max = location_sum[i]

                            if (max_dis-dis_mean) / (dis_mean-min_dis) > 1:
                                self.location_mean = self.location_mean_min
                                print("minimum distance:",min_dis)
                            else:
                                self.location_mean = self.location_mean_mean
                                print("middle distance:",dis_mean)
                
                            self.location_mean[2] = self.env.sim.get_agent_state().sensor_states['rgb'].position[1] - 0.8
                            print("location_mean:",self.location_mean)

                    if len(location_sum) == 0:
                        masked_depth = np.zeros_like(obj_depth)
                        masked_depth[box["ymin"]:box["ymax"],box["xmin"]:box["xmax"]] = 4.0
                        location_sum = self.mapper.update_object(masked_depth,posi,rota)
                        self.location_mean = np.mean(location_sum,axis=0)
                        self.location_mean[2] = self.env.sim.get_agent_state().sensor_states['rgb'].position[1] - 0.8
                        self.double_obj = 1

        return self.location_mean

    def obj_location_double(self):
        if self.double_obj == 1 and len(self.location_mean) != 0:
            while not self.env.episode_over:
                act = self.planner.get_next_action(self.location_mean[[0,2,1]])
                if act != 0:
                    self.obs = self.env.step(act)
                if act == 0:
                    break
                self.update_step()
                distance = np.sqrt(np.sum(np.square(self.location_mean[[0,2,1]][:-1] - self.mapper.current_position[:-1])))
                if distance < 0.5:
                    break
            self.rotate_panoramic()  
            self.location_mean = self.obj_location()
        return self.location_mean


    

    def make_plan(self):
        if self.key == 1:
            self.rotate_panoramic()
            self.location_mean = self.obj_location()
            self.location_mean = self.obj_location_double()
        if len(self.location_mean) != 0 and self.key == 1:
            self.target_point = self.location_mean
            self.arrived = 1
            # print("location: ",self.target_point[[0,2,1]])
        else:
            if len(self.mapper.frontier_points) != 0:
                self.target_point = self.mapper.frontier_points[-1]
                self.mapper.frontier_points = self.mapper.frontier_points[:-1]
                
                start_index_frontier = np.floor((self.mapper.current_position - self.mapper.min_bound_frontier) / 0.10).astype(int)
                goal_index_frontier = np.floor((self.target_point - self.mapper.min_bound_frontier) / 0.10).astype(int)
                goal_index_frontier[0] = np.clip(goal_index_frontier[0],0,self.mapper.grid_dimensions[0]-1)
                goal_index_frontier[1] = np.clip(goal_index_frontier[1],0,self.mapper.grid_dimensions[1]-1)
                goal_index_frontier[2] = np.clip(goal_index_frontier[2],0,self.mapper.grid_dimensions[2]-1)
                path_frontier = path_planning(self.mapper.navigable_costmap_frontier,start_index_frontier,goal_index_frontier)

                self.mapper.navigable_costmap_frontier[start_index_frontier[0],start_index_frontier[1]] = 1
                self.mapper.navigable_costmap_frontier[goal_index_frontier[0],goal_index_frontier[1]] = 1
                coordinate_list_frontier = [[node.y, node.x] for node in path_frontier]


                self.downsampled = coordinate_list_frontier[4::5]

                if not self.downsampled:
                    return [coordinate_list_frontier[-1]]

                if self.downsampled[-1] != coordinate_list_frontier[-1]:
                    self.downsampled.append(coordinate_list_frontier[-1])

                for x,y in coordinate_list_frontier:
                    if self.mapper.navigable_costmap_frontier[x,y] == 1000 :
                        print("error")

            else:
                self.success = 1

        self.target_point_trs = self.target_point[[0,2,1]]


    def step(self):
        k=1
        self.key = 1
        self.success=0
        self.TIMES = 0

        if self.arrived == 0:

            for i in range(len(self.downsampled)):
                frontier_point = np.array([self.downsampled[i][0], self.downsampled[i][1]]) * 0.10 + self.mapper.min_bound_frontier[0:2]
                frontier_point = np.append(frontier_point,self.mapper.floor_height_up)
                frontier_point = frontier_point[[0,2,1]]
                position = self.env.sim.get_agent_state().sensor_states['rgb'].position
                position = transform.habitat_translation(position)
                act = self.planner.get_next_action(frontier_point)

                while(act != 0 and not self.env.episode_over):
                    act = self.planner.get_next_action(frontier_point)
                    print("action",act)
                    distance =  np.sqrt(np.sum(np.square(self.mapper.current_position[:-1] - frontier_point[[0,2,1]][:-1])))
                    print("distance：",distance)

                    if act != 0:
                        self.obs = self.env.step(act)
                        info = self.env.get_metrics()
                        frame = observations_to_image(self.obs, info)
                        frame = overlay_frame(frame, info)
                        self.vis_frames.append(frame)
                        self.update_step()

                    if distance < 0.5:
                        break

            self.make_plan()
            if self.env.episode_over:
                self.success = 1

        if self.arrived == 1:
            self.success = 1
            distance_2 =  np.sqrt(np.sum(np.square(self.target_point[:-1] - self.mapper.current_position[:-1])))
            while(distance_2 > 1.2  and not self.env.episode_over):
                self.TIMES = self.TIMES + 1
                self.times = 0  
                nav_map,color_map,start_index,goal_index,min_bound = project_costmap(self.mapper.current_navigable_pcd,self.mapper.current_obstacle_pcd,self.mapper.current_position,self.target_point)
                path = path_planning(nav_map,start_index,goal_index)
                
                nav_map[start_index[0],start_index[1]] = 1
                nav_map[goal_index[0],goal_index[1]] = 20
                nav = []

                coordinate_list = [[node.y, node.x] for node in path]
                for x,y in coordinate_list:
                    if nav_map[x,y] == 20 or nav_map[x,y] == 1000 :
                        nav.append([x,y])
                        target = nav[-1]
                        target = np.array([target[0], target[1]]) * 0.25 + min_bound[0:2]
                        target = np.append(target,self.mapper.floor_height_up)
                        waypoint = target[[0,2,1]]

                        break
            

                action = ShortestPathFollower(self.env.sim,0.5,False).get_next_action(waypoint)
                position = self.env.sim.get_agent_state().sensor_states['rgb'].position
                position = transform.habitat_translation(position)
                distance_1 = np.sqrt(np.sum(np.square(position[:-1] - waypoint[[0,2,1]][:-1])))

                if self.TIMES > 20:
                    break   

                while(distance_1 > 0.5 and action!=0):
                    self.times = self.times + 1
                    action = ShortestPathFollower(self.env.sim,0.5,False).get_next_action(waypoint)

                    if action != 0 and not self.env.episode_over:
                        self.obs = self.env.step(action)

                        info = self.env.get_metrics()
                        frame = observations_to_image(self.obs, info)
                        frame = overlay_frame(frame, info)
                        self.vis_frames.append(frame)

                    position = self.env.sim.get_agent_state().sensor_states['rgb'].position
                    position = transform.habitat_translation(position)
                    distance_1 = np.sqrt(np.sum(np.square(position[:-1] - waypoint[[0,2,1]][:-1])))

                    if action == 0:
                        break
                    

                    if self.times > 50:
                        break

                distance_2 = np.sqrt(np.sum(np.square(position[:-1] - self.target_point[:-1])))
                if distance_2 > 1.2:
                    self.rotate_panoramic()



            if not self.env.episode_over:
                self.env.step(0)

        return self.success,self.vis_frames,self.arrived
