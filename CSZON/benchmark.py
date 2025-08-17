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
from mapper import *
from version_merge import *
from agent import *
from hm3d import *
from tqdm import tqdm


if __name__ == "__main__":
    output_path = "/home/mll"
    video_name = "video"
    habitat_config = habitat_test_config()
    habitat_env = habitat.Env(config=habitat_config)
    spl = 0
    success = 0
    name = 0
    file_path = r"/home/mll/success_rate.txt"


    habitat_mapper = Instruct_Mapper(camera_intrinsic.habitat_camera_intrinsic(habitat_config))
    habitat_agent = HM3D_Objnav_Agent(habitat_env,habitat_mapper,habitat_config)

    with open(file_path, 'w') as file:

        for i in tqdm(range(500)):
            video_name = "video"
            name = name + 1
            video_name = video_name + str(name)
            habitat_agent.reset()
            habitat_agent.make_plan()
            print("metrics",habitat_env.current_episode.object_category)

            while not habitat_env.episode_over:
                try:
                    key,vis_frames,arrived = habitat_agent.step()
                except:
                    key=1
                    arrived=0
                if key and arrived==0:
                    print("metrics",habitat_env.get_metrics())
                    spl = spl + habitat_env.get_metrics()["spl"]
                    success = success + habitat_env.get_metrics()["success"]

                    break

                if key and arrived:
                    print("metrics",habitat_env.get_metrics())
                    spl = spl + habitat_env.get_metrics()["spl"]
                    success = success + habitat_env.get_metrics()["success"]

                    break
            print("success:",success)
            print("spl:",spl)
            succ_r = str(success/(i+1))

            file.write(succ_r)
            file.write(",")


        file_path1 = r"/home/mll/result.txt"
        succ = str(success)
        spll = str(spl)
        with open(file_path1, 'w') as file1:
            file1.write(succ)
            file1.write(spll)
        print(f"{file_path1} has been created and written.")



