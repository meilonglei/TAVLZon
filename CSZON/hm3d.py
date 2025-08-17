import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from omegaconf import DictConfig, OmegaConf,open_dict
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import LookUpActionConfig,LookDownActionConfig
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat_sim.utils import viz_utils as vut
import cv2
import os
import inspect


def habitat_test_config():
    config=habitat.get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    with read_write(config):
        config.habitat.dataset.data_path="/home/mll/data/total/dataset/habitat_task/objectnav/hm3d/v2/{split}/{split}.json.gz"
        config.habitat.dataset.scenes_dir="/home/mll/data/total/dataset/scenes"
        config.habitat.simulator.scene_dataset="/home/mll/data/total/dataset/scenes/hm3d_v0.2/hm3d_annotated_val_basis.scene_dataset_config.json"
        config.habitat.task.measurements.success.success_distance = 0.50
        config.habitat.environment.iterator_options.num_episode_sample = 500
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        config.habitat.simulator.turn_angle = 30
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        config.habitat.dataset.split = "val"
        config.habitat.task.actions['look_up'] = LookUpActionConfig()
        config.habitat.task.actions['look_down'] = LookDownActionConfig()

        # config.habitat.task.measurements.update(
        #     {
        #         "top_down_map": TopDownMapMeasurementConfig(
        #             map_padding=3,
        #             map_resolution=1024,
        #             draw_source=True,
        #             draw_border=True,
        #             draw_shortest_path=False,
        #             draw_view_points=True,
        #             draw_goal_positions=True,
        #             draw_goal_aabbs=True,
        #             fog_of_war=FogOfWarConfig(
        #             draw=True,
        #             visibility_dist=5.0,
        #             fov=90,
        #             ),
        #         ),
        #         "collisions": CollisionsMeasurementConfig(),
        #     }
        # )
    return config





