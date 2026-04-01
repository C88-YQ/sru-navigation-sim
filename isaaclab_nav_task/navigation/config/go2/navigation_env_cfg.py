# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Go2 specific configuration for navigation environment."""

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

import isaaclab_nav_task.navigation.mdp as mdp
from isaaclab_nav_task.navigation.assets import GO2_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR  # isort: skip
from isaaclab_nav_task.navigation.navigation_env_cfg import NavigationEnvCfg

GO2_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


@configclass
class Go2NavigationEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config
        # from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator

        # initialize_depth_noise_generator(robot_name="go2", use_jit_precompiled=False)
        # camera_config = get_camera_config("go2")
        # _ = camera_config.resolution
        self.robot_name = "go2"

        self.scene.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.raycast_camera.offset.pos = (0.32715, 0.0, 0.10)
        self.scene.raycast_camera.offset.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.raycast_camera.pattern_cfg = patterns.PinholeCameraPatternCfg.from_ros_camera_info(
            # RealSense D435 datasheet (Rev 023, Mar 2026): depth FOV HD H=87 deg, V=58 deg.
            # Preserve SRU's 64x40 encoder input by using a 640x400 source frame with 10x downsampling.
            fx=337.2096,
            fy=360.8096,
            cx=320.0,
            cy=200.0,
            width=640,
            height=400,
            downsample_factor=10,
        )
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh"]),
            "threshold": 1.0,
        }

        self.actions.velocity_command.low_level_position_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=GO2_JOINT_NAMES,
            scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
            clip={".*": (-100.0, 100.0)},
            use_default_offset=True,
            preserve_order=True,
        )
        self.actions.velocity_command.low_level_velocity_action = None
        self.actions.velocity_command.low_level_policy_file = os.path.join(
            ISAACLAB_NAV_TASKS_ASSETS_DIR, "Policies", "locomotion", "go2", "policy.pt"
        )
        self.actions.velocity_command.reorder_joint_list = GO2_JOINT_NAMES

        self.rewards.joint_acc_l2_joint.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES)}

        self.observations.low_level_policy.base_lin_vel = None
        self.observations.low_level_policy.base_ang_vel.scale = 0.25
        self.observations.low_level_policy.projected_gravity.scale = 1.0
        self.observations.low_level_policy.velocity_commands.scale = 1.0
        self.observations.low_level_policy.joint_pos.scale = 1.0
        self.observations.low_level_policy.joint_vel.scale = 0.05
        self.observations.low_level_policy.actions.scale = 1.0
        self.observations.low_level_policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=GO2_JOINT_NAMES, preserve_order=True
        )
        self.observations.low_level_policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=GO2_JOINT_NAMES, preserve_order=True
        )
        self.observations.low_level_policy.enable_corruption = False

        self.events.randomize_action_scale = None
        self.events.randomize_low_pass_filter_alpha = None
        self.events.reset_robot_joints.params = {"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)}

        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class Go2NavigationEnvCfg_DEV(Go2NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class Go2NavigationEnvCfg_PLAY(Go2NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
