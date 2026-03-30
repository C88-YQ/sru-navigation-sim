# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Go2-specific configuration for navigation environment.

This version reuses IsaacLab's built-in UNITREE_GO2_CFG as the robot asset.

Notes:
- The current SRU navigation stack was originally written for wheeled/legged low-level policies.
- Go2 is a pure legged robot, so the low-level velocity branch is intentionally left empty.
- You still need to provide a Go2-compatible low-level locomotion policy at the configured path.
"""

import os

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_nav_task.navigation.navigation_env_cfg import NavigationEnvCfg, ObservationsCfg as BaseObservationsCfg
import isaaclab_nav_task.navigation.mdp as mdp

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab_nav_task.navigation.assets import ISAACLAB_NAV_TASKS_ASSETS_DIR  # isort: skip


LEG_JOINT_NAMES = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]


@configclass
class Go2ObservationsCfg(BaseObservationsCfg):
    @configclass
    class LowLevelPolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        velocity_commands = ObsTerm(func=mdp.generated_actions, params={"action_name": "velocity_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_low_level_action, params={"action_term": "velocity_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    low_level_policy: LowLevelPolicyCfg = LowLevelPolicyCfg()


@configclass
class Go2NavigationEnvCfg(NavigationEnvCfg):
    observations: Go2ObservationsCfg = Go2ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator
        from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config

        initialize_depth_noise_generator(robot_name="go2", use_jit_precompiled=False)
        camera_config = get_camera_config("go2")
        _camera_resolution = camera_config.resolution

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # IsaacLab's built-in Go2 asset uses the base link name `base`.
        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Keep the camera pose conservative for now. This should be updated once the
        # real sensor mounting position on your Go2 is known.
        self.scene.raycast_camera.offset.pos = (0.30, 0.0, 0.12)

        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
            "threshold": 1.0,
        }

        # Go2 is a pure legged robot. The low-level position branch drives all 12 joints.
        # The velocity branch is left empty to preserve compatibility with the current
        # hierarchical action interface until a dedicated Go2 low-level controller is added.
        self.actions.velocity_command.low_level_position_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            scale=0.25,
            use_default_offset=True,
        )
        self.actions.velocity_command.low_level_velocity_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=[],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.velocity_command.low_level_policy_file = os.path.join(
            ISAACLAB_NAV_TASKS_ASSETS_DIR, "Policies", "locomotion", "go2", "policy_go2_jit.pt"
        )

        self.rewards.joint_acc_l2_joint.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
        }

        self.events.randomize_low_pass_filter_alpha.params = {
            "alpha_range": (0.1, 0.6),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.1, 0.6),
            "alpha_range_vy": (0.1, 0.6),
            "alpha_range_omega": (0.1, 0.6),
        }

        # Go2 is smaller than B2W. Start from the same terrain band, but keep these easy
        # to tune after the first stability pass.
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
