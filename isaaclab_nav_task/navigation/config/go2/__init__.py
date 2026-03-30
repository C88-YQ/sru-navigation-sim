# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import gymnasium as gym

from . import agents, navigation_env_cfg


gym.register(
    id="Isaac-Nav-MDPO-GO2-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavMDPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Nav-MDPO-GO2-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavMDPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Nav-MDPO-GO2-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavMDPORunnerDevCfg,
    },
)


gym.register(
    id="Isaac-Nav-PPO-GO2-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Nav-PPO-GO2-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Nav-PPO-GO2-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavPPORunnerDevCfg,
    },
)
