#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play navigation with fixed velocity commands.")
parser.add_argument("--task", type=str, required=True, help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--vx", type=float, default=0.5, help="Fixed forward velocity command.")
parser.add_argument("--vy", type=float, default=0.0, help="Fixed lateral velocity command.")
parser.add_argument("--omega", type=float, default=0.0, help="Fixed yaw rate command.")
parser.add_argument("--steps", type=int, default=2000, help="Number of simulation steps.")
parser.add_argument("--warmup_steps", type=int, default=0, help="Initial steps with zero action.")
parser.add_argument("--print_every", type=int, default=50, help="Debug print interval.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg


def main() -> None:
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    action = torch.tensor([args_cli.vx, args_cli.vy, args_cli.omega], device=env.unwrapped.device, dtype=torch.float32)
    action = action.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    zero_action = torch.zeros_like(action)

    print(f"[INFO] Running task={args_cli.task}")
    print(f"[INFO] num_envs={env.unwrapped.num_envs}, fixed_action={(args_cli.vx, args_cli.vy, args_cli.omega)}")
    print(f"[INFO] warmup_steps={args_cli.warmup_steps}, steps={args_cli.steps}")

    step_count = 0
    while simulation_app.is_running() and step_count < args_cli.steps:
        current_action = zero_action if step_count < args_cli.warmup_steps else action
        env.step(current_action)

        if step_count % args_cli.print_every == 0:
            action_term = env.unwrapped.action_manager.get_term("velocity_command")
            robot = env.unwrapped.scene["robot"]
            processed = action_term.processed_actions[0].detach().cpu().tolist()
            filtered = action_term.filtered_velocity_commands[0].detach().cpu().tolist()
            low_level = action_term.low_level_actions[0].detach().cpu()
            base_lin_vel = robot.data.root_lin_vel_b[0].detach().cpu().tolist()
            joint_pos = robot.data.joint_pos[0, : min(4, robot.data.joint_pos.shape[1])].detach().cpu().tolist()
            print(
                f"[STEP {step_count}] cmd={current_action[0].detach().cpu().tolist()} "
                f"processed={processed} filtered={filtered} "
                f"ll_abs_mean={low_level.abs().mean().item():.4f} ll_first4={low_level[:4].tolist()} "
                f"base_lin_vel={base_lin_vel} joint_pos_first4={joint_pos}"
            )

        step_count += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
