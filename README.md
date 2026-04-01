## 测试控制器
```bash
# 发布固定速度
./isaaclab.sh -p source/isaaclab_nav_task/scripts/play_fixed_velocity.py     --task Isaac-Nav-PPO-Go2-Play-v0     --num_envs 1     --vx 0.5     --vy 0.0     --omega 0.0     --steps 2000     --warmup_steps 100
```

## go2训练

测试版
```bash
 ./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py     --task Isaac-Nav-PPO-Go2-Dev-v0     --num_envs 256     --max_iterations 300
```

正式训练版本
``` bash
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py     --task Isaac-Nav-PPO-Go2-v0     --num_envs 512 --headless --logger=tensorboard
```

多卡训练
``` bash
ssh dxy@10.12.120.236 -p 10244

```

``` bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 
conda activate env_isaaclab211

../IsaacLab-2.1.1/isaaclab.sh -p -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    scripts/train.py \
    --task Isaac-Nav-PPO-Go2-v0 \
    --num_envs 512 \
    --logger=tensorboard \
    --headless \
    --distributed
```

## 小工具
查看训练情况 tensorboard
```bash
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/go2_navigation_ppo/<path> --bind_all --port 6006
```

ssh后本地查看：
1. 开一个端口转发的ssh
``` bash
ssh -L 6006:localhost:6006 dxy@10.12.120.236 -p 10244
```
2. 在ssh中开tensorboard
``` bash
cd /home/dxy/workspace/sru-navigation-learning
conda activate env_isaaclab211
tensorboard serve --logdir /home/dxy/workspace/sru-navigation-sim/logs/rsl_rl/go2_navigation_ppo/<path> --port 6006
```