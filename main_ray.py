import time
import wandb
import torch
import pybullet as p
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_rl import MLPActorCritic
from utils_drone import HjAviary

DEVICE = torch.device("cpu")
CONTROL_MODE = "RL"  # RL
PERCENT = 1.0
MAX_EP_LEN = 1000
LOAD_PATH = "./data/interim/para_randomTMove_obs81_scenario_39.pt"  # _041212
RENDER = False
NUM_X = 128 * 6  # 6
NUM_Y = 128 * 6  # 6
NUM_Z = 128


def main():
    wandb.init(
        project="project-drone-test-20241122",
    )
    env = HjAviary(gui=RENDER)  # , ctrl_freq=10, pyb_freq=100

    obs_ma, info = env.reset(PERCENT)

    for i in tqdm(range(NUM_X)):
        x = (i - NUM_X/2) * 0.25
        for j in range(NUM_Y):
            y = (j - NUM_Y/2) * 0.25
            for k in range(NUM_Z):
                z = k * 0.25
                ray_from = [x, y, z]
                ray_to = [x, y, z+0.25]
                ray_results = p.rayTest(ray_from, ray_to)
                if ray_results[0][0]!=-1:
                    pass
                    # print("ray_results: ", ray_results)
                    if RENDER:
                        draw_ball(ray_results[0][3])

    time.sleep(99999)


def draw_ball(pos):
    # 定义球体的参数
    position = pos  # 球体的中心位置
    radius = 0.1  # 球体的半径
    color = [1, 0, 0, 1]  # 球体的颜色（红色，RGBA）
    # 创建球体的视觉形状
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color
    )
    # 创建球体的多体对象
    ball_id = p.createMultiBody(
        baseMass=0,  # 质量为0，表示这是一个静态物体
        baseCollisionShapeIndex=-1,  # 不需要碰撞形状
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position
    )


if __name__ == "__main__":
    main()
