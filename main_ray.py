import time
import wandb
import torch
import pybullet as p
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_rl import MLPActorCritic
from utils_drone import HjAviary

PERCENT = 1.0
RENDER = False
NUM_X = 128 * 6  # 6
NUM_Y = 128 * 6  # 6
NUM_Z = 128
SAVE_path = "./data/occ_array.npy"



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


def main():
    wandb.init(
        project="project-drone-test-20241122",
    )
    env = HjAviary(gui=RENDER)  # , ctrl_freq=10, pyb_freq=100

    obs_ma, info = env.reset(PERCENT)

    occ_array = np.zeros((NUM_X, NUM_Y, NUM_Z))

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
                    occ_array[i][j][k] = 1
                    # print("ray_results: ", ray_results)
                    if RENDER:
                        draw_ball(ray_results[0][3])
    print("np.sum(occ_array): ", np.sum(occ_array))

    np.save(SAVE_path, occ_array)


if __name__ == "__main__":
    main()
