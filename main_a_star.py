import numpy as np
import open3d as o3d
import pybullet as p
from tqdm import tqdm


def draw_ball(pos, color=None):
    # 定义球体的参数
    if color is None:
        color = [1, 0, 0, 1]
    position = pos  # 球体的中心位置
    radius = 0.1  # 球体的半径
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
    start_point = [0, 0, 0]
    target_point = [10, 10, 10]

    draw_ball(start_point, color=[1, 0, 0, 1])
    draw_ball(target_point, color=[0, 1, 0, 1])

    occ_file_path = "./data/occ_array.npy"
    occ_array = np.load(occ_file_path)

    # 将 3D occupancy array 转换为点云
    points = []
    for x in tqdm(range(occ_array.shape[0])):
        for y in range(occ_array.shape[1]):
            for z in range(occ_array.shape[2]):
                if occ_array[x, y, z] == 1:
                    points.append([x, y, z])

    # 创建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    # 可视化点云
    o3d.visualization.draw_geometries([pcd], window_name="3D Occupancy Visualization", width=800, height=600)

    print("Finished...")


if __name__ == "__main__":
    main()
