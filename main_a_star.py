import numpy as np
import open3d as o3d
from tqdm import tqdm


def draw_ball(pos, color=None):
    # 定义球体的参数
    if color is None:
        color = [1, 0, 0]  # 默认颜色为红色
    radius = 2  # 球体的半径

    # 创建球体
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(pos)  # 将球体移动到指定位置
    sphere.paint_uniform_color(color)  # 设置球体颜色

    return sphere


def main():
    start_point = [0, 0, 0]
    target_point = [0, 40, 10]

    # 绘制球体
    start_sphere = draw_ball(start_point, color=[1, 0, 0])  # 红色球体
    target_sphere = draw_ball(target_point, color=[0, 1, 0])  # 绿色球体

    occ_file_path = "./data/occ_array.npy"
    occ_array = np.load(occ_file_path)

    # 将 3D occupancy array 转换为点云
    points = []
    for x in tqdm(range(occ_array.shape[0])):
        for y in range(occ_array.shape[1]):
            for z in range(occ_array.shape[2]):
                if occ_array[x, y, z] == 1:
                    points.append([(x-128*3) * 0.25, (y-128*3) * 0.25, z * 0.25])

    # 创建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    # 可视化点云和球体
    o3d.visualization.draw_geometries(
        [pcd, start_sphere, target_sphere],
        window_name="3D Occupancy Visualization",
        width=800,
        height=600
    )

    print("Finished...")


if __name__ == "__main__":
    main()