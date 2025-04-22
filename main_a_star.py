import numpy as np
import open3d as o3d
from tqdm import tqdm
import heapq

# 定义球体绘制函数
def draw_ball(pos, color=None):
    if color is None:
        color = [1, 0, 0]  # 默认颜色为红色
    radius = 2  # 球体的半径
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(pos)  # 将球体移动到指定位置
    sphere.paint_uniform_color(color)  # 设置球体颜色
    return sphere

# 定义3D A*路径搜索算法
def a_star_3d(start, goal, occ_index):
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    neighbors = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1),
        (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1),
        (-1, -1, 1), (-1, -1, -1)
    ]

    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in neighbors:
            neighbor = (current[0] + neighbor[0], current[1] + neighbor[1], current[2] + neighbor[2])
            if not (0 <= neighbor[0] < occ_index.shape[0] and
                    0 <= neighbor[1] < occ_index.shape[1] and
                    0 <= neighbor[2] < occ_index.shape[2]):
                continue
            if occ_index[neighbor[0], neighbor[1], neighbor[2]] == 1:
                continue

            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None

# 障碍物膨胀函数
def dilate_obstacles(occ_index, dilation_radius=1):
    """
    对障碍物进行膨胀处理。
    :param occ_index: 三维占用网格数组
    :param dilation_radius: 膨胀半径，默认为1
    :return: 膨胀后的占用网格数组
    """
    dilated_occ_index = occ_index.copy()
    shape = occ_index.shape
    for x in tqdm(range(shape[0])):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if occ_index[x, y, z] == 1:
                    for dx in range(-dilation_radius, dilation_radius + 1):
                        for dy in range(-dilation_radius, dilation_radius + 1):
                            for dz in range(-dilation_radius, dilation_radius + 1):
                                nx, ny, nz = x + dx, y + dy, z + dz
                                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                                    dilated_occ_index[nx, ny, nz] = 1
    return dilated_occ_index

# 可视化函数
def vis(occ_index, path_index, start_point, target_point):
    start_sphere = draw_ball(start_point, color=[1, 0, 0])  # 红色球体
    target_sphere = draw_ball(target_point, color=[0, 1, 0])  # 绿色球体

    # points = []
    # for x in tqdm(range(occ_index.shape[0])):
    #     for y in range(occ_index.shape[1]):
    #         for z in range(occ_index.shape[2]):
    #             if occ_index[x, y, z] == 1:
    #                 points.append([(x - 128 * 3) * 0.25, (y - 128 * 3) * 0.25, z * 0.25])
    # 假设 occ_index 是一个三维布尔数组
    # 获取满足条件的索引
    indices = np.argwhere(occ_index == 1)

    # 转换坐标
    points = np.zeros((indices.shape[0], 3))
    points[:, 0] = (indices[:, 0] - 128 * 3) * 0.25
    points[:, 1] = (indices[:, 1] - 128 * 3) * 0.25
    points[:, 2] = indices[:, 2] * 0.25


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    if path_index is not None:
        path_points = []
        for point_index in path_index:
            path_points.append([(point_index[0] - 128 * 3) * 0.25, (point_index[1] - 128 * 3) * 0.25, point_index[2] * 0.25])
        path_pcd = o3d.geometry.PointCloud()
        path_pcd.points = o3d.utility.Vector3dVector(path_points)
        path_pcd.paint_uniform_color([1, 0, 1])  # 路径颜色为紫色
    else:
        print("No path found.")
        path_pcd = None

    o3d.visualization.draw_geometries(
        [pcd, start_sphere, target_sphere, path_pcd],
        window_name="3D Occupancy Visualization with Path",
        width=800,
        height=600
    )

# 主函数
def main():
    start_pos = [np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(0, 10)]
    target_pos = [np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(0, 10)]

    occ_file_path = "./data/occ_array.npy"
    occ_index = np.load(occ_file_path)

    # 对障碍物进行膨胀处理
    dilated_occ_index = dilate_obstacles(occ_index, dilation_radius=2)

    start_index = [int(start_pos[0] / 0.25 + 128 * 3), int(start_pos[1] / 0.25 + 128 * 3), int(start_pos[2] / 0.25)]
    target_index = [int(target_pos[0] / 0.25 + 128 * 3), int(target_pos[1] / 0.25 + 128 * 3), int(target_pos[2] / 0.25)]

    path_index = a_star_3d(tuple(start_index), tuple(target_index), dilated_occ_index)

    vis(dilated_occ_index, path_index, start_pos, target_pos)

    print("Finished...")

if __name__ == "__main__":
    main()