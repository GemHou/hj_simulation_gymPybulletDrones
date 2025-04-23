import time
import heapq
import numpy as np
import open3d as o3d
from tqdm import tqdm


# 定义球体绘制函数
def draw_ball(pos, color=None):
    if color is None:
        color = [1, 0, 0]  # 默认颜色为红色
    radius = 1  # 球体的半径
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(pos)  # 将球体移动到指定位置
    sphere.paint_uniform_color(color)  # 设置球体颜色
    return sphere


# 定义3D A*路径搜索算法
def a_star_3d(start, goal, occ_index):
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    # Check if start or goal point is occupied or invalid
    if occ_index[start[0], start[1], start[2]] == 1:
        print("Start point is occupied!")
        return None
    if occ_index[goal[0], goal[1], goal[2]] == 1:
        print("Goal point is occupied!")
        return None

    neighbors = [
        (1, 0, 0), (2, 0, 0),
        (-1, 0, 0), (-2, 0, 0),
        (0, 1, 0), (0, 2, 0),
        (0, -1, 0), (0, -2, 0),
        (0, 0, 1), (0, 0, 2),
        (0, 0, -1), (0, 0, -2),
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

    start_time = time.time()
    while open_list:
        if time.time() - start_time > 0.5:
            return None
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
            if (not (0 <= neighbor[0] < occ_index.shape[0] and
                     0 <= neighbor[1] < occ_index.shape[1] and
                     0 <= neighbor[2] < occ_index.shape[2]) or
                    occ_index[neighbor[0], neighbor[1], neighbor[2]] == 1):
                continue

            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    print("No path found!")
    return None


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
            path_points.append(
                [(point_index[0] - 128 * 3) * 0.25, (point_index[1] - 128 * 3) * 0.25, point_index[2] * 0.25])
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
    print("Loading...")
    start_pos = [np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(1, 10)]
    target_pos = [np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(1, 10)]
    start_index = [int(start_pos[0] / 0.25 + 128 * 3), int(start_pos[1] / 0.25 + 128 * 3), int(start_pos[2] / 0.25)]
    target_index = [int(target_pos[0] / 0.25 + 128 * 3), int(target_pos[1] / 0.25 + 128 * 3), int(target_pos[2] / 0.25)]

    dilated_occ_file_path = "./data/dilated_occ_index.npy"
    dilated_occ_index = np.load(dilated_occ_file_path)

    print("Processing path...")
    start_time = time.time()
    path_index = a_star_3d(tuple(start_index), tuple(target_index), dilated_occ_index)
    print("search time: ", time.time() - start_time)

    print("Rendering...")
    vis(dilated_occ_index, path_index, start_pos, target_pos)

    print("Finished...")


if __name__ == "__main__":
    main()
