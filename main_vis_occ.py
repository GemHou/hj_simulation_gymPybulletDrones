import numpy as np
import open3d as o3d


def main():
    occ_file_path = "./data/occ_array.npy"
    occ_array = np.load(occ_file_path)

    # 将 3D occupancy array 转换为点云
    points = []
    for x in range(occ_array.shape[0]):
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
