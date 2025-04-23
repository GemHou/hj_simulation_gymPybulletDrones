import numpy as np
from scipy.ndimage import binary_dilation


def dilate_obstacles(occ_index, dilation_radius=1):
    """
    对障碍物进行膨胀处理。
    :param occ_index: 三维占用网格数组
    :param dilation_radius: 膨胀半径，默认为1
    :return: 膨胀后的占用网格数组
    """
    # 创建一个膨胀结构元素，大小为 (2 * dilation_radius + 1) 的立方体
    structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1, 2 * dilation_radius + 1))

    # 使用 binary_dilation 进行膨胀
    dilated_occ_index = binary_dilation(occ_index, structure=structure).astype(occ_index.dtype)

    return dilated_occ_index


def main():
    occ_file_path = "./data/occ_index.npy"
    occ_index = np.load(occ_file_path)
    dilated_occ_index = dilate_obstacles(occ_index, dilation_radius=3)
    np.save("./data/dilated_occ_index.npy", dilated_occ_index)


if __name__ == "__main__":
    main()
