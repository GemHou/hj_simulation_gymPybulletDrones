import time
import numpy as np

from utils_drone import HjAviary
from main_a_star import dilate_obstacles


def main():
    print("Load env...")
    env = HjAviary(gui=True)  # , ctrl_freq=10, pyb_freq=100

    print("Load a-star...")
    occ_file_path = "./data/occ_array.npy"
    occ_index = np.load(occ_file_path)
    dilated_occ_index = dilate_obstacles(occ_index, dilation_radius=3)

    print("Looping...")
    if True:
        obs_ma, info = env.reset(percent=1)

        if True:
            env.render()
            time.sleep(1 / 30)  # * 10

    time.sleep(666)




if __name__ == "__main__":
    main()