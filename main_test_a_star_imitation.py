import time
import torch

from main_test_a_star import initialize_environment, perform_path_search, reset_environment, control_loop, save_data
from utils_model import TrajectoryPredictor

RENDER = False
DEVICE = torch.device("cpu")
print("DEVICE: ", DEVICE)


def load_model(model_path):
    """加载训练好的模型"""
    print("Loading model...")
    model = TrajectoryPredictor().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def main():
    env, dilated_occ_index = initialize_environment()
    model_path = './data/models/model_epoch_1.pth'  # 替换为实际的模型文件路径
    model = load_model(model_path)
    print("Looping...")
    while True:
        obs_ma, save_flag, list_drone_pos, list_target_pos, drone_pos, vel_x_last, vel_y_last, target_pos = reset_environment(
            env)
        path_result = perform_path_search(dilated_occ_index, drone_pos, target_pos)
        if path_result is None:
            continue
        tensor_drone_pos = torch.tensor([drone_pos], dtype=torch.float32)
        tensor_target_pos = torch.tensor([target_pos], dtype=torch.float32)
        outputs = model(tensor_drone_pos, tensor_target_pos)
        if RENDER:
            print("outputs: ", outputs)
            # TODO: vis outputs
        save_flag, list_drone_pos, list_target_pos = control_loop(env, obs_ma, dilated_occ_index, drone_pos, vel_x_last,
                                                                  vel_y_last, list_drone_pos, list_target_pos)
        save_data(save_flag, list_drone_pos, list_target_pos)
    print("Finished...")
    time.sleep(666)


if __name__ == "__main__":
    main()
