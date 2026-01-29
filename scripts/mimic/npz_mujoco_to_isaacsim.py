# convert_mujoco_to_isaac.py
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Convert Mujoco motion NPZ to IsaacSim format")
parser.add_argument("--input", type=str, required=True, help="Input NPZ file (Mujoco order)")
parser.add_argument("--output", type=str, default=None, help="Output NPZ file (Isaac order)")
parser.add_argument("--robot", type=str, choices=["g1", "m3"], required=True, help="Robot type: G1 or M3")

args = parser.parse_args()


# Isaac-Sim 里 G1 的关节顺序
G1_ISAAC_JOINT_NAMES = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw",
    "left_hip_roll", "right_hip_roll", "waist_roll",
    "left_hip_yaw", "right_hip_yaw", "waist_pitch",
    "left_knee", "right_knee", 
    "left_shoulder_pitch", "right_shoulder_pitch",
    "left_ankle_pitch", "right_ankle_pitch",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_ankle_roll", "right_ankle_roll",
    "left_shoulder_yaw", "right_shoulder_yaw",
    "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll",
    "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_yaw", "right_wrist_yaw",
]

# 2. MuJoCo 关节顺序
G1_MUJOCO_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_pitch", "waist_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# 1. Isaac-Sim 里 M3 的关节顺序
M3_ISAAC_JOINT_NAMES = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw",
    "left_hip_roll", "right_hip_roll",
    "left_shoulder_pitch", "right_shoulder_pitch",
    "left_hip_yaw", "right_hip_yaw",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_knee", "right_knee", 
    "left_shoulder_yaw", "right_shoulder_yaw",
    "left_ankle_roll", "right_ankle_roll",
    "left_elbow_pitch", "right_elbow_pitch",
    "left_ankle_pitch", "right_ankle_pitch",
    "left_elbow_yaw", "right_elbow_yaw",
]

# 2. MuJoCo 关节顺序
M3_MUJOCO_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_roll", "left_ankle_pitch",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_roll", "right_ankle_pitch",
    "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow_pitch", "left_elbow_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow_pitch", "right_elbow_yaw",
]



def create_mujoco_to_isaac_mapping(ISAAC_JOINT_NAMES, MUJOCO_JOINT_NAMES):
    """
    创建从Mujoco顺序到Isaac顺序的映射
    """
    mujoco_to_isaac = []
    for isaac_joint in ISAAC_JOINT_NAMES:
        if isaac_joint in MUJOCO_JOINT_NAMES:
            mujoco_idx = MUJOCO_JOINT_NAMES.index(isaac_joint)
            mujoco_to_isaac.append(mujoco_idx)
        else:
            # 如果Isaac关节在Mujoco中不存在，使用-1标记
            mujoco_to_isaac.append(-1)
            print(f"警告: Isaac关节 '{isaac_joint}' 在Mujoco关节列表中未找到")
    
    # 检查是否有Mujoco关节在Isaac中不存在（反向检查）
    for mujoco_joint in MUJOCO_JOINT_NAMES:
        if mujoco_joint not in ISAAC_JOINT_NAMES:
            print(f"警告: Mujoco关节 '{mujoco_joint}' 在Isaac关节列表中未找到")
    
    return mujoco_to_isaac

def convert_npz_file(input_path, output_path=None):
    """
    转换NPZ文件从Mujoco顺序到Isaac顺序
    
    Args:
        input_path: 输入的NPZ文件路径（Mujoco顺序）
        output_path: 输出的NPZ文件路径，如果不指定则自动生成
    """
    # 加载原始数据
    data = np.load(input_path)

    if args.robot == "g1":
        ISAAC_JOINT_NAMES = G1_ISAAC_JOINT_NAMES
        MUJOCO_JOINT_NAMES = G1_MUJOCO_JOINT_NAMES
    else:  # m3
        ISAAC_JOINT_NAMES = M3_ISAAC_JOINT_NAMES
        MUJOCO_JOINT_NAMES = M3_MUJOCO_JOINT_NAMES
    # 创建映射
    mapping = create_mujoco_to_isaac_mapping(ISAAC_JOINT_NAMES, MUJOCO_JOINT_NAMES)
    
    # 转换joint_pos和joint_vel
    if 'joint_pos' in data:
        mujoco_joint_pos = data['joint_pos']  # [timesteps, dof]
        # 创建新的数组
        isaac_joint_pos = np.zeros((mujoco_joint_pos.shape[0], len(ISAAC_JOINT_NAMES)), 
                                  dtype=mujoco_joint_pos.dtype)
        
        for i, mujoco_idx in enumerate(mapping):
            if mujoco_idx != -1:
                isaac_joint_pos[:, i] = mujoco_joint_pos[:, mujoco_idx]
        
        print(f"转换 joint_pos: {mujoco_joint_pos.shape} -> {isaac_joint_pos.shape}")
    
    if 'joint_vel' in data:
        mujoco_joint_vel = data['joint_vel']  # [timesteps, dof]
        # 创建新的数组
        isaac_joint_vel = np.zeros((mujoco_joint_vel.shape[0], len(ISAAC_JOINT_NAMES)), 
                                  dtype=mujoco_joint_vel.dtype)
        
        for i, mujoco_idx in enumerate(mapping):
            if mujoco_idx != -1:
                isaac_joint_vel[:, i] = mujoco_joint_vel[:, mujoco_idx]
        
        print(f"转换 joint_vel: {mujoco_joint_vel.shape} -> {isaac_joint_vel.shape}")
    
    # 生成输出文件名
    if output_path is None:
        if input_path.endswith('.npz'):
            output_path = input_path.replace('.npz', '_isaac.npz')
        else:
            output_path = input_path + '_isaac.npz'
    
    # 保存转换后的数据
    save_data = {}
    for key in data.files:
        if key == 'joint_pos':
            save_data[key] = isaac_joint_pos
        elif key == 'joint_vel':
            save_data[key] = isaac_joint_vel
        else:
            save_data[key] = data[key]
    
    np.savez(output_path, **save_data)
    print(f"转换完成！保存到: {output_path}")
    
    # 打印转换信息
    print(f"\n关节映射信息:")
    for i, (isaac_joint, mujoco_idx) in enumerate(zip(ISAAC_JOINT_NAMES, mapping)):
        if mujoco_idx != -1:
            print(f"  Isaac[{i}]: {isaac_joint:20s} <- Mujoco[{mujoco_idx}]: {MUJOCO_JOINT_NAMES[mujoco_idx]}")
    
    return output_path

def main():
    
    convert_npz_file(args.input, args.output)

if __name__ == "__main__":
    main()