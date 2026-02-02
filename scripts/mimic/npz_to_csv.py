"""
npz_to_csv.py - 将npz文件转换回CSV格式，处理关节顺序映射
"""

import argparse
import numpy as np
import torch

# 定义关节名称映射
G1_CSV_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

M3_CSV_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", 
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", 
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_elbow_yaw_joint",
]

G1_ISAAC_JOINT_NAMES = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw",
    "left_hip_roll", "right_hip_roll", "waist_roll",
    "left_hip_yaw", "right_hip_yaw", "waist_pitch",
    "left_knee", "right_knee", 
    "left_shoulder_pitch", "right_shoulder_pitch",
    "left_ankle_roll", "right_ankle_roll",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_ankle_pitch", "right_ankle_pitch",
    "left_shoulder_yaw", "right_shoulder_yaw",
    "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll",
    "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_yaw", "right_wrist_yaw",
]

M3_ISAAC_JOINT_NAMES = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw",
    "left_hip_roll", "right_hip_roll",               "left_shoulder_pitch", "right_shoulder_pitch",
    "left_hip_yaw", "right_hip_yaw",                 "left_shoulder_roll", "right_shoulder_roll",
    "left_knee", "right_knee",                       "left_shoulder_yaw", "right_shoulder_yaw",
    "left_ankle_pitch", "right_ankle_pitch",         "left_elbow_pitch", "right_elbow_pitch",
    "left_ankle_roll", "right_ankle_roll",           "left_elbow_yaw", "right_elbow_yaw",
]

parser = argparse.ArgumentParser(description="将npz文件转换回CSV格式，处理关节顺序映射")
parser.add_argument("--input_npz", type=str, required=True, help="输入的npz文件路径")
parser.add_argument("--output_csv", type=str, required=True, help="输出的csv文件路径")
parser.add_argument("--output_fps", type=int, default=50, help="csv的fps")
parser.add_argument("--input_fps", type=int, default=50, help="npz文件的fps")
parser.add_argument("--robot", type=str, required=True, choices=["g1", "m3"], help="机器人类型: G1或M3")

args = parser.parse_args()

if args.robot == "g1":
    ISAAC_JOINT_NAMES = G1_ISAAC_JOINT_NAMES
    CSV_JOINT_NAMES = G1_CSV_JOINT_NAMES
elif args.robot == "m3":  # m3
    ISAAC_JOINT_NAMES = M3_ISAAC_JOINT_NAMES
    CSV_JOINT_NAMES = M3_CSV_JOINT_NAMES
else:
    raise ValueError("不支持的机器人类型")

# 创建从Isaac关节名到CSV索引的映射
# 注意：Isaac关节名没有"_joint"后缀，CSV关节名有"_joint"后缀
def create_joint_mapping():
    """创建Isaac关节顺序到CSV关节顺序的映射"""
    mapping = []
    for isaac_joint in ISAAC_JOINT_NAMES:
        # 尝试找到对应的CSV关节名（添加"_joint"后缀）
        csv_joint = f"{isaac_joint}_joint"
        try:
            csv_index = CSV_JOINT_NAMES.index(csv_joint)
            mapping.append(csv_index)
        except ValueError:
            # 如果找不到，尝试其他可能的变体
            print(f"警告: 无法找到Isaac关节 '{isaac_joint}' 对应的CSV关节")
            # 使用-1表示未找到
            mapping.append(-1)
    
    # 验证映射
    valid_mapping = [i for i in mapping if i != -1]
    if len(valid_mapping) != len(CSV_JOINT_NAMES):
        print(f"警告: 映射不完整! CSV有{len(CSV_JOINT_NAMES)}个关节, 但只映射了{len(valid_mapping)}个")
    
    return mapping

# 创建反向映射（从CSV索引到Isaac索引）
def create_reverse_mapping():
    """创建CSV关节顺序到Isaac关节顺序的映射"""
    mapping = []
    for csv_joint in CSV_JOINT_NAMES:
        # 移除"_joint"后缀
        isaac_joint = csv_joint.replace("_joint", "")
        try:
            isaac_index = ISAAC_JOINT_NAMES.index(isaac_joint)
            mapping.append(isaac_index)
        except ValueError:
            print(f"警告: 无法找到CSV关节 '{csv_joint}' 对应的Isaac关节")
            mapping.append(-1)
    
    return mapping

def reorder_joints_from_isaac_to_csv(joint_data_isaac_order):
    """将Isaac顺序的关节数据重新排序为CSV顺序"""
    mapping = create_joint_mapping()
    num_frames = joint_data_isaac_order.shape[0]
    num_joints = len(CSV_JOINT_NAMES)
    
    joint_data_csv_order = np.zeros((num_frames, num_joints))
    
    for i, csv_index in enumerate(mapping):
        if csv_index != -1 and i < joint_data_isaac_order.shape[1]:
            joint_data_csv_order[:, csv_index] = joint_data_isaac_order[:, i]
    
    return joint_data_csv_order

def reorder_joints_from_csv_to_isaac(joint_data_csv_order):
    """将CSV顺序的关节数据重新排序为Isaac顺序"""
    mapping = create_reverse_mapping()
    num_frames = joint_data_csv_order.shape[0]
    num_joints = len(ISAAC_JOINT_NAMES)
    
    joint_data_isaac_order = np.zeros((num_frames, num_joints))
    
    for csv_index, isaac_index in enumerate(mapping):
        if isaac_index != -1 and csv_index < joint_data_csv_order.shape[1]:
            joint_data_isaac_order[:, isaac_index] = joint_data_csv_order[:, csv_index]
    
    return joint_data_isaac_order

def main():
    
    # 加载npz文件
    print(f"正在加载npz文件: {args.input_npz}")
    data = np.load(args.input_npz, allow_pickle=True)
    
    print("npz文件中的键:", list(data.keys()))
    
    # 查找可用的数据
    csv_data = None

    # 情况1: 有模拟器记录的数据
    if 'joint_pos' in data:
        print("找到关节位置数据")
        joint_pos_isaac = data['joint_pos']  # (T, 29) - Isaac顺序
        
        num_frames = joint_pos_isaac.shape[0]
        print(f"帧数: {num_frames}")
        print(f"关节位置形状 (Isaac顺序): {joint_pos_isaac.shape}")
        
        # 将Isaac顺序转换为CSV顺序
        print("将Isaac关节顺序转换为CSV关节顺序...")
        joint_pos_csv = reorder_joints_from_isaac_to_csv(joint_pos_isaac)
        print(f"转换后形状 (CSV顺序): {joint_pos_csv.shape}")
        
        # 获取root位置和旋转
        if 'body_pos_w' in data and 'body_quat_w' in data:
            print("找到root位置和旋转数据")
            
            # 提取root身体数据
            body_pos = data['body_pos_w']  # 可能是(T, num_bodies, 3)或(T, 3)
            body_quat = data['body_quat_w']  # 可能是(T, num_bodies, 4)或(T, 4)
            
            if len(body_pos.shape) == 3:  # (T, num_bodies, 3)
                base_pos = body_pos[:, 0, :]  # 第一个身体是root
                base_rot = body_quat[:, 0, :]  # 第一个身体
            else:  # (T, 3) 只有一个身体
                base_pos = body_pos
                base_rot = body_quat
            
            print(f"root位置形状: {base_pos.shape}")
            print(f"root旋转形状: {base_rot.shape}")
            
            # 转换四元数从WXYZ到XYZW
            print("将四元数从WXYZ转换为XYZW格式...")
            base_quat_xyzw = np.zeros_like(base_rot)
            base_quat_xyzw[:, 0] = base_rot[:, 1]  # X
            base_quat_xyzw[:, 1] = base_rot[:, 2]  # Y
            base_quat_xyzw[:, 2] = base_rot[:, 3]  # Z
            base_quat_xyzw[:, 3] = base_rot[:, 0]  # W
            
            # 组合数据: XYZ + QX QY QZ QW + 关节位置(CSV顺序)
            csv_data = np.concatenate([base_pos, base_quat_xyzw, joint_pos_csv], axis=1)
    
    # 情况2: 有直接的运动数据
    elif 'motion_base_poss' in data and 'motion_base_rots' in data and 'motion_dof_poss' in data:
        print("找到直接的运动数据")
        base_pos = data['motion_base_poss']  # (T, 3)
        base_rot = data['motion_base_rots']  # (T, 4) - WXYZ格式
        dof_pos = data['motion_dof_poss']    # (T, 29) - 需要确认顺序
        
        num_frames = base_pos.shape[0]
        print(f"帧数: {num_frames}")
        print(f"base_pos形状: {base_pos.shape}")
        print(f"base_rot形状: {base_rot.shape}")
        print(f"dof_pos形状: {dof_pos.shape}")
        
        # 假设dof_pos是Isaac顺序，转换为CSV顺序
        print("将Isaac关节顺序转换为CSV关节顺序...")
        joint_pos_csv = reorder_joints_from_isaac_to_csv(dof_pos)
        
        # 转换四元数从WXYZ到XYZW
        print("将四元数从WXYZ转换为XYZW格式...")
        base_quat_xyzw = np.zeros_like(base_rot)
        base_quat_xyzw[:, 0] = base_rot[:, 1]  # X
        base_quat_xyzw[:, 1] = base_rot[:, 2]  # Y
        base_quat_xyzw[:, 2] = base_rot[:, 3]  # Z
        base_quat_xyzw[:, 3] = base_rot[:, 0]  # W
        
        # 组合数据
        csv_data = np.concatenate([base_pos, base_quat_xyzw, joint_pos_csv], axis=1)
    
    else:
        print("错误: 无法找到所需的数据!")
        print("可用键:", list(data.keys()))
        return
    
    print(f"CSV数据形状: {csv_data.shape}")
    
    # 验证列数
    if args.robot == "g1":
        expected_columns = 3 + 4 + 29  # XYZ + QXQYQZQW + 29个关节
    elif args.robot == "m3":
        expected_columns = 3 + 4 + 23  # XYZ + QXQYQZQW + 23个关节
    if csv_data.shape[1] != expected_columns:
        print(f"警告: 期望{expected_columns}列，但实际有{csv_data.shape[1]}列")
        if csv_data.shape[1] < expected_columns:
            # 填充缺失的列
            padding = np.zeros((csv_data.shape[0], expected_columns - csv_data.shape[1]))
            csv_data = np.concatenate([csv_data, padding], axis=1)
        else:
            # 截断多余的列
            csv_data = csv_data[:, :expected_columns]
    
    # 如果需要重采样到原始fps
    if args.input_fps != args.output_fps:
        print(f"重采样: {args.input_fps} fps -> {args.output_fps} fps")
        num_frames = csv_data.shape[0]
        original_num_frames = int(num_frames * args.output_fps / args.input_fps)
        
        original_times = np.linspace(0, num_frames-1, original_num_frames)
        npz_times = np.arange(num_frames)
        
        csv_data_resampled = np.zeros((original_num_frames, csv_data.shape[1]))
        for i in range(csv_data.shape[1]):
            csv_data_resampled[:, i] = np.interp(original_times, npz_times, csv_data[:, i])
        
        csv_data = csv_data_resampled
        print(f"重采样后: {csv_data.shape}")

    # # ===== 在这里添加z坐标偏移 =====
    # print("正在调整根节点z坐标（+0.04米）...")
    # csv_data[:, 2] += 0.03  # 第3列是z坐标（索引为2）
    # # ============================

    # 保存CSV文件
    print(f"正在保存CSV文件: {args.output_csv}")
    np.savetxt(args.output_csv, csv_data, delimiter=',', fmt='%.6f')
    
    # 生成关节顺序对比文件
    order_file = args.output_csv.replace('.csv', '_joint_order.txt')
    with open(order_file, 'w') as f:
        f.write("关节顺序对比\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CSV文件中的关节顺序 (30 FPS):\n")
        f.write("-" * 40 + "\n")
        for i, joint in enumerate(CSV_JOINT_NAMES):
            f.write(f"列 {i+7}: {joint}\n")
        
        f.write("\n\nIsaac Sim中的关节顺序:\n")
        f.write("-" * 40 + "\n")
        for i, joint in enumerate(ISAAC_JOINT_NAMES):
            f.write(f"索引 {i}: {joint}\n")
        
        f.write("\n\n映射关系:\n")
        f.write("-" * 40 + "\n")
        f.write("CSV列 -> Isaac索引 -> CSV关节名 -> Isaac关节名\n")
        mapping = create_reverse_mapping()
        for csv_index, isaac_index in enumerate(mapping):
            if isaac_index != -1:
                f.write(f"列 {csv_index+7} -> Isaac索引 {isaac_index} -> {CSV_JOINT_NAMES[csv_index]} -> {ISAAC_JOINT_NAMES[isaac_index]}\n")
    
    print(f"转换完成! CSV已保存到: {args.output_csv}")
    print(f"关节顺序对比已保存到: {order_file}")
    
    # 显示一些示例数据
    print("\n前3行数据示例:")
    for i in range(min(3, csv_data.shape[0])):
        print(f"行 {i}: 位置[{csv_data[i, 0]:.3f}, {csv_data[i, 1]:.3f}, {csv_data[i, 2]:.3f}], "
              f"四元数[{csv_data[i, 3]:.3f}, {csv_data[i, 4]:.3f}, {csv_data[i, 5]:.3f}, {csv_data[i, 6]:.3f}], "
              f"第1个关节: {csv_data[i, 7]:.3f}")

if __name__ == "__main__":
    main()