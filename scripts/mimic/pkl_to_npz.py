import pickle
import numpy as np
import argparse
import os
from scipy import interpolate  # 用于插值处理帧数差异

def convert_pkl_to_npz(pkl_path, npz_path=None, target_frames=None):
    """
    将PKL文件转换为NPZ格式
    
    参数:
        pkl_path: 输入的PKL文件路径
        npz_path: 输出的NPZ文件路径（如果为None，则自动生成）
        target_frames: 目标帧数（如果为None，则保持原帧数）
    """
    # 加载PKL文件
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    print(f"📥 加载PKL文件: {pkl_path}")
    print(f"  原始帧数: {pkl_data['root_pos'].shape[0]}")
    print(f"  原始fps: {pkl_data['fps']}")
    
    # 确定输出路径
    if npz_path is None:
        base_name = os.path.splitext(pkl_path)[0]
        npz_path = base_name + ".npz"
    
    # 获取原始数据维度
    num_frames = pkl_data['root_pos'].shape[0]
    num_joints = pkl_data['dof_pos'].shape[1]
    
    # 如果需要调整帧数
    if target_frames is not None and target_frames != num_frames:
        print(f"🔄 调整帧数: {num_frames} -> {target_frames}")
        num_frames = target_frames
    
    # 创建NPZ数据结构
    npz_data = {}
    
    # 1. fps - 转换为整数
    npz_data['fps'] = np.array([int(round(pkl_data['fps']))], dtype=np.int64)
    
    # 2. joint_pos - 使用dof_pos
    # 注意：这里可能需要检查维度是否匹配
    if target_frames is not None and target_frames != pkl_data['dof_pos'].shape[0]:
        # 插值到目标帧数
        original_frames = pkl_data['dof_pos'].shape[0]
        x_original = np.linspace(0, 1, original_frames)
        x_target = np.linspace(0, 1, target_frames)
        
        joint_pos_resampled = np.zeros((target_frames, num_joints))
        for j in range(num_joints):
            interp_func = interpolate.interp1d(
                x_original, pkl_data['dof_pos'][:, j], 
                kind='linear', fill_value='extrapolate'
            )
            joint_pos_resampled[:, j] = interp_func(x_target)
        
        npz_data['joint_pos'] = joint_pos_resampled.astype(np.float32)
    else:
        npz_data['joint_pos'] = pkl_data['dof_pos'].astype(np.float32)
    
    # 3. joint_vel - 从joint_pos计算速度（差分）
    # 使用中心差分，边界使用前向/后向差分
    joint_pos = npz_data['joint_pos']
    joint_vel = np.zeros_like(joint_pos)
    
    # 内部点使用中心差分
    if num_frames > 2:
        joint_vel[1:-1] = (joint_pos[2:] - joint_pos[:-2]) / 2.0
    
    # 边界点使用前向/后向差分
    if num_frames > 1:
        joint_vel[0] = joint_pos[1] - joint_pos[0]
        joint_vel[-1] = joint_pos[-1] - joint_pos[-2]
    
    # 乘以fps得到实际速度（单位：单位/秒）
    joint_vel *= npz_data['fps'][0]
    npz_data['joint_vel'] = joint_vel.astype(np.float32)
    
    # 4. body_pos_w - 世界坐标系下的身体位置
    # 根据你的数据，可能只有根位置，需要扩展为30个身体部位
    num_bodies = 30  # 根据目标NPZ格式
    
    if target_frames is not None and target_frames != pkl_data['root_pos'].shape[0]:
        # 插值根位置
        original_frames = pkl_data['root_pos'].shape[0]
        x_original = np.linspace(0, 1, original_frames)
        x_target = np.linspace(0, 1, target_frames)
        
        root_pos_resampled = np.zeros((target_frames, 3))
        for j in range(3):
            interp_func = interpolate.interp1d(
                x_original, pkl_data['root_pos'][:, j], 
                kind='linear', fill_value='extrapolate'
            )
            root_pos_resampled[:, j] = interp_func(x_target)
        
        root_pos = root_pos_resampled
    else:
        root_pos = pkl_data['root_pos']
    
    # 创建body_pos_w：将根位置复制到所有身体部位
    # 注意：这是一个简化处理，实际应用中可能需要更复杂的映射
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_pos_w[:, 0, :] = root_pos  # 第0个身体部位使用根位置
    
    # 其他身体部位可以根据需要设置，这里简单设为0
    # 在实际应用中，你可能需要根据dof_pos计算其他关节的位置
    
    npz_data['body_pos_w'] = body_pos_w
    
    # 5. body_quat_w - 世界坐标系下的身体旋转（四元数）
    # 根据你的数据，只有根旋转
    if target_frames is not None and target_frames != pkl_data['root_rot'].shape[0]:
        # 插值根旋转（四元数需要特殊处理）
        original_frames = pkl_data['root_rot'].shape[0]
        x_original = np.linspace(0, 1, original_frames)
        x_target = np.linspace(0, 1, target_frames)
        
        root_rot_resampled = np.zeros((target_frames, 4))
        for j in range(4):
            interp_func = interpolate.interp1d(
                x_original, pkl_data['root_rot'][:, j], 
                kind='linear', fill_value='extrapolate'
            )
            root_rot_resampled[:, j] = interp_func(x_target)
        
        # 归一化四元数
        norms = np.linalg.norm(root_rot_resampled, axis=1, keepdims=True)
        root_rot_resampled = root_rot_resampled / np.where(norms > 0, norms, 1.0)
        
        # 将xyzw顺序转换为wxyz顺序
        # xyzw -> wxyz: [x, y, z, w] -> [w, x, y, z]
        root_rot_resampled = np.roll(root_rot_resampled, shift=1, axis=1)
        root_rot = root_rot_resampled
    else:
        # 直接转换顺序：xyzw -> wxyz
        root_rot = pkl_data['root_rot']
        # xyzw -> wxyz: [x, y, z, w] -> [w, x, y, z]
        root_rot = np.roll(root_rot, shift=1, axis=1)

    # 创建body_quat_w
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[:, 0, :] = root_rot  # 第0个身体部位使用根旋转

    # 其他身体部位设为单位四元数 [1, 0, 0, 0] (wxyz顺序)
    body_quat_w[:, 1:, :] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    npz_data['body_quat_w'] = body_quat_w
    
    # 6. body_lin_vel_w - 世界坐标系下的身体线性速度
    # 从body_pos_w计算
    body_lin_vel_w = np.zeros_like(body_pos_w)
    
    if num_frames > 2:
        body_lin_vel_w[1:-1] = (body_pos_w[2:] - body_pos_w[:-2]) / 2.0
    
    if num_frames > 1:
        body_lin_vel_w[0] = body_pos_w[1] - body_pos_w[0]
        body_lin_vel_w[-1] = body_pos_w[-1] - body_pos_w[-2]
    
    body_lin_vel_w *= npz_data['fps'][0]
    npz_data['body_lin_vel_w'] = body_lin_vel_w.astype(np.float32)
    
    # 7. body_ang_vel_w - 世界坐标系下的身体角速度
    # 从body_quat_w计算（简化处理）
    body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    npz_data['body_ang_vel_w'] = body_ang_vel_w
    
    # 保存为NPZ文件
    np.savez_compressed(npz_path, **npz_data)
    
    print(f"💾 保存NPZ文件: {npz_path}")
    
    # 验证保存的文件
    print("\n✅ 转换完成！验证输出文件：")
    verify_npz_file(npz_path)
    
    return npz_path

def verify_npz_file(npz_path):
    """验证NPZ文件内容"""
    try:
        data = np.load(npz_path)
        print(f"📁 加载成功: {npz_path}")
        print(f"📊 总数组数: {len(data.files)}")
        print("=" * 50)
        print(f"键名: {list(data.files)}")
        print("=" * 50)
        
        total_size = 0
        for key in data.files:
            arr = data[key]
            size_bytes = arr.nbytes
            total_size += size_bytes
            
            print(f"\n🔹 Key: {key}")
            print(f"   形状: {arr.shape}")
            print(f"   数据类型: {arr.dtype}")
            print(f"   元素数: {arr.size:,}")
            print(f"   内存: {size_bytes:,} bytes ({size_bytes/1024/1024:.2f} MB)")
        
        print(f"\n📊 总文件大小: {total_size/1024/1024:.2f} MB")
        data.close()
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='将PKL文件转换为NPZ格式')
    parser.add_argument('--motion_file', type=str, required=True,
                       help='输入的PKL文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出的NPZ文件路径（可选）')
    parser.add_argument('--target_frames', type=int, default=None,
                       help='目标帧数（可选，用于调整帧数）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not args.motion_file.endswith('.pkl'):
        print(f"⚠️  警告：输入文件不是.pkl格式: {args.motion_file}")
    
    # 执行转换
    try:
        convert_pkl_to_npz(
            pkl_path=args.motion_file,
            npz_path=args.output,
            target_frames=args.target_frames
        )
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()