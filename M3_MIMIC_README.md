## GMR
# 1. LAFAN数据库（BVH格式）：
    ”“”
    python scripts/bvh_to_robot.py --bvh_file <path_to_bvh_data> --robot m3 --save_path <path_to_save_robot_data.pkl> --rate_limit --format lafan
    “”“
# 2. AMASS数据库（SMPL格式）：
    """
    python scripts/smplx_to_robot.py --smplx_file <path_to_smplx_data> --robot m3 --save_path <path_to_save_robot_data.pkl> --rate_limit
    """

## MIMIC
# 1. 先通过GMR导出pkl文件至 /unitree_rl_lab/Retarget_GMR
# 2. 处理pkl文件：
    - 1. 检查pkl文件内容：
        """
        python3 scripts/mimic/check_pkl.py --motion_file ...
        """
    - 2. 通过pkl转换至npz:
        """
        python3 scripts/mimic/pkl_to_npz.py --motion_file ...
        """
# 3. 处理npz文件：
    - 1. 检查npz文件内容：
        """
        python3 scripts/mimic/check_npz.py --motion_file ...
        """
    - 2. 将npz从Mujoco顺序切换至isaacsim顺序：
        """
        python3 scripts/mimic/npz_mujoco_to_isaacsim.py --input ... --robot m3
        """
    - 3. 播放npz文件进行动作检查：
        """
        python3 scripts/mimic/replay_npz.py --robot m3 --file ... (default is g1)
        """
    - 4. 将npz文件转换至csv文件（同时进行帧率对齐）：
        """
        python3 scripts/mimic/npz_to_csv.py --input_npz ... --output_csv ... --input_fps  --output_fps  --robot m3
        """
    - 5. （可选）重新将csv转换至npz文件进行二次检查：
        """
        python3 scripts/mimic/csv_to_npz.py --input_file ... --input_fps  --output_name ... --output_fps  --robot m3
        """
# 4. 开始训练：
    - 1. 训练：
        """
        sh ./bash/train.sh
        """
    - 2. 检查播放：
        """
        sh ./bash/play.sh
        """