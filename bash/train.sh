# 生成日期文件夹（例如 nohup_logs/2025-04-20）
LOG_DATE_DIR="train_logs/$(date +%Y-%m-%d)"
mkdir -p "$LOG_DATE_DIR"

# 生成带时间戳的日志文件名（例如 14-30-00.log）
LOG_FILE="$LOG_DATE_DIR/$(date +%H-%M-%S).log"

################################################## CartWheel ##################################################
# nohup python scripts/rsl_rl/train.py \
#         --task Unitree-G1-29dof-Mimic-CartWheel --headless \
#         --device cuda:0 \
#         > "$LOG_FILE" 2>&1 &
# python3 scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-CartWheel --experiment_name CartWheel --headless --num_envs 4 --device cuda:0 \
#         --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"


# nohup python scripts/rsl_rl/train.py \
#         --task Unitree-G1-29dof-Mimic-CartWheel_Rough --experiment_name CartWheel \
#         --resume --load_run 2025-12-29_09-08-20 \
#         --device cuda:1 --headless \
#         > "$LOG_FILE" 2>&1 &



################################################## Chartz ##################################################
# nohup python scripts/rsl_rl/train.py \
#         --task Unitree-G1-29dof-Mimic-Chartz --headless \
#         --device cuda:1 \
#         > "$LOG_FILE" 2>&1 &

# nohup python scripts/rsl_rl/train.py \
#         --task Unitree-G1-29dof-Mimic-Chartz_Rough --headless \
#         --resume --load_run 2025-12-26_15-45-44 \
#         --device cuda:1 \
#         > "$LOG_FILE" 2>&1 &



################################################## Worry ##################################################
# nohup python scripts/rsl_rl/train.py \
#         --task Unitree-G1-29dof-Mimic-Worry --experiment_name Worry \
#         --device cuda:1 --headless \
#         > "$LOG_FILE" 2>&1 &

# python3 scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-Worry --experiment_name Worry --headless --num_envs 4 --device cuda:0 \
#         --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"



################################################## SideKick ##################################################
# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof-Mimic-SideKick --experiment_name SideKick \
#         --device cuda:1 --headless \
#         > "$LOG_FILE" 2>&1 &
# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof-Mimic-SideKick_Rough --experiment_name SideKick \
#         --resume --load_run 2026-01-09_11-22-33 \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &



################################################## CartWheel ##################################################
# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof-Mimic-CartWheel --experiment_name CartWheel \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &
# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof-Mimic-CartWheel_Rough --experiment_name CartWheel \
#         --resume --load_run 2026-01-20_11-49-11 \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &



# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof-Mimic-Chartz --experiment_name Chartz \
#         --run_name old_urdf \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &
# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof-Mimic-Chartz_Rough --experiment_name Chartz \
#         --run_name old_urdf \
#         --resume --load_run 2026-02-03_20-28-04_old_urdf \
#         --device cuda:0 --headless --max_iteration 100000 \
#         > "$LOG_FILE" 2>&1 &
# python3 scripts/rsl_rl/train.py --task GLR-M3-23dof-Mimic-SideKick --experiment_name SideKick --num_envs 4 --device cuda:0 \
#         --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"

################################################## Walk ##################################################
# python3 scripts/rsl_rl/train.py --task GLR-M3-23dof-Velocity --experiment_name Test --num_envs 4 --device cuda:0 --headless \
#         --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"

# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof-Velocity --experiment_name Walk \
#         --resume --load_run 2026-01-23_15-54-01 \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &

# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof-Velocity --experiment_name Walk \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &
# python3 scripts/rsl_rl/train.py --task GLR-M3-23dof_parallel-Velocity --experiment_name Test --num_envs 4 --device cuda:0 --headless \
#         --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"

# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof-Velocity_Rough --experiment_name Walk \
#         --resume --load_run 2026-02-04_16-53-06_old_urdf \
#         --run_name RoughResume \
#         --device cuda:1 --headless \
#         > "$LOG_FILE" 2>&1 &

# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof-Velocity_Rough --experiment_name Walk \
#         --resume --load_run 2026-02-04_16-53-06_old_urdf \
#         --run_name RoughResume \
#         --device cuda:1 --headless \
#         > "$LOG_FILE" 2>&1 &

# python3 scripts/rsl_rl/train.py --task GLR-M3-23dof-Velocity_Rough --experiment_name Test --num_envs 4 --device cuda:0 --headless \
#         --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"

# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof_parallel-Velocity --experiment_name Walk_Parallel \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &

# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof_parallel-Velocity --experiment_name Walk_Parallel \
#         --resume --load_run 2026-01-23_15-54-08 --run_name NewPD \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &


# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof_parallel-Mimic-Chartz --experiment_name Chartz_Parallel \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &

# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-23dof_parallel-Mimic-Chartz_Rough --experiment_name Chartz_Parallel \
#         --resume --load_run 2026-02-04_15-12-51 --run_name RoughResume \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &


# nohup python scripts/rsl_rl/train.py \
#         --task GLR-M3-12dof-Velocity --experiment_name 12dof_Walk \
#         --device cuda:0 --headless \
#         > "$LOG_FILE" 2>&1 &
# python3 scripts/rsl_rl/train.py --task GLR-M3-12dof-Velocity --experiment_name Test --num_envs 4 --device cuda:0 --headless \
#         --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"

nohup python scripts/rsl_rl/train.py \
        --task GLR-M3-23dof-Velocity --experiment_name 23dof_Walk \
        --device cuda:1 --headless \
        > "$LOG_FILE" 2>&1 &
