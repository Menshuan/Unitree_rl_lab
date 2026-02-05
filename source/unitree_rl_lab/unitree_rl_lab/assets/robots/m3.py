import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

ASSET_DIR = "/home/robot/wrk_zwx/Unitree/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/description"

INTERTIA_OF_THE_MOTOR_90_25 = 78 * 1e-6
INTERTIA_OF_THE_MOTOR_70_25 = 47 * 1e-6

ARMATURE_90_25 = INTERTIA_OF_THE_MOTOR_90_25 * 25**2
ARMATURE_70_25 = INTERTIA_OF_THE_MOTOR_70_25 * 25**2

NATURAL_FREQ = 10 * 2.0 * 3.1415926535

DAMPING_RATIO = 2.0

STIFFNESS_90_25 = ARMATURE_90_25 * NATURAL_FREQ**2
STIFFNESS_70_25 = ARMATURE_70_25 * NATURAL_FREQ**2

DAMPING_90_25 = 2.0 * DAMPING_RATIO * ARMATURE_90_25 * NATURAL_FREQ + 2
DAMPING_70_25 = 2.0 * DAMPING_RATIO * ARMATURE_70_25 * NATURAL_FREQ + 1.5

print("======================================= M3 Parameters =======================================")
print("STIFFNESS_90_25:", STIFFNESS_90_25)
print("DAMPING_90_25:", DAMPING_90_25)
print("STIFFNESS_70_25:", STIFFNESS_70_25)
print("DAMPING_70_25:", DAMPING_70_25) 
print("ARMATURE_90_25:", ARMATURE_90_25)
print("ARMATURE_70_25:", ARMATURE_70_25)
print("=============================================================================================")

@configclass
class GLRArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9

M3_CONFIG = GLRArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/m3_description/urdf/m3_23dof.urdf",
        # asset_path=f"{ASSET_DIR}/m3_description_old/urdf/M3.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_pitch_joint": -0.2,
            ".*_knee_joint": 0.4,
            ".*_ankle_pitch_joint": -0.2,
            ".*_elbow_pitch_joint": 0.0,
            ".*_elbow_yaw_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 105.0,
                ".*_hip_roll_joint": 195.0,
                ".*_hip_pitch_joint": 195.0,
                ".*_knee_joint": 195.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 11.1,
                ".*_hip_roll_joint": 11.1,
                ".*_hip_pitch_joint": 12.5,
                ".*_knee_joint": 11.1,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_90_25,
                ".*_hip_roll_joint": STIFFNESS_90_25,
                ".*_hip_yaw_joint": STIFFNESS_70_25,
                ".*_knee_joint": STIFFNESS_90_25,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_90_25,
                ".*_hip_roll_joint": DAMPING_90_25,
                ".*_hip_yaw_joint": DAMPING_70_25,
                ".*_knee_joint": DAMPING_90_25,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_90_25,
                ".*_hip_roll_joint": ARMATURE_90_25,
                ".*_hip_yaw_joint": ARMATURE_70_25,
                ".*_knee_joint": ARMATURE_90_25,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=105.0,
            velocity_limit_sim=12.5,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=STIFFNESS_70_25,
            damping=DAMPING_70_25,
            armature=ARMATURE_70_25,
        ),
        # "waist": ImplicitActuatorCfg(
        #     effort_limit_sim=105.0,
        #     velocity_limit_sim=12.5,
        #     joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        #     stiffness=STIFFNESS_70_25,
        #     damping=DAMPING_70_25,
        #     armature=ARMATURE_70_25,
        # ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=105.0,
            velocity_limit_sim=12.5,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_70_25,
            damping=DAMPING_70_25,
            armature=ARMATURE_70_25,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 105.0,
                ".*_shoulder_roll_joint": 105.0,
                ".*_shoulder_yaw_joint": 105.0,
                ".*_elbow_pitch_joint": 105.0,
                ".*_elbow_yaw_joint": 105.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 12.5,
                ".*_shoulder_roll_joint": 12.5,
                ".*_shoulder_yaw_joint": 12.5,
                ".*_elbow_pitch_joint": 12.5,
                ".*_elbow_yaw_joint": 12.5,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_70_25,
                ".*_shoulder_roll_joint": STIFFNESS_70_25,
                ".*_shoulder_yaw_joint": STIFFNESS_70_25,
                ".*_elbow_pitch_joint": STIFFNESS_70_25,
                ".*_elbow_yaw_joint": STIFFNESS_70_25,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_70_25,
                ".*_shoulder_roll_joint": DAMPING_70_25,
                ".*_shoulder_yaw_joint": DAMPING_70_25,
                ".*_elbow_pitch_joint": DAMPING_70_25,
                ".*_elbow_yaw_joint": DAMPING_70_25,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_70_25,
                ".*_shoulder_roll_joint": ARMATURE_70_25,
                ".*_shoulder_yaw_joint": ARMATURE_70_25,
                ".*_elbow_pitch_joint": ARMATURE_70_25,
                ".*_elbow_yaw_joint": ARMATURE_70_25,
            },
        ),
    },
    joint_sdk_names=[
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
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_yaw_joint",
    ],
)

M3_ACTION_SCALE = {}
for a in M3_CONFIG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            M3_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
print("ACTION SCALE :%s" % str(M3_ACTION_SCALE))


M3_VELOCITY_CONFIG = GLRArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        # asset_path=f"{ASSET_DIR}/m3_description/urdf/m3_23dof.urdf",
        asset_path=f"{ASSET_DIR}/m3_description_old/urdf/M3.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_pitch_joint": -0.2,
            ".*_knee_joint": 0.4,
            ".*_ankle_pitch_joint": -0.2,
            ".*_elbow_pitch_joint": 0.0,
            ".*_elbow_yaw_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 105.0,
                ".*_hip_roll_joint": 195.0,
                ".*_hip_pitch_joint": 195.0,
                ".*_knee_joint": 195.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 11.1,
                ".*_hip_roll_joint": 11.1,
                ".*_hip_pitch_joint": 12.5,
                ".*_knee_joint": 11.1,
            },
            stiffness={
                ".*_hip_pitch_joint": 200,
                ".*_hip_roll_joint": 200,
                ".*_hip_yaw_joint": 100,
                ".*_knee_joint": 200,
            },
            damping={
                ".*_hip_pitch_joint": 8.7,
                ".*_hip_roll_joint": 8.7,
                ".*_hip_yaw_joint": 6.8,
                ".*_knee_joint": 8.7,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_90_25,
                ".*_hip_roll_joint": ARMATURE_90_25,
                ".*_hip_yaw_joint": ARMATURE_70_25,
                ".*_knee_joint": ARMATURE_90_25,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=105.0,
            velocity_limit_sim=12.5,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=150, #STIFFNESS_70_25,
            damping=5, # DAMPING_70_25,
            armature=ARMATURE_70_25,
        ),
        # "waist": ImplicitActuatorCfg(
        #     effort_limit_sim=105.0,
        #     velocity_limit_sim=12.5,
        #     joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        #     stiffness=STIFFNESS_70_25,
        #     damping=DAMPING_70_25,
        #     armature=ARMATURE_70_25,
        # ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=105.0,
            velocity_limit_sim=12.5,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=100,
            damping=6.8,
            armature=ARMATURE_70_25,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 105.0,
                ".*_shoulder_roll_joint": 105.0,
                ".*_shoulder_yaw_joint": 105.0,
                ".*_elbow_pitch_joint": 105.0,
                ".*_elbow_yaw_joint": 105.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 12.5,
                ".*_shoulder_roll_joint": 12.5,
                ".*_shoulder_yaw_joint": 12.5,
                ".*_elbow_pitch_joint": 12.5,
                ".*_elbow_yaw_joint": 12.5,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 100,
                ".*_shoulder_roll_joint": 100,
                ".*_shoulder_yaw_joint": 100,
                ".*_elbow_pitch_joint": 100,
                ".*_elbow_yaw_joint": 100,
            },
            damping={
                ".*_shoulder_pitch_joint": 6.8,
                ".*_shoulder_roll_joint": 6.8,
                ".*_shoulder_yaw_joint": 6.8,
                ".*_elbow_pitch_joint": 6.8,
                ".*_elbow_yaw_joint": 6.8,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_70_25,
                ".*_shoulder_roll_joint": ARMATURE_70_25,
                ".*_shoulder_yaw_joint": ARMATURE_70_25,
                ".*_elbow_pitch_joint": ARMATURE_70_25,
                ".*_elbow_yaw_joint": ARMATURE_70_25,
            },
        ),
    },
    joint_sdk_names=[
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
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_yaw_joint",
    ],
)