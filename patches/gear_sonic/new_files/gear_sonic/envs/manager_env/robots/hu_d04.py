# LimX HU_D04 humanoid robot configuration for SONIC/GR00T-WholeBodyControl.
# 31 DOF: 6 left leg + 6 right leg + 3 waist + 7 left arm + 7 right arm + 2 head

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

# ---------------------------------------------------------------------------
# Motor parameters derived from SRDF rotor specs (mass * gear_ratio^2)
# ---------------------------------------------------------------------------
# Hip/Knee: rotor mass = 0.000226 kg, gear ratio = 25
ARMATURE_HIP_KNEE = 0.000226 * 25.0**2  # = 0.14125

# Ankle/Waist: rotor mass = 0.0001424 kg, gear ratio = 36
ARMATURE_ANKLE_WAIST = 0.0001424 * 36.0**2  # = 0.1845504

# Shoulder/Elbow: rotor mass = 0.000141873 kg, gear ratio = 25
ARMATURE_SHOULDER_ELBOW = 0.000141873 * 25.0**2  # = 0.08867063

# Wrist/Head: rotor mass = 0.000019308 kg, gear ratio = 28.17
ARMATURE_WRIST_HEAD = 0.000019308 * 28.17**2  # = 0.01532178

# PD gain computation
# HU_D04 tuning: keep stiffness high for tip resistance; bump damping ratio
# to 4.0 so the saturated regime is well-damped (less oscillation when at torque cap).
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
NATURAL_FREQ_ANKLE_WAIST = 10 * 2.0 * 3.1415926535  # back to 10 Hz (high K for tip resistance)
DAMPING_RATIO = 4.0  # increased from 2.0 — more damping = less tip oscillation

STIFFNESS_HIP_KNEE = ARMATURE_HIP_KNEE * NATURAL_FREQ**2
STIFFNESS_ANKLE_WAIST = ARMATURE_ANKLE_WAIST * NATURAL_FREQ_ANKLE_WAIST**2
STIFFNESS_SHOULDER_ELBOW = ARMATURE_SHOULDER_ELBOW * NATURAL_FREQ**2
STIFFNESS_WRIST_HEAD = ARMATURE_WRIST_HEAD * NATURAL_FREQ**2

DAMPING_HIP_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_HIP_KNEE * NATURAL_FREQ
DAMPING_ANKLE_WAIST = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE_WAIST * NATURAL_FREQ_ANKLE_WAIST
DAMPING_SHOULDER_ELBOW = 2.0 * DAMPING_RATIO * ARMATURE_SHOULDER_ELBOW * NATURAL_FREQ
DAMPING_WRIST_HEAD = 2.0 * DAMPING_RATIO * ARMATURE_WRIST_HEAD * NATURAL_FREQ

# ---------------------------------------------------------------------------
# IsaacLab body ordering (BFS traversal of the URDF kinematic tree)
# Index 0 is the root body. Indices 1..31 map to DOFs 0..30.
# ---------------------------------------------------------------------------
HU_D04_ISAACLAB_JOINTS = [
    "base_link",                    # 0  (root)
    "left_hip_pitch_link",          # 1
    "right_hip_pitch_link",         # 2
    "waist_yaw_link",               # 3
    "left_hip_roll_link",           # 4
    "right_hip_roll_link",          # 5
    "waist_roll_link",              # 6
    "left_hip_yaw_link",            # 7
    "right_hip_yaw_link",           # 8
    "waist_pitch_link",             # 9
    "left_knee_link",               # 10
    "right_knee_link",              # 11
    "left_shoulder_pitch_link",     # 12
    "right_shoulder_pitch_link",    # 13
    "head_yaw_link",                # 14
    "left_ankle_pitch_link",        # 15
    "right_ankle_pitch_link",       # 16
    "left_shoulder_roll_link",      # 17
    "right_shoulder_roll_link",     # 18
    "head_pitch_link",              # 19
    "left_ankle_roll_link",         # 20
    "right_ankle_roll_link",        # 21
    "left_shoulder_yaw_link",       # 22
    "right_shoulder_yaw_link",      # 23
    "left_elbow_link",              # 24
    "right_elbow_link",             # 25
    "left_wrist_yaw_link",          # 26
    "right_wrist_yaw_link",         # 27
    "left_wrist_pitch_link",        # 28
    "right_wrist_pitch_link",       # 29
    "left_wrist_roll_link",         # 30
    "right_wrist_roll_link",        # 31
]

# ---------------------------------------------------------------------------
# IsaacLab <-> MuJoCo DOF index mappings (31 DOFs)
#
# MuJoCo DFS order from the simplified MJCF (hu_d04.xml):
#   0: left_hip_pitch   6: right_hip_pitch  12: waist_yaw      15: head_yaw
#   1: left_hip_roll    7: right_hip_roll   13: waist_roll      16: head_pitch
#   2: left_hip_yaw     8: right_hip_yaw    14: waist_pitch     17..23: left arm
#   3: left_knee        9: right_knee                           24..30: right arm
#   4: left_ankle_pitch 10: right_ankle_pitch
#   5: left_ankle_roll  11: right_ankle_roll
#
# IMPORTANT: These mappings MUST be verified empirically by running the
# verify_hu_d04_mappings.py script with both IsaacLab and MuJoCo loaded.
# ---------------------------------------------------------------------------
HU_D04_ISAACLAB_TO_MUJOCO_DOF = [
    0,   # IL  0 left_hip_pitch      -> MJ  0
    6,   # IL  1 right_hip_pitch     -> MJ  6
    12,  # IL  2 waist_yaw           -> MJ 12
    1,   # IL  3 left_hip_roll       -> MJ  1
    7,   # IL  4 right_hip_roll      -> MJ  7
    13,  # IL  5 waist_roll          -> MJ 13
    2,   # IL  6 left_hip_yaw        -> MJ  2
    8,   # IL  7 right_hip_yaw       -> MJ  8
    14,  # IL  8 waist_pitch         -> MJ 14
    3,   # IL  9 left_knee           -> MJ  3
    9,   # IL 10 right_knee          -> MJ  9
    17,  # IL 11 left_shoulder_pitch -> MJ 17
    24,  # IL 12 right_shoulder_pitch-> MJ 24
    15,  # IL 13 head_yaw            -> MJ 15
    4,   # IL 14 left_ankle_pitch    -> MJ  4
    10,  # IL 15 right_ankle_pitch   -> MJ 10
    18,  # IL 16 left_shoulder_roll  -> MJ 18
    25,  # IL 17 right_shoulder_roll -> MJ 25
    16,  # IL 18 head_pitch          -> MJ 16
    5,   # IL 19 left_ankle_roll     -> MJ  5
    11,  # IL 20 right_ankle_roll    -> MJ 11
    19,  # IL 21 left_shoulder_yaw   -> MJ 19
    26,  # IL 22 right_shoulder_yaw  -> MJ 26
    20,  # IL 23 left_elbow          -> MJ 20
    27,  # IL 24 right_elbow         -> MJ 27
    21,  # IL 25 left_wrist_yaw      -> MJ 21
    28,  # IL 26 right_wrist_yaw     -> MJ 28
    22,  # IL 27 left_wrist_pitch    -> MJ 22
    29,  # IL 28 right_wrist_pitch   -> MJ 29
    23,  # IL 29 left_wrist_roll     -> MJ 23
    30,  # IL 30 right_wrist_roll    -> MJ 30
]

HU_D04_MUJOCO_TO_ISAACLAB_DOF = [
    0,   # MJ  0 left_hip_pitch      -> IL  0
    3,   # MJ  1 left_hip_roll       -> IL  3
    6,   # MJ  2 left_hip_yaw        -> IL  6
    9,   # MJ  3 left_knee           -> IL  9
    14,  # MJ  4 left_ankle_pitch    -> IL 14
    19,  # MJ  5 left_ankle_roll     -> IL 19
    1,   # MJ  6 right_hip_pitch     -> IL  1
    4,   # MJ  7 right_hip_roll      -> IL  4
    7,   # MJ  8 right_hip_yaw       -> IL  7
    10,  # MJ  9 right_knee          -> IL 10
    15,  # MJ 10 right_ankle_pitch   -> IL 15
    20,  # MJ 11 right_ankle_roll    -> IL 20
    2,   # MJ 12 waist_yaw           -> IL  2
    5,   # MJ 13 waist_roll          -> IL  5
    8,   # MJ 14 waist_pitch         -> IL  8
    13,  # MJ 15 head_yaw            -> IL 13
    18,  # MJ 16 head_pitch          -> IL 18
    11,  # MJ 17 left_shoulder_pitch -> IL 11
    16,  # MJ 18 left_shoulder_roll  -> IL 16
    21,  # MJ 19 left_shoulder_yaw   -> IL 21
    23,  # MJ 20 left_elbow          -> IL 23
    25,  # MJ 21 left_wrist_yaw      -> IL 25
    27,  # MJ 22 left_wrist_pitch    -> IL 27
    29,  # MJ 23 left_wrist_roll     -> IL 29
    12,  # MJ 24 right_shoulder_pitch-> IL 12
    17,  # MJ 25 right_shoulder_roll -> IL 17
    22,  # MJ 26 right_shoulder_yaw  -> IL 22
    24,  # MJ 27 right_elbow         -> IL 24
    26,  # MJ 28 right_wrist_yaw     -> IL 26
    28,  # MJ 29 right_wrist_pitch   -> IL 28
    30,  # MJ 30 right_wrist_roll    -> IL 30
]

# Body index mappings (32 bodies including root at index 0)
HU_D04_ISAACLAB_TO_MUJOCO_BODY = [
    0,   # IL  0 base_link               -> MJ  0
    1,   # IL  1 left_hip_pitch_link     -> MJ  1
    7,   # IL  2 right_hip_pitch_link    -> MJ  7
    13,  # IL  3 waist_yaw_link          -> MJ 13
    2,   # IL  4 left_hip_roll_link      -> MJ  2
    8,   # IL  5 right_hip_roll_link     -> MJ  8
    14,  # IL  6 waist_roll_link         -> MJ 14
    3,   # IL  7 left_hip_yaw_link       -> MJ  3
    9,   # IL  8 right_hip_yaw_link      -> MJ  9
    15,  # IL  9 waist_pitch_link        -> MJ 15
    4,   # IL 10 left_knee_link          -> MJ  4
    10,  # IL 11 right_knee_link         -> MJ 10
    18,  # IL 12 left_shoulder_pitch_link -> MJ 18
    25,  # IL 13 right_shoulder_pitch_link-> MJ 25
    16,  # IL 14 head_yaw_link           -> MJ 16
    5,   # IL 15 left_ankle_pitch_link   -> MJ  5
    11,  # IL 16 right_ankle_pitch_link  -> MJ 11
    19,  # IL 17 left_shoulder_roll_link -> MJ 19
    26,  # IL 18 right_shoulder_roll_link-> MJ 26
    17,  # IL 19 head_pitch_link         -> MJ 17
    6,   # IL 20 left_ankle_roll_link    -> MJ  6
    12,  # IL 21 right_ankle_roll_link   -> MJ 12
    20,  # IL 22 left_shoulder_yaw_link  -> MJ 20
    27,  # IL 23 right_shoulder_yaw_link -> MJ 27
    21,  # IL 24 left_elbow_link         -> MJ 21
    28,  # IL 25 right_elbow_link        -> MJ 28
    22,  # IL 26 left_wrist_yaw_link     -> MJ 22
    29,  # IL 27 right_wrist_yaw_link    -> MJ 29
    23,  # IL 28 left_wrist_pitch_link   -> MJ 23
    30,  # IL 29 right_wrist_pitch_link  -> MJ 30
    24,  # IL 30 left_wrist_roll_link    -> MJ 24
    31,  # IL 31 right_wrist_roll_link   -> MJ 31
]

HU_D04_MUJOCO_TO_ISAACLAB_BODY = [
    0,   # MJ  0 base_link               -> IL  0
    1,   # MJ  1 left_hip_pitch_link     -> IL  1
    4,   # MJ  2 left_hip_roll_link      -> IL  4
    7,   # MJ  3 left_hip_yaw_link       -> IL  7
    10,  # MJ  4 left_knee_link          -> IL 10
    15,  # MJ  5 left_ankle_pitch_link   -> IL 15
    20,  # MJ  6 left_ankle_roll_link    -> IL 20
    2,   # MJ  7 right_hip_pitch_link    -> IL  2
    5,   # MJ  8 right_hip_roll_link     -> IL  5
    8,   # MJ  9 right_hip_yaw_link      -> IL  8
    11,  # MJ 10 right_knee_link         -> IL 11
    16,  # MJ 11 right_ankle_pitch_link  -> IL 16
    21,  # MJ 12 right_ankle_roll_link   -> IL 21
    3,   # MJ 13 waist_yaw_link          -> IL  3
    6,   # MJ 14 waist_roll_link         -> IL  6
    9,   # MJ 15 waist_pitch_link        -> IL  9
    14,  # MJ 16 head_yaw_link           -> IL 14
    19,  # MJ 17 head_pitch_link         -> IL 19
    12,  # MJ 18 left_shoulder_pitch_link -> IL 12
    17,  # MJ 19 left_shoulder_roll_link -> IL 17
    22,  # MJ 20 left_shoulder_yaw_link  -> IL 22
    24,  # MJ 21 left_elbow_link         -> IL 24
    26,  # MJ 22 left_wrist_yaw_link     -> IL 26
    28,  # MJ 23 left_wrist_pitch_link   -> IL 28
    30,  # MJ 24 left_wrist_roll_link    -> IL 30
    13,  # MJ 25 right_shoulder_pitch_link-> IL 13
    18,  # MJ 26 right_shoulder_roll_link-> IL 18
    23,  # MJ 27 right_shoulder_yaw_link -> IL 23
    25,  # MJ 28 right_elbow_link        -> IL 25
    27,  # MJ 29 right_wrist_yaw_link    -> IL 27
    29,  # MJ 30 right_wrist_pitch_link  -> IL 29
    31,  # MJ 31 right_wrist_roll_link   -> IL 31
]

HU_D04_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": HU_D04_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": HU_D04_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": HU_D04_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": HU_D04_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": HU_D04_MUJOCO_TO_ISAACLAB_BODY,
}

# ---------------------------------------------------------------------------
# Articulation configuration
# ---------------------------------------------------------------------------
HU_D04_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/hu_d04/hu_d04.urdf",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # HU_D04-tuned crouched ready-stance: legs bent (G1-like), arms bent FORWARD.
        # Shoulder_pitch and elbow signs are FLIPPED from G1 because HU_D04's joint axes
        # are opposite (+pitch on HU_D04 rotates arm backward, not forward like G1).
        pos=(0.0, 0.0, 0.87),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": -0.6,              # flipped from G1's +0.6
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": -0.2,   # flipped from G1's +0.2
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": -0.2,  # flipped from G1's +0.2
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
            effort_limit_sim=140.0,
            velocity_limit_sim=5.0,
            stiffness=STIFFNESS_HIP_KNEE,
            damping=DAMPING_HIP_KNEE,
            armature=ARMATURE_HIP_KNEE,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=42.0,  # hardware spec from MJCF ctrlrange
            velocity_limit_sim=13.6,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=STIFFNESS_ANKLE_WAIST,
            damping=DAMPING_ANKLE_WAIST,
            armature=ARMATURE_ANKLE_WAIST,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=42.0,  # hardware spec from MJCF ctrlrange
            velocity_limit_sim=13.6,
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            stiffness=STIFFNESS_ANKLE_WAIST,
            damping=DAMPING_ANKLE_WAIST,
            armature=ARMATURE_ANKLE_WAIST,
        ),
        "head": ImplicitActuatorCfg(
            effort_limit_sim=19.0,
            velocity_limit_sim=13.0,
            joint_names_expr=["head_pitch_joint", "head_yaw_joint"],
            stiffness=STIFFNESS_WRIST_HEAD,
            damping=DAMPING_WRIST_HEAD,
            armature=ARMATURE_WRIST_HEAD,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=42.0,
            velocity_limit_sim=19.6,
            stiffness=STIFFNESS_SHOULDER_ELBOW,
            damping=DAMPING_SHOULDER_ELBOW,
            armature=ARMATURE_SHOULDER_ELBOW,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=19.0,
            velocity_limit_sim=13.0,
            stiffness=STIFFNESS_WRIST_HEAD,
            damping=DAMPING_WRIST_HEAD,
            armature=ARMATURE_WRIST_HEAD,
        ),
    },
)

# ---------------------------------------------------------------------------
# Action scale: 0.25 * effort_limit / stiffness (same formula as H2/G1)
# ---------------------------------------------------------------------------
HU_D04_ACTION_SCALE = {}
for a in HU_D04_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            HU_D04_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
