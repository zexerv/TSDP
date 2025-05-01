# config.py
import numpy as np

# --- Robot Configuration ---
NUM_JOINTS = 6
JOINT_LIMITS_MIN = np.deg2rad([-360] * NUM_JOINTS)
JOINT_LIMITS_MAX = np.deg2rad([ 360] * NUM_JOINTS)

# --- Planner Configuration ---
OMPL_AVAILABLE = False # Will be set to True if import succeeds in planner.py

# --- OMPL Cost/Objective Function Weights ---
WEIGHT_PATH_LENGTH = 1.0
WEIGHT_MANIPULABILITY = 10.0 # Set > 0 to enable, Tune
# WEIGHT_JLIM_OBJECTIVE = 0.0 # Example

# --- Manipulability Objective Parameters ---
MANIPULABILITY_EPSILON = 1e-6
MAX_MANIPULABILITY_COST = 1e8

# --- IK Solution Selection Criteria Weights ---
IK_SELECT_WEIGHT_MANIP_COST = 10.0 # Tune
IK_SELECT_WEIGHT_JLIM_COST  = 5.0  # Tune
IK_SELECT_WEIGHT_DIST_COST  = 1.0  # Tune
IK_SELECT_JLIM_EPSILON      = 1e-4

# --- Path Processing ---
INTERPOLATE_PATH_POINTS = 50

# --- Visualization Configuration ---
VIS_3D_SNAPSHOTS = 10 # Reduced snapshots per segment maybe
VIS_PLOT_LIMITS = ([-1.0, 1.0], [-1.0, 1.0], [-0.2, 1.5])
VIS_BASE_FRAME_SIZE = 0.15
VIS_TCP_FRAME_SIZE = 0.1

# --- Optimization Algorithm Selection ---
# Selects algorithm used for refinement (GD or PSO)
OPTIMIZATION_ALGORITHM = "PSO"  # Options: "GD", "PSO"

# --- Planning Time Limits ---
# Default time for final plan evaluation of a refined pose
FINAL_EVAL_PLANNING_TIME_LIMIT = 0.05
# Time limits for evaluations *during* optimization
GD_EVAL_PLANNING_TIME_LIMIT = 0.01
PSO_EVAL_PLANNING_TIME_LIMIT = 0.01

# --- Gradient Descent Configuration ---
ENABLE_GD_REFINEMENT = True # Used if OPTIMIZATION_ALGORITHM="GD"
GD_ITERATIONS = 50 # Reduced iterations maybe for multi-segment
GD_TOLERANCE = 1e-5
GD_EPSILON = 0.005
GD_STEP_SIZES_POS = np.array([0.001, 0.001, 0.001])
GD_STEP_SIZES_ROT = np.array([0.005, 0.005, 0.005, 0.005])
GD_ENABLE_CLIPPING = True
GD_MAX_GRAD_NORM = 1e6

# --- PSO Configuration ---
PSO_N_PARTICLES = 25 # Reduced particles maybe
PSO_ITERATIONS = 30  # Reduced iterations maybe
PSO_OPTIONS = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# --- Multi-Segment Configuration ---
NUM_PLANNING_SEGMENTS = 3 # Number of segments to plan (q0->S1*, q1*->S2*, ...)

# List defining parameters for each target set
# Ensure length >= NUM_PLANNING_SEGMENTS
TARGET_SET_DEFINITIONS = [
    # Set 1 (S1) Parameters
    {
        'name': 'Set1',
        'color': 'blue', # For visualization
        'base_q_deg': [60, -90, 110, -100, -90, 0], # Base config for center generation
        'perturb_min_deg': -10,
        'perturb_max_deg': 10,
        'pos_interval_m': [0.05] * 3, # [dx, dy, dz] relative to center
        'rot_interval_rad': 0.1,     # Approx rot tolerance
        'num_samples': 50,           # Number of samples for this segment
        'segment_planning_time_limit': 0.03 # Planning time limit for sampling evals in this segment
    },
    # Set 2 (S2) Parameters
    {
        'name': 'Set2',
        'color': 'green',
        'base_q_deg': [90, -90, 90, -90, -90, 0], # Different base config
        'perturb_min_deg': -15,
        'perturb_max_deg': 15,
        'pos_interval_m': [0.08] * 3,
        'rot_interval_rad': 0.15,
        'num_samples': 50, # Can vary per segment
        'segment_planning_time_limit': 0.03
    },
    # Set 3 (S3) Parameters
    {
        'name': 'Set3',
        'color': 'red',
        'base_q_deg': [0, -45, 135, -90, 90, 0],
        'perturb_min_deg': -5,
        'perturb_max_deg': 5,
        'pos_interval_m': [0.03] * 3,
        'rot_interval_rad': 0.05,
        'num_samples': 50,
        'segment_planning_time_limit': 0.03
    },
    # Add definitions for S4, S5... if NUM_PLANNING_SEGMENTS > 3
]

# --- Final Print Statement ---
# print(f"config.py loaded (Algorithm: {config.OPTIMIZATION_ALGORITHM}, Segments: {config.NUM_PLANNING_SEGMENTS})")

# # config.py
# import numpy as np

# # --- Robot Configuration ---
# NUM_JOINTS = 6
# # Joint limits (radians) - Assuming standard UR limits, adjust if needed
# JOINT_LIMITS_MIN = np.deg2rad([-360] * NUM_JOINTS)
# JOINT_LIMITS_MAX = np.deg2rad([ 360] * NUM_JOINTS)

# # --- Planner Configuration ---
# OMPL_AVAILABLE = False # Will be set to True if import succeeds in planner.py

# # --- NEW: OMPL Cost/Objective Function Weights ---
# WEIGHT_PATH_LENGTH = 1.0      # Weight for standard path length
# WEIGHT_MANIPULABILITY = 1.0   # Weight for manipulability (SET > 0 TO ENABLE) - TUNABLE
# # WEIGHT_SMOOTHNESS = 0.0     # Placeholder for future objectives

# # --- NEW: Manipulability Objective Parameters ---
# MANIPULABILITY_EPSILON = 1e-6 # Small value added to manipulability before division
# MAX_MANIPULABILITY_COST = 1e8 # Maximum cost assigned if manipulability is near zero or Jacobian fails

# # --- Planning Time Limits ---
# PLANNING_TIME_LIMIT = 0.05 # Default planning time limit (seconds) - For final plan eval
# GD_EVAL_PLANNING_TIME_LIMIT = 0.01 # Time limit for GD finite difference evaluations (seconds)
# PSO_EVAL_PLANNING_TIME_LIMIT = 0.01 # Time limit for *each* PSO fitness evaluation (Keep short!)

# # --- Path Processing ---
# INTERPOLATE_PATH_POINTS = 50 # Number of points for final path density

# # --- Visualization Configuration ---
# VIS_3D_SNAPSHOTS = 15 # Number of snapshots to show along the 3D path
# VIS_PLOT_LIMITS = ([-1.0, 1.0], [-1.0, 1.0], [-0.2, 1.5]) # Axes limits for 3D plot [X, Y, Z]
# VIS_BASE_FRAME_SIZE = 0.15
# VIS_TCP_FRAME_SIZE = 0.1

# # --- Set Definition ---
# # Perturbation range for generating set centers
# PERTURB_MIN_DEG = -10
# PERTURB_MAX_DEG = 10
# # Default intervals for defining task space sets around a center pose
# DEFAULT_POS_INTERVAL = [0.3] * 3 # meters [dx, dy, dz]
# DEFAULT_ROT_INTERVAL = 0.1       # radians (approx tolerance around center orientation - simplistic)

# # --- Optimization Algorithm Selection ---
# OPTIMIZATION_ALGORITHM = "PSO"  # Options: "GD", "PSO"

# # --- Sampling Configuration ---
# NUM_SAMPLES_S1 = 1 # Number of random target candidates in Set 1

# # --- Gradient Descent Configuration (Parameters used if OPTIMIZATION_ALGORITHM="GD") ---
# ENABLE_GD_REFINEMENT = True # Only relevant if OPTIMIZATION_ALGORITHM="GD"
# GD_ITERATIONS = 300
# GD_TOLERANCE = 1e-5
# GD_EPSILON = 0.005 # Perturbation for finite differencing
# # Dimension-Specific Step Sizes
# c = 2
# GD_STEP_SIZES_POS = c*np.array([0.001, 0.001, 0.001]) # Tunable [x, y, z]
# GD_STEP_SIZES_ROT = c*np.array([0.005, 0.005, 0.005, 0.005]) # Tunable [w, qx, qy, qz]
# # Gradient Clipping
# GD_ENABLE_CLIPPING = True
# GD_MAX_GRAD_NORM = 1e6 # Adjusted max norm for raw gradient - Tunable

# # --- PSO Configuration (Parameters used if OPTIMIZATION_ALGORITHM="PSO") ---
# PSO_N_PARTICLES = 100     # Number of particles in the swarm (tune)
# PSO_ITERATIONS = 4      # Number of iterations (tune)
# # PySwarms options dictionary for GlobalBestPSO (standard defaults, tune these)
# PSO_OPTIONS = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# # --- IK Solution Selection Criteria Weights ---
# # Weights used in utils.initialize_robot_taskspace to select the best q_goal
# IK_SELECT_WEIGHT_MANIP_COST = 10.0  # Weight for 1/(manipulability + eps). Higher val -> more emphasis on high manip.
# IK_SELECT_WEIGHT_JLIM_COST  = 5.0   # Weight for joint limit proximity cost. Higher val -> more emphasis on avoiding limits.
# IK_SELECT_WEIGHT_DIST_COST  = 1.0   # Weight for C-space distance from hint/ref. Higher val -> more emphasis on staying close.
# # Epsilons for calculations (prevent division by zero)
# IK_SELECT_JLIM_EPSILON      = 1e-4  # For joint limit cost denominator
# # --- Final Print Statement ---
# print(f"config.py loaded (Algorithm: {OPTIMIZATION_ALGORITHM})")