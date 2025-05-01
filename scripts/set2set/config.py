# config.py
import numpy as np

# --- Robot Configuration ---
NUM_JOINTS = 6
# Joint limits (radians)
JOINT_LIMITS_MIN = np.deg2rad([-360] * NUM_JOINTS)
JOINT_LIMITS_MAX = np.deg2rad([ 360] * NUM_JOINTS)

# --- Planner Configuration ---
PLANNING_TIME_LIMIT = 0.03 # Default planning time limit (seconds) - CHANGED
GD_EVAL_PLANNING_TIME_LIMIT = 0.01 # Time limit for GD evaluations (seconds) - ADJUSTED
INTERPOLATE_PATH_POINTS = 50 # Number of points to interpolate path for execution/visualization density
OMPL_AVAILABLE = False # Will be set to True if import succeeds in planner.py

# --- Visualization Configuration ---
VIS_3D_SNAPSHOTS = 15 # Number of snapshots to show along the 3D path
VIS_PLOT_LIMITS = ([-1.0, 1.0], [-1.0, 1.0], [-0.2, 1.5]) # Axes limits for 3D plot [X, Y, Z]
VIS_BASE_FRAME_SIZE = 0.15
VIS_TCP_FRAME_SIZE = 0.1

# --- Set Definition ---
# Perturbation range for generating set centers
PERTURB_MIN_DEG = -10
PERTURB_MAX_DEG = 10
# Default intervals for defining task space sets around a center pose
DEFAULT_POS_INTERVAL = [0.05] * 3 # meters [dx, dy, dz]
DEFAULT_ROT_INTERVAL = 0.1       # radians (approx tolerance around center orientation - simplistic)

# --- OMPL Cost/Objective Function Weights (Placeholders) ---
# TODO: Implement and tune these weights
WEIGHT_PATH_LENGTH = 1.0
WEIGHT_MANIPULABILITY = 0.0 # Set > 0 to enable
WEIGHT_SMOOTHNESS = 0.0     # Set > 0 to enable

# --- Optimization Configuration ---
# Random Sampling
NUM_SAMPLES_S1 = 1 # Number of random target candidates in Set 1 (Increased slightly)

# Gradient Descent Refinement
ENABLE_GD_REFINEMENT = True # Set to False to skip GD step
GD_ITERATIONS = 200          # Max number of GD steps
GD_STEP_SIZE = 0.01        # Learning rate for position update (Keep small for now)
GD_TOLERANCE = 1e-5         # Stop GD if cost improvement is less than this
GD_EPSILON = 0.005         # Perturbation distance for numerical gradient

# --- NEW: Gradient Clipping ---
GD_ENABLE_CLIPPING = True   # Enable/disable gradient clipping
GD_MAX_GRAD_NORM = 50000000000000.0   # Maximum allowed norm for the gradient (prevent huge steps) - ADJUST AS NEEDED

# --- TODO: Adaptive Step Size Parameters ---
# GD_ENABLE_ADAPTIVE_STEP = False
# GD_ADAPTIVE_FACTOR_INCREASE = 1.1 # Factor to increase step size on success
# GD_ADAPTIVE_FACTOR_DECREASE = 0.5 # Factor to decrease step size on failure
# GD_MIN_STEP_SIZE = 1e-5
# GD_MAX_STEP_SIZE = 0.01


print("config.py loaded (GD Clipping, Time Limits)")

