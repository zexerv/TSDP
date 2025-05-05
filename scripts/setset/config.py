# config.py
import numpy as np
import math
# Keep yaml/Path imports if utils doesn't handle everything needed downstream
# import yaml
# from pathlib import Path

# --- Configuration File Paths ---
# *** Path for Probabilistic Approach (Boundary/Cross-section GMM stats) ***
PROBABILISTIC_STATS_YAML_PATH = "/home/kadi/Desktop/Thesis/demo_processor_new/results/cross_section_stats.yaml" # <<< CONFIRM THIS PATH is correct
# Other paths
INTERFACE_TRANSFORMS_YAML_PATH = "/home/kadi/Desktop/Thesis/demo_processor_new/scripts/setset/interface_transforms.yaml"
ENVIRONMENT_YAML_PATH = "/home/kadi/Desktop/Thesis/demo_processor_new/scripts/setset/environment_config.yaml"

# --- Interface ID ---
INTERFACE_ID = "button_292" # <<< SET YOUR DESIRED INTERFACE ID HERE

# --- Configuration for Probabilistic Approach ---

# *** NEW: Enable GMM Waypoint Optimization ***
# If True, runs optimization (PSO/GD) within the GMM distribution at each boundary.
# If False, uses the simpler PROB_WAYPOINT_METHOD below.
ENABLE_GMM_OPTIMIZATION = True # <<< SET TO True TO ENABLE OPTIMIZATION

# Method to extract INITIAL target pose from boundary GMMs (used if optimization is OFF, or as starting guess for optimization)
# Options: 'mean_of_best_component', 'sample', 'overall_mean'
PROB_WAYPOINT_METHOD = 'mean_of_best_component'

# --- GMM Waypoint Optimization Parameters (if ENABLE_GMM_OPTIMIZATION=True) ---
GMM_OPT_ALGORITHM = "PSO" # Options: "PSO", "GD"
# Cost function weights
GMM_OPT_WEIGHT_PLANNING = 1.0 # Weight for OMPL planning cost
GMM_OPT_WEIGHT_GMM_DIST = 5.0 # Weight for Mahalanobis distance from GMM components (Higher = stay closer to demo distribution)
# Planning time limit used *inside* the GMM optimization cost evaluation
GMM_OPT_EVAL_PLANNING_TIME = 0.01 # Keep this short!

# PSO Parameters for GMM Optimization
GMM_PSO_N_PARTICLES = 10 # Number of particles exploring the 6D GMM state space
GMM_PSO_ITERATIONS = 5  # Number of iterations
GMM_PSO_OPTIONS = {'c1': 0.5, 'c2': 0.3, 'w': 0.9} # Standard PSO parameters (tune if needed)
# Optional: Define bounds for the 6D state space search based on GMM stats?
# GMM_PSO_BOUNDS_STD_DEV = 3.0 # E.g., search within N std devs of the overall GMM mean/covariance

# GD Parameters for GMM Optimization (If GMM_OPT_ALGORITHM="GD")
GMM_GD_ITERATIONS = 20
GMM_GD_LEARNING_RATE = 0.01 # Tune learning rate for 6D state updates
GMM_GD_EPSILON = 1e-4 # Finite difference step for gradient estimation (if needed)
GMM_GD_TOLERANCE = 1e-5 # Convergence tolerance

# --- Robot Configuration ---
NUM_JOINTS = 6
JOINT_LIMITS_MIN = np.deg2rad([-360] * NUM_JOINTS)
JOINT_LIMITS_MAX = np.deg2rad([ 360] * NUM_JOINTS)

# --- Planner Configuration ---
OMPL_AVAILABLE = False # Set automatically in planner.py
# Planning time limit used BETWEEN the final optimized waypoints (per segment).
INTER_WAYPOINT_PLANNING_TIME = 0.001 # <<< Keep this potentially longer for final path/refinement

# --- OMPL Cost/Objective Function Weights ---
WEIGHT_PATH_LENGTH = 10.0
WEIGHT_MANIPULABILITY = 1 # Keep disabled for now based on previous issues
MANIPULABILITY_EPSILON = 1e-30
MAX_MANIPULABILITY_COST = 1e8

# --- IK Solution Selection Criteria Weights ---
# Used by initialize_robot_taskspace (e.g., for initial state)
# And potentially by get_ik_solution_for_branch if multiple valid solutions exist for a branch tuple (unlikely for closed-form)
IK_SELECT_WEIGHT_MANIP_COST = 1 # Keep disabled
IK_SELECT_WEIGHT_JLIM_COST  = 5.0
IK_SELECT_WEIGHT_DIST_COST  = 1.0
IK_SELECT_JLIM_EPSILON      = 1e-4

# --- Path Processing ---
INTERPOLATE_PATH_POINTS = 50 # Number of points to interpolate final path (0 = disable)

# --- Visualization Configuration ---
VIS_3D_SNAPSHOTS = 10
VIS_PLOT_LIMITS = ([-1.0, 1.0], [-1.0, 1.0], [-0.2, 1.5])
VIS_BASE_FRAME_SIZE = 0.15
VIS_TCP_FRAME_SIZE = 0.1

# --- Print loaded configuration summary ---
print(f"--- config.py Loaded (Probabilistic Approach + GMM Optimization) ---")
print(f"Probabilistic Stats Path: {PROBABILISTIC_STATS_YAML_PATH}")
print(f"Enable GMM Waypoint Optimization: {ENABLE_GMM_OPTIMIZATION}")
if ENABLE_GMM_OPTIMIZATION:
    print(f"  GMM Opt Algorithm: {GMM_OPT_ALGORITHM}")
    print(f"  Cost Weights: Plan={GMM_OPT_WEIGHT_PLANNING}, GMM_Dist={GMM_OPT_WEIGHT_GMM_DIST}")
    if GMM_OPT_ALGORITHM == "PSO":
        print(f"  PSO Params: Particles={GMM_PSO_N_PARTICLES}, Iterations={GMM_PSO_ITERATIONS}")
    elif GMM_OPT_ALGORITHM == "GD":
        print(f"  GD Params: Iterations={GMM_GD_ITERATIONS}, LR={GMM_GD_LEARNING_RATE}")
else:
    print(f"  Initial Waypoint Method (No Opt): {PROB_WAYPOINT_METHOD}")
print(f"Inter-Waypoint Planning Time: {INTER_WAYPOINT_PLANNING_TIME}")
print(f"Interface ID: {INTERFACE_ID}")
print(f"-------------------------------------------------------------")

