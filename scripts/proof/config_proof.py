# proof/config_proof.py
"""
Configuration file for the Multi-Dimensional trajectory analysis proof-of-concept.
"""
import os

# ==============================================================================
# FILE PATHS & LOADING
# ==============================================================================
_current_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure this path points to the directory containing your CSV files (e.g., linear_switch)
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/reorganized_data/button' # User's path

LOAD_ALL_FILES = True
FILE_LIST = [] # Not used if LOAD_ALL_FILES is True

# ==============================================================================
# FEATURE SELECTION (MULTI-DIMENSIONAL)
# ==============================================================================
# List of ALL dimension names (strings) to be extracted from the input CSVs.
# Example: ['tx', 'ty', 'tz', 'r00', 'r01', 'r02', 'r10', 'r11', 'r12', 'r20', 'r21', 'r22']
# For initial testing, let's use a few. Ensure these columns exist in your CSVs.
DIMENSIONS_TO_USE = ['tx', 'ty', 'tz', 'r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33'] # Example: 3 pos, 3 rot (diag of rot matrix)

# Define which of the DIMENSIONS_TO_USE are considered 'position'
POSITION_DIMENSIONS = ['tx', 'ty', 'tz']

# Define which of the DIMENSIONS_TO_USE are considered 'rotation'
# Ensure these are also in DIMENSIONS_TO_USE and don't overlap with POSITION_DIMENSIONS
ROTATION_DIMENSIONS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33'] # Example

# Weights for cost function
WEIGHT_POSITION = 1.0  # Weight for position dimensions
WEIGHT_ROTATION = 0.5  # Weight for rotation dimensions

# Sanity check for dimension lists (optional, but good practice)
# _s_dims_to_use = set(DIMENSIONS_TO_USE)
# _s_pos_dims = set(POSITION_DIMENSIONS)
# _s_rot_dims = set(ROTATION_DIMENSIONS)
# if not _s_pos_dims.issubset(_s_dims_to_use) or not _s_rot_dims.issubset(_s_dims_to_use):
#     raise ValueError("POSITION_DIMENSIONS and ROTATION_DIMENSIONS must be subsets of DIMENSIONS_TO_USE.")
# if not _s_pos_dims.isdisjoint(_s_rot_dims):
#     raise ValueError("POSITION_DIMENSIONS and ROTATION_DIMENSIONS must be disjoint.")
# if (_s_pos_dims | _s_rot_dims) != _s_dims_to_use:
#     raise ValueError("All DIMENSIONS_TO_USE must be categorized as either POSITION or ROTATION.")


# ==============================================================================
# ALIGNMENT (DTW)
# ==============================================================================
# For multi-dimensional DTW, fastdtw can use Euclidean distance between D-dim vectors.
USE_FASTDTW_APPROXIMATION = True
FASTRADIUS = 1 

# ==============================================================================
# DATA PROCESSING FOR PELT (Downsampling)
# ==============================================================================
DEFAULT_RESOLUTION_FACTOR = 1 # 1 = original resolution. >1 for downsampling.

# ==============================================================================
# SEGMENTATION (PELT)
# ==============================================================================
# Lambda penalty for PELT. May need tuning for multi-dimensional costs.
DEFAULT_LAMBDA_PELT = 0.01 # Adjusted from 0.2, as cost magnitude might increase

# ==============================================================================
# VISUALIZATION & EXPERIMENT SETTINGS
# ==============================================================================
PLOT_FONT = 'serif'
# Store results in a new directory for these multi-dim experiments
RESULTS_DIR_PROOF = os.path.join(os.path.dirname(__file__), "results_multidim_pelt_experiments") 
LANDSCAPE_RESOLUTION_FACTOR = 2 # Change this value in config to control landscape downsampling

# For experiments, the primary plot is the optimal PELT segmentation.
ENABLE_OPTIMAL_PELT_PLOT = True 
# Other plots are generally disabled to keep output clean for batch runs.
ENABLE_DETAILED_SEGMENT_PLOTS = False 
ENABLE_FULL_TUBE_PLOT = False       
ENABLE_PREFIX_SUM_PLOTS = False     
# For the cost landscape, it will have its own script and result directory.

# Dimensions to plot in the final PELT segmentation visualization
# Choose a subset of DIMENSIONS_TO_USE
DIMENSIONS_TO_PLOT_IN_PELT_VIS = ['tx', 'ty', 'tz', 'r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33'] # Example

# --- Print Loaded Configuration ---
print("--- Proof Config Loaded (Multi-Dimensional PELT Experiments) ---")
print(f"PARENT_FOLDER_PATH: {PARENT_FOLDER_PATH}")
print(f"DIMENSIONS_TO_USE: {DIMENSIONS_TO_USE} (D={len(DIMENSIONS_TO_USE)})")
print(f"POSITION_DIMENSIONS: {POSITION_DIMENSIONS}")
print(f"ROTATION_DIMENSIONS: {ROTATION_DIMENSIONS}")
print(f"WEIGHT_POSITION: {WEIGHT_POSITION}")
print(f"WEIGHT_ROTATION: {WEIGHT_ROTATION}")
print(f"Default RESOLUTION_FACTOR: {DEFAULT_RESOLUTION_FACTOR}")
print(f"Default LAMBDA_PELT: {DEFAULT_LAMBDA_PELT}")
print(f"RESULTS_DIR_PROOF: {RESULTS_DIR_PROOF}")
print(f"Plotting for experiments: Optimal PELT plot ENABLED for selected dims.")
print("-" * 25)
