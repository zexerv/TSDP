import numpy as np

# ==============================================================================
# FILE PATHS & LOADING
# Keywords: data, input, csv, loading
# ==============================================================================
# Absolute path to the parent directory containing subfolders like 'button', 'lever', etc.
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/reorganized_data/linear_switch'
# Load all CSV files found in PARENT_FOLDER_PATH if True, otherwise load from FILE_LIST.
LOAD_ALL_FILES = True
# Explicit list of filenames (relative to PARENT_FOLDER_PATH) to load if LOAD_ALL_FILES is False.
FILE_LIST = []

# ==============================================================================
# ALIGNMENT (DTW)
# Keywords: dtw, alignment, warping, distance, penalty, events
# ==============================================================================
# Method for aligning trajectories ('DTW', 'DDTW', 'Combined').
ALIGNMENT_TYPE = 'DTW'
# Base features used for calculating the distance in DTW.
ALIGNMENT_BASE_FEATURE_COLS = ['tx', 'ty', 'tz']
# Normalize position and derivative features before combining (only for 'Combined' type).
NORMALIZE_FOR_COMBINED = True
# Weight multiplier for rotation features relative to position features in DTW distance.
ALIGNMENT_ROTATION_WEIGHT = 0.1
# Weight multiplier for derivative features relative to position features (only for 'Combined' type).
ALIGNMENT_DERIVATIVE_WEIGHT = 1.0

# --- Alignment Method Selection ---
# If True, use fastdtw library (approximate, faster, no event penalties).
# If False, use manual DTW implementation (exact, slower, allows event penalties).
USE_FASTDTW_APPROXIMATION = True

# --- Event Penalties (Only used if USE_FASTDTW_APPROXIMATION = False) ---
# Penalty added to DTW cost for aligning a state_change event with a non-state_change point.
ALIGNMENT_STATE_CHANGE_PENALTY = 1.0
# Penalty for mismatch of the first waypoint *before* state_change.
ALIGNMENT_PRE_WP1_PENALTY = 0.8
# Penalty for mismatch of the first waypoint *at or after* state_change.
ALIGNMENT_POST_WP1_PENALTY = 0.8
# Penalty for mismatch of the second waypoint *before* state_change (conditional).
ALIGNMENT_PRE_WP2_PENALTY = 0.5
# Penalty for mismatch of the second waypoint *at or after* state_change (conditional).
ALIGNMENT_POST_WP2_PENALTY = 0.5
# Penalty for mismatch of the first gripper change *before* state_change.
ALIGNMENT_PRE_GRIPPER1_PENALTY = 0.7
# Penalty for mismatch of the first gripper change *at or after* state_change.
ALIGNMENT_POST_GRIPPER1_PENALTY = 0.7

# --- Conditional Penalty Activation (Only used if USE_FASTDTW_APPROXIMATION = False) ---
# Minimum fraction (0.0-1.0) of trajectories needing the N-th (N>1) waypoint for its penalty to apply globally.
WAYPOINT_PRESENCE_THRESHOLD = 0.8

# ==============================================================================
# SEGMENTATION
# Keywords: segmentation, dynamic programming, cost, segments, tube, events
# ==============================================================================
# Features used to define the tube boundaries for segmentation cost calculation.
SEGMENTATION_FEATURE_COLS = ['tx', 'ty', 'tz', 'r11','r12','r13','r21','r22','r23','r31','r32','r33']
# Maximum number of segments to consider in the dynamic programming optimization.
MAX_SEGMENTS = 10
# Penalty added for each additional segment in the DP cost function (controls granularity).
LAMBDA_PENALTY = 0.05
# Weight multiplier for rotation features relative to position features in the segmentation geometric cost.
ROTATION_WEIGHT = 0.1
# Penalty added to a segment's cost if it contains both the representative state_change
# event and its immediate representative neighbor event (wp or gripper). Discourages co-occurrence.
SEGMENTATION_EVENT_COOCCURRENCE_PENALTY = 10.0
# If True, also penalize segments containing *both* the pre-neighbor and post-neighbor event.
PENALIZE_NEIGHBOR_COOCCURRENCE = False # Set to False to disable this specific penalty
# Penalty value used if PENALIZE_NEIGHBOR_COOCCURRENCE is True.
SEGMENTATION_NEIGHBOR_COOCCURRENCE_PENALTY = 10.0 # Example value, tune as needed

# ==============================================================================
# DOWNSAMPLING
# Keywords: downsampling, performance, DP, length
# ==============================================================================
# Trajectories longer than this will be downsampled before segmentation DP calculation.
MAX_DP_LENGTH = 300

# ==============================================================================
# EVENT DETECTION & FEATURE DEFINITIONS
# Keywords: events, state change, waypoint, gripper, features, columns
# ==============================================================================
# Columns used to detect specific non-state_change events directly.
EVENT_COLUMNS = {
    'wp_saved': 'WP_Saved',
    'gripper': 'GripperState',
}
# Features considered 'position' for weighting purposes.
POS_COLS = ['tx', 'ty', 'tz']
# Threshold for detecting significant lever movement in find_events.
LEVER_CHANGE_THRESHOLD = 0.001
# List of ALL potential TCA columns that might appear in ANY CSV file.
# Ensures they are loaded if needed by find_events based on interface_id.
ALL_TCA_COLS_TO_LOAD = [
    'TCA_button_288_state','TCA_button_289_state','TCA_button_292_state',
    'TCA_button_293_state','TCA_button_704_state','TCA_button_705_state',
    'TCA_button_706_state','TCA_switch_290_state','TCA_switch_291_state',
    'TCA_switch_294_state','TCA_switch_295_state','TCA_switch_296_state',
    'TCA_switch_297_state','TCA_switch_298_state','TCA_switch_299_state',
    'TCA_switch_300_state','TCA_switch_301_state','TCA_switch_302_state',
    'TCA_switch_303_state','TCA_switch_707_state','TCA_switch_708_state',
    'TCA_switch_709_state','TCA_switch_710_state','TCA_switch_711_state',
    'TCA_switch_712_state','TCA_switch_713_state','TCA_switch_714_state',
    'TCA_switch_715_state','TCA_switch_716_state','TCA_switch_717_state',
    'TCA_switch_718_state','TCA_lever_0_value','TCA_lever_1_value',
    'TCA_lever_2_value','TCA_lever_5_value'
]

# ==============================================================================
# VISUALIZATION
# Keywords: plot, visualization, output, figures
# ==============================================================================
# Number of columns in the multi-plot grid for segmentation results.
PLOT_GRID_COLUMNS = 3
# Font family for matplotlib plots.
PLOT_FONT = 'serif'

# ==============================================================================
# DERIVED CONFIGURATION (Internal - Do Not Modify Manually)
# ==============================================================================
# --- Columns to Load ---
# Combines features needed for alignment, segmentation, basic events, interface_id, and all TCA columns.
_alignment_features = set(ALIGNMENT_BASE_FEATURE_COLS)
_segmentation_features = set(SEGMENTATION_FEATURE_COLS)
_basic_event_cols = set(col for col in EVENT_COLUMNS.values() if isinstance(col, str))
_tca_cols = set(ALL_TCA_COLS_TO_LOAD)
_required_misc_cols = set(['interface_id'])

ALL_LOAD_COLS = list(
    _alignment_features |
    _segmentation_features |
    _basic_event_cols |
    _tca_cols |
    _required_misc_cols
)
# ==============================================================================
# OUTPUT FILES
# Keywords: output, results, analysis, statistics
# ==============================================================================
# Absolute path for saving the calculated segment statistics file (e.g., Yaml, JSON format).
SEGMENT_STATS_OUTPUT_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/results/segment_stats.yaml' # CHANGE AS NEEDED

# Add a print statement for it too if you like:
print(f"SEGMENT_STATS_OUTPUT_PATH: {SEGMENT_STATS_OUTPUT_PATH}")

# --- Print Loaded Configuration ---
print("--- Configuration Loaded ---")
print(f"PARENT_FOLDER_PATH: {PARENT_FOLDER_PATH}")
print(f"LOAD_ALL_FILES: {LOAD_ALL_FILES}")
print(f"ALIGNMENT_TYPE: {ALIGNMENT_TYPE}")
print(f"ALIGNMENT_BASE_FEATURE_COLS: {ALIGNMENT_BASE_FEATURE_COLS}")
print(f"USE_FASTDTW_APPROXIMATION: {USE_FASTDTW_APPROXIMATION}")
if not USE_FASTDTW_APPROXIMATION:
    print(f"  ALIGNMENT_STATE_CHANGE_PENALTY: {ALIGNMENT_STATE_CHANGE_PENALTY}")
    print(f"  ALIGNMENT_PRE_WP1_PENALTY: {ALIGNMENT_PRE_WP1_PENALTY}")
    print(f"  ALIGNMENT_POST_WP1_PENALTY: {ALIGNMENT_POST_WP1_PENALTY}")
    print(f"  ALIGNMENT_PRE_WP2_PENALTY: {ALIGNMENT_PRE_WP2_PENALTY}")
    print(f"  ALIGNMENT_POST_WP2_PENALTY: {ALIGNMENT_POST_WP2_PENALTY}")
    print(f"  ALIGNMENT_PRE_GRIPPER1_PENALTY: {ALIGNMENT_PRE_GRIPPER1_PENALTY}")
    print(f"  ALIGNMENT_POST_GRIPPER1_PENALTY: {ALIGNMENT_POST_GRIPPER1_PENALTY}")
    print(f"  WAYPOINT_PRESENCE_THRESHOLD: {WAYPOINT_PRESENCE_THRESHOLD}")
print(f"SEGMENTATION_FEATURE_COLS: {SEGMENTATION_FEATURE_COLS}")
print(f"MAX_SEGMENTS: {MAX_SEGMENTS}")
print(f"LAMBDA_PENALTY: {LAMBDA_PENALTY}")
print(f"SEGMENTATION_EVENT_COOCCURRENCE_PENALTY: {SEGMENTATION_EVENT_COOCCURRENCE_PENALTY}")
print(f"MAX_DP_LENGTH: {MAX_DP_LENGTH}")
print(f"Columns To Load: {sorted(ALL_LOAD_COLS)}")
print("-" * 25)

