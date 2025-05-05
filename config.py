# -*- coding: utf-8 -*-
"""
Configuration file for the segdp trajectory segmentation and analysis pipeline.
"""

import os
import numpy as np
import re # Import regex module

# ==============================================================================
# FILE PATHS & LOADING
# Keywords: data, input, csv, loading
# ==============================================================================
# Absolute path to the parent directory containing subfolders like 'button', 'lever', etc.
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/reorganized_data/linear_button'
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
USE_FASTDTW_APPROXIMATION = True # Set to False to use manual DTW with penalties

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
# Ensure this includes position and rotation matrix columns if needed for tube viz
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
    # 'state_change' is derived dynamically using interface_id and TCA columns
}
# Features considered 'position' for weighting purposes.
POS_COLS = ['tx', 'ty', 'tz']
# Columns representing the raw orientation (used for conversion and analysis)
# Assuming 3x3 Rotation Matrix format based on previous context
ROT_MAT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
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
# GMM CROSS-SECTION ANALYSIS
# Keywords: gmm, analysis, cross-section, boundary, event, statistics
# ==============================================================================
# State vector components for GMM training at cross-sections: [pos, orient_log_map]
GMM_POS_COLS = ['tx', 'ty', 'tz'] # Position columns
GMM_ORIENT_LOG_MAP_COLS = ['vx_log', 'vy_log', 'vz_log'] # Log map columns (will be added LATER)
GMM_STATE_COLS = GMM_POS_COLS + GMM_ORIENT_LOG_MAP_COLS # Combined state vector

# Number of components for the GMM trained at each cross-section.
# This acts as an upper limit; the actual number used will be min(GMM_N_COMPONENTS, n_snapshots)
GMM_N_COMPONENTS = 3 # Example value, tune based on data complexity and N trajectories

# Flag to enable analysis of event-based cross-sections (average time of events within a segment).
ANALYZE_EVENT_CROSS_SECTIONS = True # Set to False to only analyze DP boundaries

# Event types from mapped_events_list to consider for event-based cross-sections.
# Uses keys from the dictionary returned by find_events/map_events.
EVENTS_FOR_CROSS_SECTIONS = ['state_change', 'wp_saved', 'gripper_change']

# Absolute path for saving the cross-section GMM analysis results (YAML format recommended).
CROSS_SECTION_STATS_OUTPUT_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/results/cross_section_stats.yaml' # CHANGE AS NEEDED

# ==============================================================================
# BACKUP GLOBAL GMM ANALYSIS
# Keywords: gmm, backup, global, trajectory, time
# ==============================================================================
# Flag to enable the backup analysis: training a single GMM on all (z(t), t) data.
RUN_BACKUP_GLOBAL_GMM = True

# Number of components for the backup global GMM.
BACKUP_GMM_N_COMPONENTS = 5 # Example value

# Absolute path for saving the backup global GMM parameters (YAML format recommended).
BACKUP_GMM_OUTPUT_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/results/backup_global_gmm.yaml' # CHANGE AS NEEDED


# ==============================================================================
# VISUALIZATION
# Keywords: plot, visualization, output, figures
# ==============================================================================
# Number of columns in the multi-plot grid for segmentation results.
PLOT_GRID_COLUMNS = 3 # Adjust based on number of features plotted
# Font family for matplotlib plots.
PLOT_FONT = 'serif'
# Features to include in the main segmentation plot
# This list dictates which subplots are generated. Include derived columns like 'vx_log' here.
PLOT_FEATURE_COLS = ['tx', 'ty', 'tz', 'vx_log', 'vy_log', 'vz_log'] # Example: Pos + LogMap
# PLOT_FEATURE_COLS = ['tx', 'ty', 'tz', 'r11','r12','r13','r21','r22','r23','r31','r32','r33'] # Example: Pos + RotMat
# PLOT_FEATURE_COLS = ['tx', 'ty', 'tz'] # Example: Position only

# Flag to enable the additional plot showing geodesic distance from mean orientation.
PLOT_ORIENTATION_DEVIATION = True # Requires SciPy

# ==============================================================================
# DERIVED CONFIGURATION (Internal - Do Not Modify Manually)
# ==============================================================================
# --- Columns to Load ---
# Defines columns to be loaded from the initial CSV files.
# Combines features needed for alignment, segmentation (if loadable), basic events,
# interface_id, raw orientation, and all TCA columns.
_alignment_features = set(ALIGNMENT_BASE_FEATURE_COLS)
# Include segmentation features ONLY if they are expected in the raw data
# (e.g., tx, ty, tz, r11..r33 are usually raw data)
_segmentation_features_loadable = set(f for f in SEGMENTATION_FEATURE_COLS if f not in GMM_ORIENT_LOG_MAP_COLS) # Exclude derived log maps
_basic_event_cols = set(col for col in EVENT_COLUMNS.values() if isinstance(col, str))
_tca_cols = set(ALL_TCA_COLS_TO_LOAD)
_required_misc_cols = set(['interface_id'])
_raw_orientation_cols = set(ROT_MAT_COLS)

# Define the list of columns to ACTUALLY load from CSV
ALL_LOAD_COLS = sorted(list(
    _alignment_features |
    _segmentation_features_loadable | # Only loadable segmentation features
    _basic_event_cols |
    _tca_cols |
    _required_misc_cols |
    _raw_orientation_cols
))

# NOTE: PLOT_FEATURE_COLS might contain derived columns (like vx_log).
#       These derived columns are NOT loaded here but are expected to be created
#       during preprocessing in segment_analyzer.py before plotting.
#       The plotting function will use the 'processed_aligned_trajs' which HAVE these columns.
#       The tube calculation for plotting must also use 'processed_aligned_trajs'.

# --- Print Loaded Configuration ---
print("--- Configuration Loaded ---")
print(f"PARENT_FOLDER_PATH: {PARENT_FOLDER_PATH}")
print(f"LOAD_ALL_FILES: {LOAD_ALL_FILES}")
print(f"ALIGNMENT_TYPE: {ALIGNMENT_TYPE}")
print(f"USE_FASTDTW_APPROXIMATION: {USE_FASTDTW_APPROXIMATION}")
print(f"SEGMENTATION_FEATURE_COLS: {SEGMENTATION_FEATURE_COLS}")
print(f"MAX_SEGMENTS: {MAX_SEGMENTS}")
print(f"LAMBDA_PENALTY: {LAMBDA_PENALTY}")
print(f"MAX_DP_LENGTH: {MAX_DP_LENGTH}")
print(f"ROT_MAT_COLS: {ROT_MAT_COLS}")
print(f"GMM_STATE_COLS: {GMM_STATE_COLS}")
print(f"GMM_N_COMPONENTS: {GMM_N_COMPONENTS}")
print(f"ANALYZE_EVENT_CROSS_SECTIONS: {ANALYZE_EVENT_CROSS_SECTIONS}")
print(f"EVENTS_FOR_CROSS_SECTIONS: {EVENTS_FOR_CROSS_SECTIONS}")
print(f"CROSS_SECTION_STATS_OUTPUT_PATH: {CROSS_SECTION_STATS_OUTPUT_PATH}")
print(f"RUN_BACKUP_GLOBAL_GMM: {RUN_BACKUP_GLOBAL_GMM}")
print(f"BACKUP_GMM_N_COMPONENTS: {BACKUP_GMM_N_COMPONENTS}")
print(f"BACKUP_GMM_OUTPUT_PATH: {BACKUP_GMM_OUTPUT_PATH}")
print(f"PLOT_FEATURE_COLS: {PLOT_FEATURE_COLS}")
print(f"PLOT_ORIENTATION_DEVIATION: {PLOT_ORIENTATION_DEVIATION}")
print(f"Columns To Load (Unique, Sorted): {sorted(list(set(ALL_LOAD_COLS)))}") # Show the actual load list
print("-" * 25)
