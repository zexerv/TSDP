# -*- coding: utf-8 -*-
"""
Configuration file for the segdp trajectory segmentation and analysis pipeline.
MODIFIED TO:
- Use Rotation Matrix for cross-section analysis state vector.
- Use Rotation Matrix for the main segmentation plot features.
- Set GMM components to 1 for mean/covariance calculation.
- Remove backup GMM.
"""

import os
import numpy as np
import re # Import regex module

# ==============================================================================
# FILE PATHS & LOADING
# Keywords: data, input, csv, loading
# ==============================================================================
# Absolute path to the parent directory containing subfolders like 'button', 'lever', etc.
# !!! ADJUST THIS PATH TO YOUR ACTUAL DATA LOCATION !!!
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
ALIGNMENT_ROTATION_WEIGHT = 0.7
# Weight multiplier for derivative features relative to position features (only for 'Combined' type).
ALIGNMENT_DERIVATIVE_WEIGHT = 0.6

# --- Alignment Method Selection ---
# If True, use fastdtw library (approximate, faster, no event penalties).
# If False, use manual DTW implementation (exact, slower, allows event penalties).
USE_FASTDTW_APPROXIMATION = False # Set to False to use manual DTW with penalties

# --- Event Penalties (Only used if USE_FASTDTW_APPROXIMATION = False) ---
ALIGNMENT_STATE_CHANGE_PENALTY = 1.0
ALIGNMENT_PRE_WP1_PENALTY = 0.8
ALIGNMENT_POST_WP1_PENALTY = 0.8
ALIGNMENT_PRE_WP2_PENALTY = 0.5
ALIGNMENT_POST_WP2_PENALTY = 0.5
ALIGNMENT_PRE_GRIPPER1_PENALTY = 0.7
ALIGNMENT_POST_GRIPPER1_PENALTY = 0.7

# --- Conditional Penalty Activation (Only used if USE_FASTDTW_APPROXIMATION = False) ---
WAYPOINT_PRESENCE_THRESHOLD = 0.8

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
# Assuming 3x3 Rotation Matrix format
ROT_MAT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
# Threshold for detecting significant lever movement in find_events.
LEVER_CHANGE_THRESHOLD = 0.001
# List of ALL potential TCA columns that might appear in ANY CSV file.
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
# SEGMENTATION
# Keywords: segmentation, dynamic programming, cost, segments, tube, events
# ==============================================================================
# Features used to define the tube boundaries for segmentation cost calculation.
# <<< MODIFIED: Using Position + Rotation Matrix for segmentation cost >>>
SEGMENTATION_FEATURE_COLS = POS_COLS + ROT_MAT_COLS

# Maximum number of segments to consider in the dynamic programming optimization.
MAX_SEGMENTS = 10
# Penalty added for each additional segment in the DP cost function (controls granularity).
LAMBDA_PENALTY = 0.2 # Adjust if needed based on new features
# Weight multiplier for rotation features relative to position features in the segmentation geometric cost.
# <<< This now applies to r11..r33. TUNE THIS VALUE based on results. >>>
ROTATION_WEIGHT = 0.5 # Example value, TUNE this

# Event co-occurrence penalties
SEGMENTATION_EVENT_COOCCURRENCE_PENALTY = 10.0
PENALIZE_NEIGHBOR_COOCCURRENCE = False
SEGMENTATION_NEIGHBOR_COOCCURRENCE_PENALTY = 10.0

# ==============================================================================
# DOWNSAMPLING
# Keywords: downsampling, performance, DP, length
# ==============================================================================
# Trajectories longer than this will be downsampled before segmentation DP calculation.
MAX_DP_LENGTH = 300

# ==============================================================================
# CROSS-SECTION ANALYSIS (Using GMM K=1 for Mean/Covariance)
# Keywords: analysis, cross-section, boundary, event, statistics, mean, covariance, gmm
# ==============================================================================
# Representation for orientation in the state vector during cross-section analysis GMM fitting.
# Options: 'rotation_matrix' (9 elements, r11-r33),
#          'log_map' (3 elements, vx_log, vy_log, vz_log),
#          'quaternion' (4 elements, qw, qx, qy, qz)
# <<< MODIFIED: Set to use rotation matrix for analysis >>>
ANALYSIS_ORIENTATION_REPRESENTATION = 'rotation_matrix'

# Columns defining the position part of the state vector
ANALYSIS_POS_COLS = ['tx', 'ty', 'tz'] # Same as POS_COLS usually

# Number of components for the GMM trained at each cross-section.
# <<< MODIFIED: Set to 1 to calculate the equivalent of mean and covariance >>>
GMM_N_COMPONENTS = 1

# Flag to enable analysis of event-based cross-sections (average time of events within a segment).
ANALYZE_EVENT_CROSS_SECTIONS = True

# Event types from mapped_events_list to consider for event-based cross-sections (if enabled).
EVENTS_FOR_CROSS_SECTIONS = ['state_change', 'wp_saved', 'gripper_change']

# Output path for the cross-section GMM analysis results (containing mean/cov if K=1)
# !!! ADJUST THIS PATH IF NEEDED !!!
CROSS_SECTION_STATS_OUTPUT_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/results/cross_section_stats.yaml'

# ==============================================================================
# VISUALIZATION
# Keywords: plot, visualization, output, figures
# ==============================================================================
# Number of columns in the multi-plot grid for segmentation results.
PLOT_GRID_COLUMNS = 4 # Adjust based on number of features plotted (Pos + RotMat = 12 features)
# Font family for matplotlib plots.
PLOT_FONT = 'serif'

# Features to include in the main segmentation plot generated by plot_segmentation_with_trapezoids
# <<< MODIFIED: Plot Position + Rotation Matrix >>>
PLOT_FEATURE_COLS = ANALYSIS_POS_COLS + ROT_MAT_COLS

# Orientation deviation plot feature removed
# PLOT_ORIENTATION_DEVIATION = False

# ==============================================================================
# DERIVED CONFIGURATION (Internal - Do Not Modify Manually)
# ==============================================================================
# --- Columns to Load ---
# Defines columns to be loaded from the initial CSV files.
_alignment_features = set(ALIGNMENT_BASE_FEATURE_COLS)
# Include segmentation features ONLY if they are expected in the raw data
_segmentation_features_loadable = set(f for f in SEGMENTATION_FEATURE_COLS) # Now includes RotMat cols
_basic_event_cols = set(col for col in EVENT_COLUMNS.values() if isinstance(col, str))
_tca_cols = set(ALL_TCA_COLS_TO_LOAD)
_required_misc_cols = set(['interface_id'])
# Ensure ROT_MAT_COLS are defined and included
_raw_orientation_cols = set(ROT_MAT_COLS)

# Define the list of columns to ACTUALLY load from CSV
ALL_LOAD_COLS = sorted(list(
    _alignment_features |
    _segmentation_features_loadable | # Includes RotMat now
    _basic_event_cols |
    _tca_cols |
    _required_misc_cols |
    _raw_orientation_cols # Explicitly includes RotMat
))

# --- Print Loaded Configuration ---
print("--- Configuration Loaded ---")
print(f"PARENT_FOLDER_PATH: {PARENT_FOLDER_PATH}")
print(f"LOAD_ALL_FILES: {LOAD_ALL_FILES}")
print(f"ALIGNMENT_TYPE: {ALIGNMENT_TYPE}")
print(f"USE_FASTDTW_APPROXIMATION: {USE_FASTDTW_APPROXIMATION}")
print(f"SEGMENTATION_FEATURE_COLS: {SEGMENTATION_FEATURE_COLS} (Count: {len(SEGMENTATION_FEATURE_COLS)})")
print(f"ROTATION_WEIGHT (Segmentation): {ROTATION_WEIGHT}")
print(f"MAX_SEGMENTS: {MAX_SEGMENTS}")
print(f"LAMBDA_PENALTY: {LAMBDA_PENALTY}")
print(f"MAX_DP_LENGTH: {MAX_DP_LENGTH}")
print(f"ROT_MAT_COLS: {ROT_MAT_COLS}")
print(f"ANALYSIS_ORIENTATION_REPRESENTATION: {ANALYSIS_ORIENTATION_REPRESENTATION}")
print(f"GMM_N_COMPONENTS: {GMM_N_COMPONENTS} (Set to 1 for Mean/Covariance)")
print(f"ANALYZE_EVENT_CROSS_SECTIONS: {ANALYZE_EVENT_CROSS_SECTIONS}")
print(f"EVENTS_FOR_CROSS_SECTIONS: {EVENTS_FOR_CROSS_SECTIONS}")
print(f"CROSS_SECTION_STATS_OUTPUT_PATH: {CROSS_SECTION_STATS_OUTPUT_PATH}")
print(f"PLOT_FEATURE_COLS: {PLOT_FEATURE_COLS} (Count: {len(PLOT_FEATURE_COLS)})")
# print(f"PLOT_ORIENTATION_DEVIATION: Feature Removed")
print(f"Columns To Load (Unique, Sorted): {sorted(list(set(ALL_LOAD_COLS)))}")
print("-" * 25)

