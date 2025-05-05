#!/usr/bin/env python3
import time
import sys
import os
import yaml # For saving results
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings # To suppress specific warnings like KMeans convergence
import traceback # For detailed error printing

# --- Scikit-learn & SciPy ---
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.exceptions import ConvergenceWarning
    from scipy.spatial.transform import Rotation as R, Slerp # Import Slerp directly
    from scipy.interpolate import interp1d
    import numpy.linalg
except ImportError as e:
    print(f"FATAL ERROR: Missing essential libraries (scikit-learn, scipy). Install them: pip install scikit-learn scipy")
    print(f"Error details: {e}")
    sys.exit(1)

# --- Ensure sibling modules can be imported ---
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# --- Import necessary functions from project modules ---
try:
    import config
    from data_loading import load_selected_data, find_events
    from alignment import align_trajectories # Keep alignment import
    from segmentation import (calculate_tube, downsample_tube_data,
                              find_optimal_segmentation, map_boundaries_to_original)
    # Import event finding helpers from alignment (if needed, or keep in segmentation)
    from alignment import find_pre_post_waypoints, find_first_gripper_changes
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary project modules.")
    print(f"Ensure all .py files are in the same directory: {script_dir}")
    print(f"Error details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during project imports: {e}")
    sys.exit(1)

# --- Helper Functions for Orientation & State ---

def convert_to_quaternions(df, rot_mat_cols=config.ROT_MAT_COLS):
    """Converts rotation matrix columns in a DataFrame to quaternions (w, x, y, z)."""
    if not all(col in df.columns for col in rot_mat_cols):
        # print(f"Warning: Missing one or more rotation columns ({rot_mat_cols}) in DataFrame. Cannot convert to quaternions.")
        return None # Indicate failure
    try:
        # Reshape the selected columns into (N, 3, 3) matrices
        rot_matrices = df[rot_mat_cols].values.reshape(-1, 3, 3)
        # Convert to Rotation objects
        rotations = R.from_matrix(rot_matrices)
        # Return as quaternions (w, x, y, z) - SciPy default is (x, y, z, w)
        quats_xyzw = rotations.as_quat()
        # Reorder to w, x, y, z for consistency if desired (adjust if needed)
        quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
        return quats_wxyz
    except ValueError as e:
        print(f"Error converting rotation matrices to quaternions: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during quaternion conversion: {e}")
        return None

def convert_to_log_map(quaternions_wxyz):
    """Converts quaternions (w, x, y, z) to orientation log map vectors (axis-angle)."""
    if quaternions_wxyz is None or quaternions_wxyz.ndim != 2 or quaternions_wxyz.shape[1] != 4:
        print("Warning: Invalid input for log map conversion.")
        return None
    try:
        # SciPy expects (x, y, z, w)
        quats_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]
        rotations = R.from_quat(quats_xyzw)
        # Get rotation vectors (axis-angle representation, magnitude is angle)
        log_map_vectors = rotations.as_rotvec()
        return log_map_vectors
    except Exception as e:
        print(f"Error converting quaternions to log map: {e}")
        return None

# --- MODIFIED Helper Function ---
# Renamed from get_state_at_time to get_state_at_time_mod
# Now returns position_vector and quaternion_wxyz separately
def get_state_at_time_mod(traj_df, time_index, pos_cols, quat_col_name='quat_wxyz'):
    """
    Gets the position vector and quaternion at a specific time index,
    using linear interpolation for position and Slerp for quaternions.

    Args:
        traj_df (pd.DataFrame): Single trajectory DataFrame with pos_cols and 'quat_wxyz'.
        time_index (float): The target time index (can be fractional).
        pos_cols (list): List of column names for the position vector.
        quat_col_name (str): Name of the column holding the [w, x, y, z] quaternion arrays.

    Returns:
        tuple: (position_vector, quaternion_wxyz) or (None, None) if error/out of bounds.
               position_vector is a numpy array (len(pos_cols),).
               quaternion_wxyz is a numpy array (4,).
    """
    if traj_df.empty: return None, None # Handle empty input df
    min_time = traj_df.index.min()
    max_time = traj_df.index.max()

    # Handle edge case where df has only one row
    if len(traj_df) == 1:
        if np.isclose(time_index, min_time): # Use isclose for float comparison
            try:
                # Check if columns exist before accessing
                if not all(col in traj_df.columns for col in pos_cols):
                    # print(f"Warning: Missing position columns in single-row DataFrame.") # Reduce verbosity
                    return None, None
                if quat_col_name not in traj_df.columns:
                     # print(f"Warning: Missing quaternion column '{quat_col_name}' in single-row DataFrame.") # Reduce verbosity
                     return None, None

                position_vector = traj_df.loc[min_time, pos_cols].values.astype(float)
                quaternion_wxyz = traj_df.loc[min_time, quat_col_name]
                # Ensure quaternion is numpy array if needed downstream
                if not isinstance(quaternion_wxyz, np.ndarray): quaternion_wxyz = np.array(quaternion_wxyz)
                # Check shape and finite values
                if quaternion_wxyz.shape != (4,) or not np.all(np.isfinite(quaternion_wxyz)):
                     # print(f"Warning: Invalid quaternion data in single-row DataFrame.") # Reduce verbosity
                     return None, None
                return position_vector, quaternion_wxyz
            except Exception as e:
                 print(f"Error accessing data in single-row DataFrame: {e}")
                 return None, None
        else:
            return None, None # Cannot interpolate with one point if time doesn't match

    if not (min_time <= time_index <= max_time):
        # print(f"Warning: Target time index {time_index} is outside trajectory bounds [{min_time}, {max_time}].") # Reduce verbosity
        return None, None

    # Check if the time index exists exactly
    if time_index in traj_df.index:
        try:
            # Check columns exist
            if not all(col in traj_df.columns for col in pos_cols):
                # print(f"Warning: Missing position columns for exact time index {time_index}.") # Reduce verbosity
                return None, None
            if quat_col_name not in traj_df.columns:
                # print(f"Warning: Missing quaternion column '{quat_col_name}' for exact time index {time_index}.") # Reduce verbosity
                return None, None

            position_vector = traj_df.loc[time_index, pos_cols].values.astype(float)
            quaternion_wxyz = traj_df.loc[time_index, quat_col_name]
            if not isinstance(quaternion_wxyz, np.ndarray): quaternion_wxyz = np.array(quaternion_wxyz)
            # Check shape and finite values
            if quaternion_wxyz.shape != (4,) or not np.all(np.isfinite(quaternion_wxyz)):
                 # print(f"Warning: Invalid quaternion data at exact time index {time_index}.") # Reduce verbosity
                 return None, None
            return position_vector, quaternion_wxyz
        except Exception as e:
            print(f"Error accessing data for exact time index {time_index}: {e}")
            return None, None
    else:
        # Interpolate
        idx_before, idx_after = None, None
        try:
            # Find bounding indices
            valid_indices = traj_df.index
            indices_before = valid_indices[valid_indices < time_index]
            indices_after = valid_indices[valid_indices > time_index]

            if not indices_before.empty: idx_before = indices_before.max()
            if not indices_after.empty: idx_after = indices_after.min()

            # Handle cases where time_index might be numerically close to bounds
            if idx_before is None and np.isclose(time_index, min_time): idx_before = min_time
            if idx_after is None and np.isclose(time_index, max_time): idx_after = max_time

            # Check if bounding indices are valid
            if idx_before is None or idx_after is None:
                # print(f"Warning: Could not find valid bounding indices for interpolation at time {time_index}.") # Reduce verbosity
                return None, None
            if np.isclose(idx_before, idx_after): # If indices are the same (or very close)
                 # Recursively call to handle potential single point case again, avoids duplicating logic
                 return get_state_at_time_mod(traj_df, idx_before, pos_cols, quat_col_name)

            # Check required columns exist at boundary points
            if not all(col in traj_df.columns for col in pos_cols):
                 # print(f"Warning: Missing position columns for interpolation bounds {idx_before}/{idx_after}.") # Reduce verbosity
                 return None, None
            if quat_col_name not in traj_df.columns:
                 # print(f"Warning: Missing quaternion column '{quat_col_name}' for interpolation bounds {idx_before}/{idx_after}.") # Reduce verbosity
                 return None, None

            # Interpolation factor
            t_factor = (time_index - idx_before) / (idx_after - idx_before)

            # --- Linear Interpolation for Position Vector ---
            state_before = traj_df.loc[idx_before, pos_cols].values.astype(float)
            state_after = traj_df.loc[idx_after, pos_cols].values.astype(float)
            # Check for NaNs in position before interpolating
            if not np.all(np.isfinite(state_before)) or not np.all(np.isfinite(state_after)):
                 # print(f"Warning: Non-finite position values found during interpolation at t={time_index}.") # Reduce verbosity
                 return None, None
            position_vector = state_before + (state_after - state_before) * t_factor # Interpolate ONLY position

            # --- Spherical Linear Interpolation (Slerp) for Quaternions ---
            quat_before_wxyz = traj_df.loc[idx_before, quat_col_name]
            quat_after_wxyz = traj_df.loc[idx_after, quat_col_name]

            if not isinstance(quat_before_wxyz, np.ndarray): quat_before_wxyz = np.array(quat_before_wxyz)
            if not isinstance(quat_after_wxyz, np.ndarray): quat_after_wxyz = np.array(quat_after_wxyz)

            # Check quaternion validity before Slerp
            if quat_before_wxyz.shape != (4,) or quat_after_wxyz.shape != (4,):
                 # print(f"Warning: Invalid quaternion shape found during interpolation at t={time_index}.") # Reduce verbosity
                 return None, None
            if not np.all(np.isfinite(quat_before_wxyz)) or not np.all(np.isfinite(quat_after_wxyz)):
                 # print(f"Warning: Non-finite quaternion values found during interpolation at t={time_index}.") # Reduce verbosity
                 return None, None
            # Normalize quaternions before Slerp for robustness
            norm_before = np.linalg.norm(quat_before_wxyz)
            norm_after = np.linalg.norm(quat_after_wxyz)
            if norm_before < 1e-6 or norm_after < 1e-6: # Avoid division by zero
                 # print(f"Warning: Near-zero quaternion found during interpolation at t={time_index}.") # Reduce verbosity
                 # Decide fallback: return nearest neighbor? Or None? Returning None is safer.
                 return None, None
            quat_before_wxyz = quat_before_wxyz / norm_before
            quat_after_wxyz = quat_after_wxyz / norm_after

            # Convert to SciPy Rotation objects (needs xyzw format)
            key_rots = R.from_quat([quat_before_wxyz[[1, 2, 3, 0]], quat_after_wxyz[[1, 2, 3, 0]]])
            key_times = [idx_before, idx_after] # Use original indices as times

            # Create Slerp object
            slerp_interpolator = Slerp(key_times, key_rots)

            # Interpolate rotation at the target time
            interp_rotation = slerp_interpolator([time_index])[0] # Slerp returns a list

            # Convert back to quaternion [w, x, y, z]
            interp_quat_xyzw = interp_rotation.as_quat()
            quaternion_wxyz = interp_quat_xyzw[[3, 0, 1, 2]]

            return position_vector, quaternion_wxyz

        except Exception as e:
            bound_msg = f"between {idx_before} and {idx_after}" if idx_before is not None and idx_after is not None else "(bounds not determined)"
            print(f"Error during interpolation at time {time_index} {bound_msg}: {e}")
            traceback.print_exc() # Print full traceback for debugging interpolation errors
            return None, None


def replace_nan_with_none(obj):
    """ Recursively replaces NaN values with None in nested data structures. """
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(elem) for elem in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        if np.isnan(obj): return None
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        if np.issubdtype(obj.dtype, np.number):
             obj = np.where(np.isnan(obj), None, obj)
        return replace_nan_with_none(obj.tolist())
    return obj

# --- Main Analysis Function ---
def analyze_segments_cross_section():
    """
    Runs the main processing pipeline and calculates GMM (K=1)/statistical properties
    at cross-sections using the configured state representation.
    Saves results to YAML.
    Returns a dictionary containing data needed for plotting by main.py.
    """
    overall_start_time = time.time()

    # Initialize variables to be returned, in case of early exit
    processed_aligned_trajs = None
    min_vals_plot_tube = None
    max_vals_plot_tube = None
    optimal_segment_indices_orig = None
    valid_mapped_events_list = None
    raw_costs_per_segment_count = None
    optimal_num_segments = -1 # Indicate failure initially
    n_timesteps_orig = 0 # Initialize

    try: # Wrap the whole process in try/except

        # --- Steps 1-7: Load, Find Events, Align, Tube, Downsample, Segment, Map Boundaries ---
        # --- 1. Load Data ---
        print("--- 1. Loading Data ---")
        original_trajs_df_list, loaded_files = load_selected_data(
            config.PARENT_FOLDER_PATH, load_all=config.LOAD_ALL_FILES,
            file_list=config.FILE_LIST, columns_to_load=config.ALL_LOAD_COLS
        )
        if not original_trajs_df_list:
            print("Exiting: No trajectories loaded.")
            return None
        n_trajectories = len(original_trajs_df_list)
        print(f"Loaded {n_trajectories} trajectories.")

        # --- 2. Find Events ---
        print("\n--- 2. Finding Events in Original Trajectories ---")
        original_events_list = []
        for i, df in enumerate(original_trajs_df_list):
            if df.empty: events = {'start': [], 'end': [], 'state_change': [], 'wp_saved': [], 'gripper_change': []}
            else: events = find_events(df, config.EVENT_COLUMNS)
            original_events_list.append(events)

        # --- 3. Align Trajectories ---
        print("\n--- 3. Aligning Trajectories & Mapping Events ---")
        aligned_trajs_df_list, mapped_events_list, ref_idx = align_trajectories(
            original_trajs_df_list, original_events_list, config.ALIGNMENT_TYPE,
            config.ALIGNMENT_BASE_FEATURE_COLS, config.ALL_LOAD_COLS, config.NORMALIZE_FOR_COMBINED
        )
        if ref_idx == -1 or not aligned_trajs_df_list:
            print("Exiting: Alignment failed.")
            return None

        valid_indices = [i for i, df in enumerate(aligned_trajs_df_list) if not df.empty]
        valid_aligned_trajs_df = [aligned_trajs_df_list[i] for i in valid_indices]
        valid_mapped_events_list = [mapped_events_list[i] for i in valid_indices] # Keep this for return
        if not valid_aligned_trajs_df:
            print("Exiting: Alignment produced no valid trajectories.")
            return None
        n_valid_trajectories = len(valid_aligned_trajs_df)
        print(f"Produced {n_valid_trajectories} valid aligned trajectories.")

        # --- 4. Determine Original Timesteps ---
        n_timesteps_orig = len(valid_aligned_trajs_df[0]) if valid_aligned_trajs_df else 0
        if n_timesteps_orig <= 1:
             print("Warning: Not enough timesteps in aligned data for meaningful analysis.")
             # Consider exiting if analysis requires multiple timesteps
             # return None
        else:
            print(f"  Determined original timesteps: {n_timesteps_orig}")

        # --- 5. Downsample for DP ---
        print("\n--- 5. Downsampling Data for DP (if needed) ---")
        # Calculate tube using features defined for segmentation cost
        min_vals_dp_basis, max_vals_dp_basis, _ = calculate_tube(
             valid_aligned_trajs_df, config.SEGMENTATION_FEATURE_COLS
        )
        if min_vals_dp_basis is None:
            print("Exiting: Failed to calculate tube for DP basis features.")
            return None

        min_vals_dp, max_vals_dp, n_timesteps_dp, stride = downsample_tube_data(
            min_vals_dp_basis, max_vals_dp_basis, config.MAX_DP_LENGTH
        )

        # --- 6. Find Optimal Segmentation ---
        print("\n--- 6. Finding Optimal Segmentation (DP) ---")
        segment_indices_dp, optimal_num_segments, _, raw_costs_per_segment_count = find_optimal_segmentation(
            min_vals_dp, max_vals_dp, n_timesteps_dp, config.MAX_SEGMENTS, config.LAMBDA_PENALTY,
            config.SEGMENTATION_FEATURE_COLS, config.POS_COLS, config.ROTATION_WEIGHT,
            valid_mapped_events_list, stride
        )
        if optimal_num_segments == -1 or segment_indices_dp is None:
            print("Exiting: Segmentation failed.")
            return None

        # --- 7. Map Boundaries to Original Scale ---
        print("\n--- 7. Mapping Segment Boundaries ---")
        optimal_segment_indices_orig = map_boundaries_to_original( # Keep this for return
            segment_indices_dp, stride, n_timesteps_orig
        )
        print(f"Optimal Segmentation Boundaries (Original Time Scale): {optimal_segment_indices_orig}")

        # --- 8. Data Preprocessing for Analysis (Add Quat, LogMap, FlatQuat) ---
        print("\n--- 8. Preprocessing Aligned Data for Analysis ---")
        processed_aligned_trajs = [] # Keep this for return
        temp_processed_trajs = []
        log_map_cols = ['vx_log', 'vy_log', 'vz_log'] # Define log map column names

        for i, traj_df in enumerate(valid_aligned_trajs_df):
            print(f"  Preprocessing trajectory {i}...")
            traj_df_copy = traj_df.copy()

            # Add Quaternion Column ('quat_wxyz')
            quats_wxyz = convert_to_quaternions(traj_df_copy)
            if quats_wxyz is None:
                print(f"  Error: Failed to convert rotations to quaternions for trajectory {i}. Skipping trajectory.")
                continue
            traj_df_copy['quat_wxyz'] = list(quats_wxyz)

            # Add Log Map Columns ('vx_log', 'vy_log', 'vz_log')
            log_map_vectors = convert_to_log_map(quats_wxyz)
            if log_map_vectors is None:
                print(f"  Error: Failed to convert quaternions to log map for trajectory {i}. Skipping trajectory.")
                continue
            for j, col_name in enumerate(log_map_cols):
                traj_df_copy[col_name] = log_map_vectors[:, j]

            # Add Flattened Quaternion Columns ('qw', 'qx', 'qy', 'qz')
            traj_df_copy['qw'] = quats_wxyz[:, 0]
            traj_df_copy['qx'] = quats_wxyz[:, 1]
            traj_df_copy['qy'] = quats_wxyz[:, 2]
            traj_df_copy['qz'] = quats_wxyz[:, 3]

            temp_processed_trajs.append(traj_df_copy)

        processed_aligned_trajs = temp_processed_trajs # Assign the fully processed list

        if not processed_aligned_trajs:
            print("Exiting: No trajectories remaining after preprocessing (orientation conversion failed).")
            return None
        n_processed_trajectories = len(processed_aligned_trajs)
        print(f"Finished preprocessing {n_processed_trajectories} trajectories.")

        # --- Calculate Tube for Plotting (using PLOT_FEATURE_COLS) ---
        print("\n--- Calculating Tube (for Visualization Features) ---")
        if processed_aligned_trajs and config.PLOT_FEATURE_COLS:
             min_vals_plot_tube, max_vals_plot_tube, _ = calculate_tube(
                 processed_aligned_trajs, config.PLOT_FEATURE_COLS # Use processed data & plot features
             )
             if min_vals_plot_tube is None:
                  print("  Warning: Failed to calculate tube for plotting features.")
             else:
                  print(f"  Tube for plotting calculated using {len(config.PLOT_FEATURE_COLS)} features.")
        else:
             print("  Skipping tube calculation for plotting (no processed trajectories or no plot features defined).")
             min_vals_plot_tube, max_vals_plot_tube = None, None

        # --- 9. Identify Cross-Section Times ---
        print("\n--- 9. Identifying Cross-Section Times ---")
        cross_section_times = defaultdict(lambda: {'type': None, 'involved_events': set()})
        if n_timesteps_orig > 0:
            # Add DP boundary times
            for boundary_idx in optimal_segment_indices_orig:
                clamped_idx = np.clip(boundary_idx, 0, n_timesteps_orig - 1)
                cross_section_times[clamped_idx]['type'] = 'boundary'
                # print(f"  Added boundary cross-section at index: {clamped_idx}") # Reduce verbosity

            # Add event average times (optional)
            if config.ANALYZE_EVENT_CROSS_SECTIONS:
                print("  Analyzing event-based cross-sections...")
                for i in range(len(optimal_segment_indices_orig) - 1):
                    seg_start = optimal_segment_indices_orig[i]
                    seg_end = optimal_segment_indices_orig[i+1] - 1
                    if seg_start > seg_end: continue

                    segment_event_indices = []
                    segment_event_types = set()

                    # Use valid_mapped_events_list which corresponds to processed_aligned_trajs
                    for mapped_events in valid_mapped_events_list:
                        for event_type in config.EVENTS_FOR_CROSS_SECTIONS:
                            if event_type in mapped_events:
                                for event_idx in mapped_events[event_type]:
                                    if seg_start <= event_idx <= seg_end:
                                        segment_event_indices.append(event_idx)
                                        segment_event_types.add(event_type)

                    if segment_event_indices:
                        t_ave_event = np.mean(segment_event_indices)
                        t_ave_event_idx = float(t_ave_event) # Keep as float for potential interpolation
                        t_ave_event_idx = np.clip(t_ave_event_idx, 0, n_timesteps_orig - 1)

                        # Check if it coincides (numerically) with a boundary
                        is_boundary = False
                        for boundary_t in optimal_segment_indices_orig:
                            if np.isclose(t_ave_event_idx, float(boundary_t)):
                                is_boundary = True; break

                        if not is_boundary:
                             cross_section_times[t_ave_event_idx]['type'] = 'event'
                             cross_section_times[t_ave_event_idx]['involved_events'].update(segment_event_types)
                             # print(f"  Added event cross-section at average index: {t_ave_event_idx:.2f} (Segment {i}, Events: {segment_event_types})") # Reduce verbosity
        else:
            print("  Skipping cross-section identification as n_timesteps_orig is 0.")

        sorted_cross_section_times = sorted(cross_section_times.keys())
        print(f"  Total unique cross-section times identified: {len(sorted_cross_section_times)}")

        # --- 10. Analyze Cross-Sections (Using GMM K=1) ---
        print("\n--- 10. Analyzing Cross-Sections (GMM K=1 for Mean/Covariance) ---")
        cross_section_results = []

        # Determine State Columns for GMM based on Config
        state_cols_for_gmm = []
        orientation_cols_names = []

        if config.ANALYSIS_ORIENTATION_REPRESENTATION == 'rotation_matrix':
            orientation_cols_names = config.ROT_MAT_COLS
            state_cols_for_gmm = config.ANALYSIS_POS_COLS + orientation_cols_names
            print(f"  Using state representation: Position + Rotation Matrix ({len(state_cols_for_gmm)} dims)")
        elif config.ANALYSIS_ORIENTATION_REPRESENTATION == 'log_map':
            orientation_cols_names = log_map_cols # Use the list defined in Step 8
            state_cols_for_gmm = config.ANALYSIS_POS_COLS + orientation_cols_names
            print(f"  Using state representation: Position + Log Map ({len(state_cols_for_gmm)} dims)")
        elif config.ANALYSIS_ORIENTATION_REPRESENTATION == 'quaternion':
            orientation_cols_names = ['qw', 'qx', 'qy', 'qz']
            state_cols_for_gmm = config.ANALYSIS_POS_COLS + orientation_cols_names
            print(f"  Using state representation: Position + Quaternion ({len(state_cols_for_gmm)} dims)")
        else:
            print(f"Error: Invalid ANALYSIS_ORIENTATION_REPRESENTATION '{config.ANALYSIS_ORIENTATION_REPRESENTATION}' in config. Exiting analysis.")
            return None

        # Loop through Cross-Section Times
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

        for t_cs in sorted_cross_section_times:
            print(f"  Analyzing cross-section at time index: {t_cs:.2f}...")
            snapshot_states = []
            snapshot_positions_for_bounds = []

            for traj_df in processed_aligned_trajs:
                # Use the modified get_state_at_time_mod function
                pos_vector, quat_wxyz = get_state_at_time_mod(traj_df, float(t_cs), config.ANALYSIS_POS_COLS)

                if pos_vector is not None and quat_wxyz is not None:
                    snapshot_positions_for_bounds.append(pos_vector)

                    # Construct the full state vector for GMM fitting
                    current_state_vector_list = list(pos_vector)
                    orient_rep = None
                    valid_orient_rep = True

                    # Derive the chosen orientation representation
                    if config.ANALYSIS_ORIENTATION_REPRESENTATION == 'rotation_matrix':
                        try:
                            rot = R.from_quat(quat_wxyz[[1, 2, 3, 0]])
                            orient_rep = rot.as_matrix().flatten().tolist()
                        except Exception as e:
                            orient_rep = [np.nan] * 9
                            valid_orient_rep = False
                    elif config.ANALYSIS_ORIENTATION_REPRESENTATION == 'log_map':
                        try:
                            rot = R.from_quat(quat_wxyz[[1, 2, 3, 0]])
                            orient_rep = rot.as_rotvec().tolist()
                        except Exception as e:
                            orient_rep = [np.nan] * 3
                            valid_orient_rep = False
                    elif config.ANALYSIS_ORIENTATION_REPRESENTATION == 'quaternion':
                        orient_rep = quat_wxyz.tolist()

                    # Append and add to snapshots if valid
                    if orient_rep is not None:
                        current_state_vector_list.extend(orient_rep)
                        if valid_orient_rep and len(current_state_vector_list) == len(state_cols_for_gmm):
                            snapshot_states.append(current_state_vector_list)

            # Fit GMM (K=1)
            n_snapshots = len(snapshot_states)
            gmm_params = None
            n_clean_snapshots = 0

            if n_snapshots > 0:
                snapshot_states_np = np.array(snapshot_states)
                valid_rows_mask = ~np.isnan(snapshot_states_np).any(axis=1)
                snapshot_states_clean_np = snapshot_states_np[valid_rows_mask, :]
                n_clean_snapshots = len(snapshot_states_clean_np)

                n_components_actual = min(config.GMM_N_COMPONENTS, n_clean_snapshots) if n_clean_snapshots > 0 else 0

                if n_components_actual >= 1:
                    try:
                        gmm = GaussianMixture(n_components=n_components_actual, covariance_type='full',
                                              random_state=0, n_init=5, max_iter=200, tol=1e-3)
                        gmm.fit(snapshot_states_clean_np)
                        gmm_params = {
                            'weights': gmm.weights_.tolist(), 'means': gmm.means_.tolist(),
                            'covariances': gmm.covariances_.tolist(), 'converged': gmm.converged_,
                            'n_components_used': n_components_actual
                        }
                        if not gmm.converged_:
                            print(f"    Warning: GMM with {n_components_actual} components did not converge at time {t_cs:.2f}.")
                    except Exception as e:
                        print(f"    Error training GMM with {n_components_actual} components at time {t_cs:.2f}: {e}")
                        gmm_params = None
                # else: No warning needed if n_components_actual is 0

            # Calculate Position Bounds
            pos_min_bounds, pos_max_bounds = None, None
            if snapshot_positions_for_bounds:
                snapshot_positions_np = np.array(snapshot_positions_for_bounds)
                pos_min_bounds = np.min(snapshot_positions_np, axis=0).tolist()
                pos_max_bounds = np.max(snapshot_positions_np, axis=0).tolist()

            # Append Results
            cs_info = cross_section_times[t_cs]
            cross_section_results.append({
                "time_index": float(t_cs),
                "type": cs_info['type'],
                "involved_events": sorted(list(cs_info['involved_events'])),
                "num_snapshots": n_snapshots,
                "num_valid_snapshots": n_clean_snapshots,
                "state_representation_used": config.ANALYSIS_ORIENTATION_REPRESENTATION,
                "state_feature_names": state_cols_for_gmm,
                "segment_gmm_params": gmm_params, # Store GMM results (mean/cov if K=1)
                "pos_min_bounds": pos_min_bounds,
                "pos_max_bounds": pos_max_bounds,
            })

        warnings.filterwarnings("default", category=ConvergenceWarning, module="sklearn")

        # --- 11. Save Cross-Section Results ---
        print("\n--- 11. Saving Cross-Section Analysis Results ---")
        output_path_str = config.CROSS_SECTION_STATS_OUTPUT_PATH
        output_path = Path(output_path_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cleaned_stats = replace_nan_with_none(cross_section_results)
            with open(output_path, 'w') as f:
                yaml.dump(cleaned_stats, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
            print(f"Cross-section statistics saved successfully to: {output_path}")
        except Exception as e:
            print(f"Error saving cross-section statistics to {output_path}: {e}")

        # --- 12. Backup Global GMM Analysis (REMOVED) ---

    except Exception as e:
        print(f"\n--- FATAL ERROR during analysis pipeline ---")
        print(f"Error: {e}")
        traceback.print_exc()
        return None # Indicate failure

    finally:
        # --- Prepare Return Dictionary ---
        results_for_main = {
            "processed_aligned_trajs": processed_aligned_trajs,
            "min_vals_plot_tube": min_vals_plot_tube, # Tube for plot features
            "max_vals_plot_tube": max_vals_plot_tube, # Tube for plot features
            "optimal_segment_indices_orig": optimal_segment_indices_orig,
            "valid_mapped_events_list": valid_mapped_events_list,
            "raw_costs_per_segment_count": raw_costs_per_segment_count,
            "optimal_num_segments": optimal_num_segments
        }
        overall_duration = time.time() - overall_start_time
        print(f"\n--- Analysis Function Finished in {overall_duration:.2f} seconds ---")
        return results_for_main

# --- Main execution block (for testing if run directly) ---
if __name__ == "__main__":
    analysis_output = analyze_segments_cross_section()
    if analysis_output:
        print("\nAnalysis completed. Output dictionary contains:")
        for key, value in analysis_output.items():
             if isinstance(value, (list, tuple)) and value and isinstance(value[0], (pd.DataFrame, np.ndarray)):
                 print(f" - {key}: List of {len(value)} DataFrames/Arrays")
             elif isinstance(value, (np.ndarray)):
                  print(f" - {key}: Numpy array with shape {value.shape}")
             elif isinstance(value, (list, tuple)):
                  print(f" - {key}: List/Tuple with {len(value)} elements")
             else:
                  print(f" - {key}: {type(value)}")
    else:
        print("\nAnalysis failed or returned no results.")

