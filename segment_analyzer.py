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

# --- Scikit-learn & SciPy ---
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.exceptions import ConvergenceWarning
    # --- Corrected IMPORT ---
    from scipy.spatial.transform import Rotation as R, Slerp # Import Slerp directly
    # --- END Corrected IMPORT ---
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
    if quaternions_wxyz is None or quaternions_wxyz.shape[1] != 4:
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

def calculate_geometric_mean_quaternion(quaternions_wxyz):
    """
    Calculates the geometric mean of quaternions using SciPy's Rotation.mean().
    Input: numpy array of quaternions (N, 4) in [w, x, y, z] format.
    Output: numpy array (4,) representing the mean quaternion [w, x, y, z], or None.
    """
    if quaternions_wxyz is None or quaternions_wxyz.ndim != 2 or quaternions_wxyz.shape[1] != 4 or quaternions_wxyz.shape[0] == 0:
        # print("Warning: Invalid input for geometric mean calculation.")
        return None
    try:
        # Convert to SciPy format [x, y, z, w]
        quats_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]
        rotations = R.from_quat(quats_xyzw)
        # Calculate the mean rotation
        mean_rotation = rotations.mean()
        # Convert back to quaternion [w, x, y, z]
        mean_quat_xyzw = mean_rotation.as_quat()
        mean_quat_wxyz = mean_quat_xyzw[[3, 0, 1, 2]]
        return mean_quat_wxyz
    except Exception as e:
        print(f"Error calculating geometric mean quaternion: {e}")
        return None

def calculate_geodesic_distance(q1_wxyz, q2_wxyz):
    """
    Calculates the geodesic distance (angle in degrees) between two quaternions.
    Input: Quaternions q1, q2 as numpy arrays (4,) in [w, x, y, z] format.
    Output: Angle in degrees, or np.nan if error.
    """
    if q1_wxyz is None or q2_wxyz is None: return np.nan
    try:
        # Convert to SciPy format [x, y, z, w]
        q1_xyzw = q1_wxyz[[1, 2, 3, 0]]
        q2_xyzw = q2_wxyz[[1, 2, 3, 0]]
        r1 = R.from_quat(q1_xyzw)
        r2 = R.from_quat(q2_xyzw)
        # Relative rotation: r_rel = r1.inv() * r2
        # Angle is angle of rotation vector of r_rel
        diff_rotvec = (r1.inv() * r2).as_rotvec()
        angle_rad = np.linalg.norm(diff_rotvec)
        # Ensure angle is in [0, pi], angle_rad can sometimes be slightly > pi due to precision
        angle_rad = np.clip(angle_rad, 0, np.pi)
        return np.degrees(angle_rad)
    except Exception as e:
        # print(f"Warning: Error calculating geodesic distance: {e}") # Can be verbose
        return np.nan

def get_state_at_time(traj_df, time_index, state_cols, quat_col_name='quat_wxyz', logmap_cols=config.GMM_ORIENT_LOG_MAP_COLS):
    """
    Gets the state vector [pos, logmap] and quaternion at a specific time index,
    using linear interpolation for state and Slerp for quaternions if needed.

    Args:
        traj_df (pd.DataFrame): Single trajectory DataFrame with state, quat_wxyz, logmap cols.
        time_index (float): The target time index (can be fractional).
        state_cols (list): List of column names for the state vector (e.g., ['tx', 'ty', 'tz', 'vx_log', ...]).
        quat_col_name (str): Name of the column holding the [w, x, y, z] quaternion arrays.
        logmap_cols (list): Names of the log map columns.

    Returns:
        tuple: (state_vector, quaternion_wxyz) or (None, None) if error/out of bounds.
               state_vector is a numpy array (len(state_cols),).
               quaternion_wxyz is a numpy array (4,).
    """
    min_time = traj_df.index.min()
    max_time = traj_df.index.max()

    # Handle edge case where df has only one row
    if len(traj_df) == 1:
        if time_index == min_time:
             state_vector = traj_df.loc[time_index, state_cols].values.astype(float)
             quaternion_wxyz = traj_df.loc[time_index, quat_col_name]
             return state_vector, quaternion_wxyz
        else:
            return None, None # Cannot interpolate with one point

    if not (min_time <= time_index <= max_time):
        # print(f"Warning: Target time index {time_index} is outside trajectory bounds [{min_time}, {max_time}].")
        return None, None

    # Check if the time index exists exactly
    if time_index in traj_df.index:
        state_vector = traj_df.loc[time_index, state_cols].values.astype(float)
        quaternion_wxyz = traj_df.loc[time_index, quat_col_name] # Assumes quat is stored directly
        return state_vector, quaternion_wxyz
    else:
        # Interpolate
        idx_before, idx_after = None, None # Initialize for error message
        try:
            # Find bounding indices
            idx_before = traj_df.index[traj_df.index < time_index].max()
            idx_after = traj_df.index[traj_df.index > time_index].min()

            # Handle cases where time_index might be exactly on an endpoint after float conversion
            if pd.isna(idx_before) and np.isclose(time_index, min_time): # Use np.isclose for float comparison
                 idx_before = min_time
            if pd.isna(idx_after) and np.isclose(time_index, max_time):
                 idx_after = max_time

            # Check if bounding indices are valid
            if pd.isna(idx_before) or pd.isna(idx_after):
                 print(f"Warning: Could not find valid bounding indices for interpolation at time {time_index}.")
                 return None, None
            if idx_before == idx_after: # Should not happen if time_index not in index, but safety check
                 state_vector = traj_df.loc[idx_before, state_cols].values.astype(float)
                 quaternion_wxyz = traj_df.loc[idx_before, quat_col_name]
                 return state_vector, quaternion_wxyz

            # Interpolation factor
            # Avoid division by zero if indices are identical (should be caught above, but safety)
            if np.isclose(idx_after, idx_before):
                t_factor = 0.0
            else:
                t_factor = (time_index - idx_before) / (idx_after - idx_before)

            # --- Linear Interpolation for State Vector ---
            state_before = traj_df.loc[idx_before, state_cols].values.astype(float)
            state_after = traj_df.loc[idx_after, state_cols].values.astype(float)
            state_vector = state_before + (state_after - state_before) * t_factor

            # --- Spherical Linear Interpolation (Slerp) for Quaternions ---
            quat_before_wxyz = traj_df.loc[idx_before, quat_col_name]
            quat_after_wxyz = traj_df.loc[idx_after, quat_col_name]

            # Ensure quaternions are numpy arrays
            if not isinstance(quat_before_wxyz, np.ndarray): quat_before_wxyz = np.array(quat_before_wxyz)
            if not isinstance(quat_after_wxyz, np.ndarray): quat_after_wxyz = np.array(quat_after_wxyz)

            # Convert to SciPy Rotation objects (needs xyzw format)
            key_rots = R.from_quat([quat_before_wxyz[[1, 2, 3, 0]], quat_after_wxyz[[1, 2, 3, 0]]])
            key_times = [idx_before, idx_after] # Use original indices as times

            # --- Corrected SLERP USAGE ---
            # Create Slerp object directly using the imported class
            slerp_interpolator = Slerp(key_times, key_rots)
            # --- END Corrected SLERP USAGE ---

            # Interpolate rotation at the target time
            interp_rotation = slerp_interpolator([time_index])[0] # Slerp returns a list

            # Convert back to quaternion [w, x, y, z]
            interp_quat_xyzw = interp_rotation.as_quat()
            quaternion_wxyz = interp_quat_xyzw[[3, 0, 1, 2]]

            return state_vector, quaternion_wxyz

        except Exception as e:
            # Improved error message
            bound_msg = f"between {idx_before} and {idx_after}" if idx_before is not None and idx_after is not None else "(bounds not determined)"
            print(f"Error during interpolation at time {time_index} {bound_msg}: {e}")
            # Fallback: return state at nearest index? Or None? Let's return None.
            return None, None


def replace_nan_with_none(obj):
    """ Recursively replaces NaN values with None in nested data structures. """
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(elem) for elem in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    # Handle potential numpy int/float types explicitly
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        # Check for NaN again after potential type casting
        if np.isnan(obj): return None
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
         # Check for NaN in arrays before converting to list
        if np.issubdtype(obj.dtype, np.number): # Only check numeric arrays
             # Use np.nan_to_num or similar if None is not desired in numeric arrays
             # Here we replace NaN with None before converting to list
             obj = np.where(np.isnan(obj), None, obj)
        return replace_nan_with_none(obj.tolist()) # Convert arrays to lists
    return obj

# --- Main Analysis Function ---
def analyze_segments_cross_section():
    """
    Runs the main processing pipeline and calculates GMM/statistical properties
    at cross-sections (DP boundaries and optionally event averages).
    Saves results to YAML. Also runs backup global GMM if configured.
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

    try: # Wrap the whole process in try/except to ensure return on error

        # --- Steps 1-7: Load, Find Events, Align, Tube, Downsample, Segment, Map Boundaries ---
        # --- 1. Load Data ---
        print("--- 1. Loading Data ---")
        original_trajs_df_list, loaded_files = load_selected_data(
            config.PARENT_FOLDER_PATH, load_all=config.LOAD_ALL_FILES,
            file_list=config.FILE_LIST, columns_to_load=config.ALL_LOAD_COLS
        )
        if not original_trajs_df_list:
            print("Exiting: No trajectories loaded.")
            return None # Return None on failure
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
            return None # Return None on failure

        valid_indices = [i for i, df in enumerate(aligned_trajs_df_list) if not df.empty]
        valid_aligned_trajs_df = [aligned_trajs_df_list[i] for i in valid_indices]
        valid_mapped_events_list = [mapped_events_list[i] for i in valid_indices] # Keep this for return
        if not valid_aligned_trajs_df:
            print("Exiting: Alignment produced no valid trajectories.")
            return None # Return None on failure
        n_valid_trajectories = len(valid_aligned_trajs_df)
        print(f"Produced {n_valid_trajectories} valid aligned trajectories.")

        # --- 4. Determine Original Timesteps ---
        # We need n_timesteps_orig for mapping boundaries and potentially clipping indices
        n_timesteps_orig = len(valid_aligned_trajs_df[0]) if valid_aligned_trajs_df else 0
        if n_timesteps_orig <= 1:
             print("Warning: Not enough timesteps in aligned data.")
             # If analysis depends heavily on multiple timesteps, might need to exit or handle differently
        else:
            print(f"  Determined original timesteps: {n_timesteps_orig}")


        # --- 5. Downsample for DP ---
        print("\n--- 5. Downsampling Data for DP (if needed) ---")
        min_vals_dp_basis, max_vals_dp_basis, _ = calculate_tube(
             valid_aligned_trajs_df, config.SEGMENTATION_FEATURE_COLS
        )
        if min_vals_dp_basis is None:
            print("Exiting: Failed to calculate tube for DP basis features.")
            return None # Return None on failure

        min_vals_dp, max_vals_dp, n_timesteps_dp, stride = downsample_tube_data(
            min_vals_dp_basis, max_vals_dp_basis, config.MAX_DP_LENGTH
        )

        # --- 6. Find Optimal Segmentation ---
        print("\n--- 6. Finding Optimal Segmentation (DP) ---")
        segment_indices_dp, optimal_num_segments, _, raw_costs_per_segment_count = find_optimal_segmentation(
            min_vals_dp, max_vals_dp, n_timesteps_dp, config.MAX_SEGMENTS, config.LAMBDA_PENALTY,
            config.SEGMENTATION_FEATURE_COLS, config.POS_COLS, config.ROTATION_WEIGHT,
            valid_mapped_events_list, stride # Pass the mapped events corresponding to valid trajectories
        )
        if optimal_num_segments == -1 or segment_indices_dp is None:
            print("Exiting: Segmentation failed.")
            return None # Return None on failure

        # --- 7. Map Boundaries to Original Scale ---
        print("\n--- 7. Mapping Segment Boundaries ---")
        optimal_segment_indices_orig = map_boundaries_to_original( # Keep this for return
            segment_indices_dp, stride, n_timesteps_orig
        )
        print(f"Optimal Segmentation Boundaries (Original Time Scale): {optimal_segment_indices_orig}")

        # --- 8. Data Preprocessing for Analysis ---
        print("\n--- 8. Preprocessing Aligned Data for Analysis ---")
        processed_aligned_trajs = [] # Keep this for return
        temp_processed_trajs = [] # Use a temporary list for iteration safety
        for i, traj_df in enumerate(valid_aligned_trajs_df):
            print(f"  Preprocessing trajectory {i}...")
            traj_df_copy = traj_df.copy() # Work on a copy to avoid modifying original list elements directly
            # Add Quaternion Column
            quats_wxyz = convert_to_quaternions(traj_df_copy)
            if quats_wxyz is None:
                print(f"  Error: Failed to convert rotations to quaternions for trajectory {i}. Skipping trajectory.")
                continue # Skip this trajectory

            traj_df_copy['quat_wxyz'] = list(quats_wxyz)

            # Add Log Map Columns
            log_map_vectors = convert_to_log_map(quats_wxyz)
            if log_map_vectors is None:
                 print(f"  Error: Failed to convert quaternions to log map for trajectory {i}. Skipping trajectory.")
                 continue

            for j, col_name in enumerate(config.GMM_ORIENT_LOG_MAP_COLS):
                traj_df_copy[col_name] = log_map_vectors[:, j]

            # Add normalized time column (if needed for backup GMM)
            if config.RUN_BACKUP_GLOBAL_GMM:
                 # Ensure length matches for linspace
                 if len(traj_df_copy) > 0:
                    traj_df_copy['time_normalized'] = np.linspace(0, 1, len(traj_df_copy))
                 else:
                     traj_df_copy['time_normalized'] = [] # Assign empty if df is empty

            temp_processed_trajs.append(traj_df_copy) # Add the processed copy

        processed_aligned_trajs = temp_processed_trajs # Assign the fully processed list

        if not processed_aligned_trajs:
            print("Exiting: No trajectories remaining after preprocessing (orientation conversion failed).")
            return None # Return None on failure
        n_processed_trajectories = len(processed_aligned_trajs)
        print(f"Finished preprocessing {n_processed_trajectories} trajectories.")

        # --- Calculate Tube for Plotting (Moved Here) ---
        print("\n--- Calculating Tube (for Visualization Features) ---")
        if processed_aligned_trajs and config.PLOT_FEATURE_COLS:
             min_vals_plot_tube, max_vals_plot_tube, _ = calculate_tube(
                 processed_aligned_trajs, config.PLOT_FEATURE_COLS # Use processed data
             )
             if min_vals_plot_tube is None:
                  print("  Warning: Failed to calculate tube for plotting features.")
             else:
                  print(f"  Tube for plotting calculated using {len(config.PLOT_FEATURE_COLS)} features.")
                  # --- Debug Print for Plot Tube ---
                  if min_vals_plot_tube is not None:
                      print(f"  DEBUG: Plot tube calculated. Shape: {min_vals_plot_tube.shape}. Expected features: {len(config.PLOT_FEATURE_COLS)}")
                      if min_vals_plot_tube.shape[1] != len(config.PLOT_FEATURE_COLS):
                           print(f"  DEBUG: WARNING - Plot tube dimension mismatch!")
                  else:
                      print(f"  DEBUG: Plot tube is None.")
                  # --- End Debug Print ---
        else:
             print("  Skipping tube calculation for plotting (no processed trajectories or no plot features defined).")
             min_vals_plot_tube, max_vals_plot_tube = None, None
        # --- End Moved Tube Calculation ---


        # --- 9. Identify Cross-Section Times ---
        print("\n--- 9. Identifying Cross-Section Times ---")
        cross_section_times = defaultdict(lambda: {'type': None, 'involved_events': set()})
        if n_timesteps_orig > 0: # Check if timesteps exist
            # Add DP boundary times
            for boundary_idx in optimal_segment_indices_orig:
                # Use np.clip for safer clamping
                clamped_idx = np.clip(boundary_idx, 0, n_timesteps_orig - 1)
                cross_section_times[clamped_idx]['type'] = 'boundary'
                print(f"  Added boundary cross-section at index: {clamped_idx}")

            # Add event average times (optional)
            if config.ANALYZE_EVENT_CROSS_SECTIONS:
                print("  Analyzing event-based cross-sections...")
                for i in range(len(optimal_segment_indices_orig) - 1):
                    seg_start = optimal_segment_indices_orig[i]
                    seg_end = optimal_segment_indices_orig[i+1] - 1
                    if seg_start > seg_end: continue

                    segment_event_indices = []
                    segment_event_types = set()

                    # Collect relevant event indices within this segment from all trajectories
                    # Iterate through indices corresponding to processed_aligned_trajs
                    for proc_idx in range(n_processed_trajectories):
                        # Find the original index of this processed trajectory
                        original_traj_index = valid_indices[proc_idx] # Map back to original index
                        mapped_events = mapped_events_list[original_traj_index] # Get events using original index

                        for event_type in config.EVENTS_FOR_CROSS_SECTIONS:
                            if event_type in mapped_events:
                                for event_idx in mapped_events[event_type]:
                                    if seg_start <= event_idx <= seg_end:
                                        segment_event_indices.append(event_idx)
                                        segment_event_types.add(event_type)

                    if segment_event_indices:
                        t_ave_event = np.mean(segment_event_indices)
                        t_ave_event_idx = t_ave_event # Keep as float
                        # Use np.clip for safer clamping
                        t_ave_event_idx = np.clip(t_ave_event_idx, 0, n_timesteps_orig - 1)

                        # Check if type is None before checking value
                        existing_type = cross_section_times[t_ave_event_idx]['type']
                        # Add event CS only if the exact time doesn't already exist as a boundary
                        # Using np.isclose for float comparison might be needed if t_ave_event_idx is float
                        is_boundary = False
                        for boundary_t in optimal_segment_indices_orig:
                             # Convert boundary_t to float for comparison
                             if np.isclose(t_ave_event_idx, float(boundary_t)):
                                  is_boundary = True
                                  break

                        if not is_boundary:
                             cross_section_times[t_ave_event_idx]['type'] = 'event'
                             cross_section_times[t_ave_event_idx]['involved_events'].update(segment_event_types)
                             print(f"  Added event cross-section at average index: {t_ave_event_idx:.2f} (Segment {i}, Events: {segment_event_types})")
        else:
            print("  Skipping cross-section identification as n_timesteps_orig is 0.")


        # Sort the cross-section times
        sorted_cross_section_times = sorted(cross_section_times.keys())
        print(f"  Total unique cross-section times identified: {len(sorted_cross_section_times)}")

        # --- 10. Analyze Each Cross-Section ---
        print("\n--- 10. Analyzing Cross-Sections ---")
        cross_section_results = []
        gmm_state_cols = config.GMM_STATE_COLS

        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        for t_cs in sorted_cross_section_times:
            print(f"  Analyzing cross-section at time index: {t_cs:.2f}...")
            snapshot_states = []
            snapshot_quats_wxyz = []
            snapshot_positions = []

            for traj_df in processed_aligned_trajs:
                state_vec, quat_wxyz = get_state_at_time(traj_df, float(t_cs), gmm_state_cols, 'quat_wxyz') # Ensure t_cs is float
                if state_vec is not None and quat_wxyz is not None:
                    snapshot_states.append(state_vec)
                    snapshot_quats_wxyz.append(quat_wxyz)
                    snapshot_positions.append(state_vec[:len(config.GMM_POS_COLS)])
                # else: # Reduce verbosity
                #      print(f"    Warning: Could not get state/quat snapshot for a trajectory at time {t_cs:.2f}.")

            # --- Dynamic GMM component adjustment ---
            n_snapshots = len(snapshot_states)
            # Determine actual number of components to use
            # Use at most the configured number, but no more than available snapshots
            # Also ensure at least 1 component is used if snapshots exist
            n_components_actual = min(config.GMM_N_COMPONENTS, n_snapshots) if n_snapshots > 0 else 0
            gmm_params = None # Initialize gmm_params

            # Proceed only if we have enough data for at least one component
            if n_components_actual >= 1:
                snapshot_states_np = np.array(snapshot_states)
                try:
                    # Use n_components_actual instead of config.GMM_N_COMPONENTS
                    gmm = GaussianMixture(n_components=n_components_actual, covariance_type='full',
                                          random_state=0, n_init=5, max_iter=200, tol=1e-3)
                    gmm.fit(snapshot_states_np)
                    if gmm.converged_:
                         gmm_params = {'weights': gmm.weights_.tolist(), 'means': gmm.means_.tolist(), 'covariances': gmm.covariances_.tolist(), 'n_components_used': n_components_actual}
                         print(f"      GMM trained successfully with {n_components_actual} components.")
                    else:
                         print(f"      Warning: GMM with {n_components_actual} components did not converge at time {t_cs:.2f}.")
                         gmm_params = {'weights': gmm.weights_.tolist(), 'means': gmm.means_.tolist(), 'covariances': gmm.covariances_.tolist(), 'converged': False, 'n_components_used': n_components_actual}
                except Exception as e:
                    print(f"    Error training GMM with {n_components_actual} components at time {t_cs:.2f}: {e}")
                    gmm_params = None # Set to None on error
            elif n_snapshots > 0: # Snapshots exist, but not enough for even 1 component
                 print(f"    Warning: Only {n_snapshots} snapshots available at time {t_cs:.2f}. Cannot train GMM.")
                 # gmm_params remains None
            else: # n_snapshots == 0
                 print(f"    Warning: No valid snapshots found at time {t_cs:.2f}. Skipping analysis.")
                 # gmm_params remains None
                 continue # Skip to next cross-section if no snapshots

            # --- End Dynamic GMM component adjustment ---


            pos_min_bounds, pos_max_bounds = None, None
            if snapshot_positions:
                 snapshot_positions_np = np.array(snapshot_positions)
                 pos_min_bounds = np.min(snapshot_positions_np, axis=0).tolist()
                 pos_max_bounds = np.max(snapshot_positions_np, axis=0).tolist()

            orient_mean_wxyz, orient_geodesic_max_dist = None, None
            if snapshot_quats_wxyz:
                snapshot_quats_np = np.array(snapshot_quats_wxyz)
                orient_mean_wxyz = calculate_geometric_mean_quaternion(snapshot_quats_np)
                if orient_mean_wxyz is not None:
                    distances = [calculate_geodesic_distance(orient_mean_wxyz, q) for q in snapshot_quats_np]
                    valid_distances = [d for d in distances if not np.isnan(d)]
                    if valid_distances: orient_geodesic_max_dist = float(np.max(valid_distances))
                    orient_mean_wxyz = orient_mean_wxyz.tolist() # Convert mean to list

            cs_info = cross_section_times[t_cs]
            cross_section_results.append({
                "time_index": float(t_cs), "type": cs_info['type'],
                "involved_events": sorted(list(cs_info['involved_events'])),
                "num_snapshots": len(snapshot_states), "segment_gmm_params": gmm_params, # Use the potentially None gmm_params
                "pos_min_bounds": pos_min_bounds, "pos_max_bounds": pos_max_bounds,
                "orient_mean_quaternion": orient_mean_wxyz,
                "orient_geodesic_max_distance": orient_geodesic_max_dist
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

        # --- 12. Backup Global GMM Analysis (Optional) ---
        if config.RUN_BACKUP_GLOBAL_GMM:
            print("\n--- 12. Running Backup Global GMM Analysis ---")
            all_global_data = []
            required_cols = config.GMM_STATE_COLS + ['time_normalized']
            for traj_df in processed_aligned_trajs:
                if all(col in traj_df.columns for col in required_cols):
                    data_subset = traj_df[required_cols].values
                    all_global_data.append(data_subset)
                # else: # Less verbose
                #     print(f"  Warning: Trajectory missing required columns for global GMM. Skipping.")

            if all_global_data:
                global_gmm_input = np.vstack(all_global_data)
                print(f"  Training global GMM on {global_gmm_input.shape[0]} total data points...")
                try:
                    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                    global_gmm = GaussianMixture(n_components=config.BACKUP_GMM_N_COMPONENTS,
                                                 covariance_type='full', random_state=0, n_init=3, max_iter=150, tol=1e-3)
                    global_gmm.fit(global_gmm_input)
                    warnings.filterwarnings("default", category=ConvergenceWarning, module="sklearn")
                    global_gmm_params = {
                        'weights': global_gmm.weights_.tolist(), 'means': global_gmm.means_.tolist(),
                        'covariances': global_gmm.covariances_.tolist(), 'converged': global_gmm.converged_,
                        'n_components': config.BACKUP_GMM_N_COMPONENTS, 'data_shape': global_gmm_input.shape,
                        'features': required_cols
                    }
                    print(f"    Global GMM trained (Converged: {global_gmm.converged_}).") # BIC: {global_gmm.bic(global_gmm_input):.2f}")
                    backup_output_path_str = config.BACKUP_GMM_OUTPUT_PATH
                    backup_output_path = Path(backup_output_path_str)
                    backup_output_path.parent.mkdir(parents=True, exist_ok=True)
                    cleaned_global_params = replace_nan_with_none(global_gmm_params)
                    with open(backup_output_path, 'w') as f:
                        yaml.dump(cleaned_global_params, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
                    print(f"  Backup global GMM parameters saved successfully to: {backup_output_path}")
                except Exception as e:
                    print(f"  Error during backup global GMM training or saving: {e}")
            else:
                print("  No data available for backup global GMM training.")

    except Exception as e:
        print(f"\n--- FATAL ERROR during analysis pipeline ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        # Ensure we return None or partial results if an error occurs mid-way
        return None # Indicate failure

    finally:
        # --- Prepare Return Dictionary ---
        # This block executes even if errors occurred in the try block,
        # returning whatever data was successfully computed before the error.
        results_for_main = {
            "processed_aligned_trajs": processed_aligned_trajs,
            "min_vals_plot_tube": min_vals_plot_tube, # Tube for plot features
            "max_vals_plot_tube": max_vals_plot_tube, # Tube for plot features
            "optimal_segment_indices_orig": optimal_segment_indices_orig,
            "valid_mapped_events_list": valid_mapped_events_list, # Mapped events for valid trajs
            "raw_costs_per_segment_count": raw_costs_per_segment_count,
            "optimal_num_segments": optimal_num_segments
        }
        overall_duration = time.time() - overall_start_time
        print(f"\n--- Analysis Function Finished in {overall_duration:.2f} seconds ---")
        return results_for_main # Return the dictionary

if __name__ == "__main__":
    analysis_output = analyze_segments_cross_section() # Run the analysis
    if analysis_output:
        print("\nAnalysis completed. Output dictionary contains:")
        for key, value in analysis_output.items():
             if isinstance(value, (list, tuple)) and value and isinstance(value[0], (pd.DataFrame, np.ndarray)): # Check if list not empty
                 print(f" - {key}: List of {len(value)} DataFrames/Arrays")
             elif isinstance(value, (np.ndarray)):
                  print(f" - {key}: Numpy array with shape {value.shape}")
             elif isinstance(value, (list, tuple)):
                  print(f" - {key}: List/Tuple with {len(value)} elements")
             else:
                  print(f" - {key}: {type(value)}")
    else:
        print("\nAnalysis failed or returned no results.")