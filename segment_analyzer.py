#!/usr/bin/env python3
import time
import sys
import os
import yaml # Import YAML library
import numpy as np
import pandas as pd
from pathlib import Path
# Removed SciPy import as orientation math helpers are removed

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
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary modules.")
    print(f"Ensure all .py files (config, data_loading, alignment, segmentation) are in the same directory: {script_dir}")
    print(f"Error details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during imports: {e}")
    sys.exit(1)

# --- Helper Functions for Statistics ---

def calculate_trajectory_stats(aligned_trajs_list, feature_cols, segment_start, segment_end):
    """
    Calculates mean and variance for specified features across all trajectories
    within a given time segment. Ensures float/None output.
    """
    segment_data = []
    # Determine columns actually present in the input dataframes for the requested features
    if not aligned_trajs_list: return {col: {'mean': None, 'variance': None} for col in feature_cols}
    cols_present_in_data = [col for col in feature_cols if col in aligned_trajs_list[0].columns]
    if not cols_present_in_data: return {col: {'mean': None, 'variance': None} for col in feature_cols}


    for traj_df in aligned_trajs_list:
        if not traj_df.empty and segment_start < len(traj_df):
            end_slice = min(segment_end + 1, len(traj_df))
            # Select only present columns before slicing
            segment_part = traj_df.iloc[segment_start:end_slice][cols_present_in_data]
            if not segment_part.empty:
                segment_data.append(segment_part.values) # Append only values of present columns

    if not segment_data:
        # Return None if no data points found in this segment
        nan_stats = {'mean': None, 'variance': None}
        return {col: nan_stats for col in feature_cols}

    try:
        # vstack should work now as we only selected present columns consistently
        all_segment_points = np.vstack(segment_data)
    except ValueError as e:
         print(f"  Warning: Could not vstack segment data between {segment_start}-{segment_end}. Error: {e}")
         nan_stats = {'mean': None, 'variance': None}
         return {col: nan_stats for col in feature_cols}

    # Calculate stats only if data exists
    if all_segment_points.size == 0:
         nan_stats = {'mean': None, 'variance': None}
         return {col: nan_stats for col in feature_cols}


    with np.errstate(invalid='ignore', divide='ignore'): # Ignore warnings for std dev of single point etc.
        means = np.nanmean(all_segment_points, axis=0)
        variances = np.nanvar(all_segment_points, axis=0)

    stats_dict = {}
    for i, col_name in enumerate(cols_present_in_data): # Iterate through columns actually processed
         if i < len(means):
              mean_val = float(means[i]) if not np.isnan(means[i]) else None
              # Variance is 0 for a single point, handle potential NaN if input was all NaN
              var_val = float(variances[i]) if not np.isnan(variances[i]) else None
              stats_dict[col_name] = {'mean': mean_val, 'variance': var_val}
         else:
              stats_dict[col_name] = {'mean': None, 'variance': None}

    # Ensure all requested columns are in the output dict, even if not found in data
    for col_name in feature_cols:
        if col_name not in stats_dict:
             stats_dict[col_name] = {'mean': None, 'variance': None}

    return stats_dict


def calculate_event_stats(mapped_events_list, segment_start, segment_end):
    """
    Calculates count and index statistics for events within a segment. (Unchanged)
    """
    segment_events = {'wp_saved': [], 'state_change': [], 'gripper_change': [], 'start': [], 'end': []}
    event_types = list(segment_events.keys())

    for events_dict in mapped_events_list:
        for event_type in event_types:
            if event_type in events_dict:
                for idx in events_dict[event_type]:
                    if segment_start <= idx <= segment_end:
                        segment_events[event_type].append(idx)

    stats_dict = {}
    for event_type, indices in segment_events.items():
        count = len(indices)
        if count > 0:
            indices_arr = np.array(indices)
            min_idx = int(np.min(indices_arr))
            max_idx = int(np.max(indices_arr))
            mean_idx = float(np.mean(indices_arr))
            var_idx = float(np.var(indices_arr)) if count > 1 else 0.0
        else:
            min_idx, max_idx, mean_idx, var_idx = None, None, None, None

        stats_dict[event_type] = {
            'count': count, 'min_idx': min_idx, 'max_idx': max_idx,
            'mean_idx': mean_idx, 'var_idx': var_idx
        }
    return stats_dict

# --- REMOVED Orientation Helper Functions ---
# get_orientations_in_segment, calculate_central_orientation,
# angle_between_matrices, calculate_orientation_stats_for_plot
# are no longer needed for this output structure.

# --- Main Analysis Function ---
def analyze_segments():
    """
    Runs the main processing pipeline and calculates segment statistics
    (tube min/max, trajectory mean/variance, events) saving to YAML.
    """
    overall_start_time = time.time()

    # --- Steps 1-7: Load, Find Events, Align, Tube, Downsample, Segment, Map Boundaries ---
    # (Pipeline execution - code omitted for brevity, assumes success)
    # --- 1. Load Data ---
    print("--- 1. Loading Data ---")
    original_trajs_df, loaded_files = load_selected_data(
        config.PARENT_FOLDER_PATH, load_all=config.LOAD_ALL_FILES,
        file_list=config.FILE_LIST, columns_to_load=config.ALL_LOAD_COLS
    )
    if not original_trajs_df: sys.exit("Exiting: No trajectories loaded.")
    print(f"Loaded {len(original_trajs_df)} trajectories.")
    # --- 2. Find Events ---
    print("\n--- 2. Finding Events in Original Trajectories ---")
    original_events_list = []
    for i, df in enumerate(original_trajs_df):
         if df.empty: events = {'start': [], 'end': [], 'state_change': [], 'wp_saved': [], 'gripper_change': []}
         else: events = find_events(df, config.EVENT_COLUMNS)
         original_events_list.append(events)
    # --- 3. Align Trajectories ---
    print("\n--- 3. Aligning Trajectories & Mapping Events ---")
    aligned_trajs_df, mapped_events_list, ref_idx = align_trajectories(
        original_trajs_df, original_events_list, config.ALIGNMENT_TYPE,
        config.ALIGNMENT_BASE_FEATURE_COLS, config.ALL_LOAD_COLS, config.NORMALIZE_FOR_COMBINED
    )
    if ref_idx == -1 or not aligned_trajs_df: sys.exit("Exiting: Alignment failed.")
    valid_indices = [i for i, df in enumerate(aligned_trajs_df) if not df.empty]
    valid_aligned_trajs_df = [aligned_trajs_df[i] for i in valid_indices]
    valid_mapped_events_list = [mapped_events_list[i] for i in valid_indices]
    if not valid_aligned_trajs_df: sys.exit("Exiting: Alignment produced no valid trajectories.")
    print(f"Produced {len(valid_aligned_trajs_df)} valid aligned trajectories.")
    # --- 4. Calculate Tube ---
    print("\n--- 4. Calculating Tube ---")
    # Use all features specified in segmentation config for tube calculation
    tube_features = config.SEGMENTATION_FEATURE_COLS
    min_vals_orig, max_vals_orig, n_timesteps_orig = calculate_tube(
        valid_aligned_trajs_df, tube_features
    )
    if min_vals_orig is None or n_timesteps_orig <= 1: sys.exit("Exiting: Failed to calculate tube.")
    # --- 5. Downsample for DP ---
    print("\n--- 5. Downsampling Data for DP (if needed) ---")
    min_vals_dp, max_vals_dp, n_timesteps_dp, stride = downsample_tube_data(
        min_vals_orig, max_vals_orig, config.MAX_DP_LENGTH
    )
    # --- 6. Find Optimal Segmentation ---
    print("\n--- 6. Finding Optimal Segmentation (DP) ---")
    segment_indices_dp, optimal_num_segments, _, _ = find_optimal_segmentation(
        min_vals_dp, max_vals_dp, n_timesteps_dp, config.MAX_SEGMENTS, config.LAMBDA_PENALTY,
        config.SEGMENTATION_FEATURE_COLS, config.POS_COLS, config.ROTATION_WEIGHT,
        valid_mapped_events_list, stride
    )
    if optimal_num_segments == -1 or segment_indices_dp is None: sys.exit("Exiting: Segmentation failed.")
    # --- 7. Map Boundaries to Original Scale ---
    print("\n--- 7. Mapping Segment Boundaries ---")
    optimal_segment_indices_orig = map_boundaries_to_original(
        segment_indices_dp, stride, n_timesteps_orig
    )
    print(f"Optimal Segmentation Boundaries (Original Time Scale): {optimal_segment_indices_orig}")
    # --- End Steps 1-7 ---


    # --- 8. Analyze Segments ---
    print("\n--- 8. Analyzing Segments ---")
    all_segment_stats = []
    # Define features for which to calculate stats (matching the prompt)
    pos_features = ['tx', 'ty', 'tz']
    rot_features = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
    features_for_stats = pos_features + rot_features # Combine pos and rot features

    for i in range(len(optimal_segment_indices_orig) - 1):
        seg_start_orig = optimal_segment_indices_orig[i]
        seg_end_orig = optimal_segment_indices_orig[i+1] - 1 # Inclusive end index

        if seg_start_orig > seg_end_orig:
            print(f"  Warning: Skipping segment {i} due to invalid boundaries ({seg_start_orig} > {seg_end_orig}).")
            continue

        print(f"  Analyzing Segment {i} (Time: {seg_start_orig} to {seg_end_orig})...")

        # --- Initialize segment_info with the structure from the prompt ---
        segment_info = {
            "segment_index": i,
            # tube_stats will contain min/max for features_for_stats
            "tube_stats": {},
            # trajectory_stats will contain mean/variance for features_for_stats
            "trajectory_stats": {},
            # Keep event stats as it provides useful context
            "event_stats": {}
        }

        # --- Calculate Tube Stats (Min/Max) ---
        # Use the features defined for stats (pos + rot)
        if seg_start_orig < n_timesteps_orig:
             end_slice = min(seg_end_orig + 1, n_timesteps_orig)
             # Need to get the correct column indices from the tube data
             tube_min_segment = min_vals_orig[seg_start_orig:end_slice]
             tube_max_segment = max_vals_orig[seg_start_orig:end_slice]
             # Map feature names to column indices in the tube data (based on tube_features)
             tube_col_map = {name: idx for idx, name in enumerate(tube_features)}

             for feature_name in features_for_stats:
                  col_idx = tube_col_map.get(feature_name)
                  if col_idx is not None and col_idx < tube_min_segment.shape[1]:
                      min_val = float(np.min(tube_min_segment[:, col_idx])) if tube_min_segment.size > 0 else None
                      max_val = float(np.max(tube_max_segment[:, col_idx])) if tube_max_segment.size > 0 else None
                      segment_info["tube_stats"][feature_name] = {"min": min_val, "max": max_val}
                  else:
                      # If feature wasn't in tube calculation, stats are None
                      segment_info["tube_stats"][feature_name] = {"min": None, "max": None}
        else:
             print(f"  Warning: Segment {i} start index {seg_start_orig} out of bounds for tube data.")
             for feature_name in features_for_stats:
                 segment_info["tube_stats"][feature_name] = {"min": None, "max": None}

        # --- Calculate Trajectory Stats (Mean/Variance) ---
        # Use the features defined for stats (pos + rot)
        traj_stats = calculate_trajectory_stats(
            valid_aligned_trajs_df, features_for_stats, seg_start_orig, seg_end_orig
        )
        # Ensure the structure matches the prompt (key: {mean: val, variance: val})
        segment_info["trajectory_stats"] = traj_stats # Function already returns this structure

        # --- Calculate Event Stats ---
        segment_info["event_stats"] = calculate_event_stats(
            valid_mapped_events_list, seg_start_orig, seg_end_orig
        )

        all_segment_stats.append(segment_info)

    # --- 9. Save Results to YAML ---
    print("\n--- 9. Saving Segment Statistics (YAML) ---")
    output_path_str = config.SEGMENT_STATS_OUTPUT_PATH
    output_path = Path(output_path_str)
    if not (output_path.suffix.lower() in ['.yaml', '.yml']): # Case-insensitive check
        print(f"Warning: Output path '{output_path_str}' does not end with .yaml or .yml.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        def replace_nan_with_none(obj):
            if isinstance(obj, dict): return {k: replace_nan_with_none(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [replace_nan_with_none(elem) for elem in obj]
            elif isinstance(obj, float) and np.isnan(obj): return None
            # Handle potential numpy int/float types explicitly
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
            elif isinstance(obj, (np.ndarray,)): return replace_nan_with_none(obj.tolist()) # Convert arrays to lists
            return obj
        cleaned_stats = replace_nan_with_none(all_segment_stats)
        with open(output_path, 'w') as f:
            yaml.dump(cleaned_stats, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Segment statistics saved successfully to: {output_path}")
    except ImportError:
         print("Error: PyYAML library not found. Please install it: pip install pyyaml")
    except IOError as e:
        print(f"Error: Could not write statistics file to {output_path}. Error: {e}")
    except Exception as e:
         print(f"Error: Could not dump statistics to YAML. Error: {e}")

    overall_duration = time.time() - overall_start_time
    print(f"\n--- Analysis Finished in {overall_duration:.2f} seconds ---")

if __name__ == "__main__":
    # Removed SciPy check as it's no longer directly needed here
    analyze_segments()

