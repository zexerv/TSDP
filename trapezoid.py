# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw # Import fastdtw
import seaborn as sns # For setting style
from sklearn.preprocessing import MinMaxScaler # For normalization
import sys # To manage recursion depth for DP if needed
import time # To measure execution time

# Increase recursion depth limit if necessary for deep DP tables
# sys.setrecursionlimit(3000)

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
# !!! PLEASE UPDATE THIS PATH !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/linear_switch' # <--- CHANGE THIS

# Option 1: Load all CSV files in the folder
LOAD_ALL_FILES = True
# Option 2: Specify a list of filenames (if LOAD_ALL_FILES is False)
# FILE_LIST = ['1.csv', '3.csv', '5.csv']
FILE_LIST = [] # Ignored if LOAD_ALL_FILES is True

# Alignment Type: 'DTW', 'DDTW', 'Combined'
ALIGNMENT_TYPE = 'Combined'
# Base features for alignment (position)
ALIGNMENT_BASE_FEATURE_COLS = ['tx', 'ty', 'tz', 'r11','r12','r13','r21','r22','r23','r31','r32','r33']
# Apply normalization before calculating combined DTW distance?
NORMALIZE_FOR_COMBINED = True

# --- Segmentation Configuration ---
# Features to use for segmentation (usually the same as alignment base)
SEGMENTATION_FEATURE_COLS = ['tx', 'ty', 'tz', 'r11','r12','r13','r21','r22','r23','r31','r32','r33']
# Maximum number of segments to consider
MAX_SEGMENTS = 30
# Penalty coefficient for adding a segment (tune this value)
# Higher lambda favors fewer segments. Start with a value and adjust.
LAMBDA_PENALTY = 0.1 # Example value, needs tuning
# --- NEW: Weight for rotation features relative to position features (weight=1.0) ---
ROTATION_WEIGHT = 0.01 # Example: Rotation contributes 10% as much as position to the cost
# --- NEW: Downsampling Configuration ---
# Maximum length of the time series to use for DP calculation
# If the original length exceeds this, the data will be downsampled.
MAX_DP_LENGTH = 300

print(f"--- Configuration ---")
print(f"Data Folder: {PARENT_FOLDER_PATH}")
print(f"Alignment Type: {ALIGNMENT_TYPE}")
print(f"Base features for Alignment: {ALIGNMENT_BASE_FEATURE_COLS}")
if ALIGNMENT_TYPE == 'Combined':
    print(f"Normalization before combined DTW: {NORMALIZE_FOR_COMBINED}")
print(f"Features for Segmentation: {SEGMENTATION_FEATURE_COLS}")
print(f"Max Segments: {MAX_SEGMENTS}")
print(f"Segment Penalty (Lambda): {LAMBDA_PENALTY}")
print(f"Max DP Length (Downsampling Threshold): {MAX_DP_LENGTH}") # Display new config
print("-" * 20)

# --- Data Loading (Copied from your script) ---
def load_selected_data(parent_folder, load_all=True, file_list=None, feature_columns=None):
    """Loads specified trajectories and features from CSV files."""
    all_trajectories = []
    loaded_filenames = []
    POS_COLS = ['tx', 'ty', 'tz']
    ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
    required_columns = list(set(POS_COLS + ROT_COLS + (feature_columns if feature_columns else [])))

    if feature_columns is None: feature_columns = ['tx', 'ty', 'tz']

    if load_all:
        try:
            all_files_in_dir = [f for f in os.listdir(parent_folder) if os.path.isfile(os.path.join(parent_folder, f))]
            filenames = [f for f in all_files_in_dir if f.endswith('.csv')]
            try: filenames.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError: filenames.sort()
            print(f"Found {len(filenames)} CSV files to load: {filenames}")
            if not filenames: print(f"Warning: No CSV files found in {parent_folder}"); return [], []
        except FileNotFoundError: print(f"Error: Parent folder not found - {parent_folder}"); return [], []
    elif file_list: filenames = file_list; print(f"Loading specified files: {filenames}")
    else: print("Error: No files specified to load."); return [], []

    for filename in filenames:
        file_path = os.path.join(parent_folder, filename)
        try:
            df = pd.read_csv(file_path)
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols: raise KeyError(f"Missing columns: {missing_cols}")
            all_trajectories.append(df[required_columns])
            loaded_filenames.append(filename)
        except FileNotFoundError: print(f"Warning: File not found - {file_path}")
        except KeyError as e: print(f"Error: Missing expected column in {file_path}: {e}")
        except Exception as e: print(f"Error loading or processing {file_path}: {e}")

    if not all_trajectories: print("Warning: No valid trajectory data loaded.")
    return all_trajectories, loaded_filenames

# --- Alignment (DTW/DDTW/Combined) (Copied from your script) ---
def calculate_derivative(trajectory):
    """Estimates derivative using simple finite difference."""
    if len(trajectory) < 2: return np.zeros_like(trajectory)
    return np.diff(trajectory, axis=0, prepend=trajectory[0:1,:])

def align_trajectories(trajectories_df_list, alignment_type, alignment_base_feature_cols, normalize_combined=True):
    """Aligns trajectories using DTW, DDTW, or Combined approach."""
    if not trajectories_df_list: return [], -1
    if alignment_type not in ['DTW', 'DDTW', 'Combined']:
        raise ValueError("alignment_type must be 'DTW', 'DDTW', or 'Combined'")

    trajectories_full_np = [df.values for df in trajectories_df_list]
    required_columns = trajectories_df_list[0].columns.tolist()
    trajectories_base_np = [df[alignment_base_feature_cols].values for df in trajectories_df_list]

    lengths = [len(t) for t in trajectories_base_np]
    if not lengths: return [], -1
    reference_index = np.argmax(lengths)
    reference_trajectory_base = trajectories_base_np[reference_index]
    reference_trajectory_full = trajectories_full_np[reference_index]
    ref_len = len(reference_trajectory_base)
    print(f"\nUsing trajectory {reference_index} (length {ref_len}) as reference for {alignment_type}.")

    aligned_trajectories_np = []
    for i, traj_base in enumerate(trajectories_base_np):
        traj_full = trajectories_full_np[i]
        # print(f"  Aligning trajectory {i} (length {len(traj_base)}) to reference...") # Less verbose
        if len(traj_base) == 0:
             # print(f"    Skipping empty trajectory {i}")
             aligned_trajectories_np.append(np.zeros_like(reference_trajectory_full))
             continue
        if i == reference_index:
            aligned_trajectories_np.append(reference_trajectory_full.copy())
            continue

        ref_data_for_dtw, traj_data_for_dtw = None, None
        if alignment_type == 'DTW':
            ref_data_for_dtw = reference_trajectory_base
            traj_data_for_dtw = traj_base
        elif alignment_type == 'DDTW':
            ref_data_for_dtw = calculate_derivative(reference_trajectory_base)
            traj_data_for_dtw = calculate_derivative(traj_base)
            if not np.any(ref_data_for_dtw) and not np.any(traj_data_for_dtw):
                 # print("    Warning: Both derivative sequences zero. Using DTW fallback.")
                 ref_data_for_dtw = reference_trajectory_base
                 traj_data_for_dtw = traj_base
        elif alignment_type == 'Combined':
            ref_deriv = calculate_derivative(reference_trajectory_base)
            traj_deriv = calculate_derivative(traj_base)
            ref_pos = reference_trajectory_base
            traj_pos = traj_base
            if normalize_combined:
                scaler_pos = MinMaxScaler()
                scaler_deriv = MinMaxScaler()
                combined_pos = np.vstack((ref_pos, traj_pos))
                combined_deriv = np.vstack((ref_deriv, traj_deriv))
                if np.ptp(combined_pos, axis=0).min() > 1e-6:
                     scaler_pos.fit(combined_pos)
                     ref_pos_norm = scaler_pos.transform(ref_pos)
                     traj_pos_norm = scaler_pos.transform(traj_pos)
                else:
                     # print("    Warning: Low variance in position data, skipping pos normalization.")
                     ref_pos_norm = ref_pos
                     traj_pos_norm = traj_pos
                if np.ptp(combined_deriv, axis=0).min() > 1e-6:
                     scaler_deriv.fit(combined_deriv)
                     ref_deriv_norm = scaler_deriv.transform(ref_deriv)
                     traj_deriv_norm = scaler_deriv.transform(traj_deriv)
                else:
                     # print("    Warning: Low variance in derivative data, skipping deriv normalization.")
                     ref_deriv_norm = ref_deriv
                     traj_deriv_norm = traj_deriv
                ref_data_for_dtw = np.hstack((ref_pos_norm, ref_deriv_norm))
                traj_data_for_dtw = np.hstack((traj_pos_norm, traj_deriv_norm))
            else:
                ref_data_for_dtw = np.hstack((ref_pos, ref_deriv))
                traj_data_for_dtw = np.hstack((traj_pos, traj_deriv))

        distance, path = fastdtw(ref_data_for_dtw, traj_data_for_dtw, dist=euclidean)
        # print(f"    Alignment distance ({alignment_type}): {distance:.4f}") # Less verbose

        path_array = np.array(path)
        warped_traj_full = np.zeros_like(reference_trajectory_full)
        ref_indices_in_path = path_array[:, 0]; traj_indices_in_path = path_array[:, 1]
        for ref_idx in range(ref_len):
            matching_path_indices = np.where(ref_indices_in_path == ref_idx)[0]
            if len(matching_path_indices) > 0:
                traj_idx = traj_indices_in_path[matching_path_indices[-1]]
                traj_idx = min(traj_idx, len(traj_full) - 1)
                warped_traj_full[ref_idx] = traj_full[traj_idx]
            else:
                if ref_idx > 0: warped_traj_full[ref_idx] = warped_traj_full[ref_idx - 1]
                elif len(traj_full) > 0: warped_traj_full[ref_idx] = traj_full[0]
        aligned_trajectories_np.append(warped_traj_full)

    aligned_trajectories_df = [pd.DataFrame(data=arr, columns=required_columns) for arr in aligned_trajectories_np]
    return aligned_trajectories_df, reference_index

# --- Tube Calculation ---
def calculate_tube(aligned_trajectories_df_list, feature_cols):
    """Calculates the min/max envelope (tube) for specified features."""
    if not aligned_trajectories_df_list:
        print("Warning: No aligned trajectories to calculate tube from.")
        return None, None, 0

    # Use length of the first valid trajectory as target length
    target_len = 0
    for df in aligned_trajectories_df_list:
        if not df.empty:
            target_len = len(df)
            break
    if target_len == 0:
        print("Warning: Could not determine target length from aligned trajectories.")
        return None, None, 0

    stacked_data = []
    for i, df in enumerate(aligned_trajectories_df_list):
        if not df.empty and all(c in df.columns for c in feature_cols):
            if len(df) != target_len:
                print(f"Warning: Skipping trajectory {i} with inconsistent length {len(df)} (expected {target_len}).")
                continue
            stacked_data.append(df[feature_cols].values)
        # else: print(f"Warning: Skipping empty or incomplete trajectory {i}")

    if not stacked_data:
        print(f"Warning: No valid data found for features {feature_cols} in aligned trajectories.")
        return None, None, 0

    try:
        trajectory_array = np.stack(stacked_data, axis=0)
    except ValueError as e:
         print(f"Error stacking trajectories: {e}. Check alignment output.")
         return None, None, 0

    min_vals = np.min(trajectory_array, axis=0)
    max_vals = np.max(trajectory_array, axis=0)
    n_timesteps = min_vals.shape[0]

    print(f"\nCalculated tube: {n_timesteps} time steps, {min_vals.shape[1]} dimensions.")
    return min_vals, max_vals, n_timesteps

# --- NEW: Downsampling Function ---
def downsample_tube_data(min_vals, max_vals, max_length):
    """Downsamples tube data if its length exceeds max_length."""
    n_timesteps_orig = min_vals.shape[0]
    if n_timesteps_orig <= max_length:
        print("Data length within limit, no downsampling needed.")
        return min_vals, max_vals, n_timesteps_orig, 1 # Stride is 1

    # Calculate stride needed to get length <= max_length
    stride = math.ceil(n_timesteps_orig / max_length)
    print(f"Downsampling data: Original length={n_timesteps_orig}, Target max={max_length}, Stride={stride}")

    # Apply stride using simple slicing
    min_vals_ds = min_vals[::stride]
    max_vals_ds = max_vals[::stride]
    n_timesteps_ds = min_vals_ds.shape[0]
    print(f"Downsampled length: {n_timesteps_ds}")

    return min_vals_ds, max_vals_ds, n_timesteps_ds, stride




# --- Segmentation Cost Function (Unchanged) ---
def calculate_segment_cost(start_idx, end_idx, min_vals_dim, max_vals_dim, weight=1.0):
    """
    Calculates the cost of approximating a tube segment with linear boundaries.
    Cost is the sum of squared errors between actual tube boundaries and
    linear interpolation of the segment's start/end points.
    """
    if start_idx >= end_idx:
        return 0.0

    segment_len_points = end_idx - start_idx + 1
    if segment_len_points <= 1: return 0.0 # Avoid division by zero later

    # Ensure indices are within the bounds of the potentially downsampled array
    if start_idx < 0 or end_idx >= len(min_vals_dim):
        print(f"Warning: Invalid indices {start_idx}, {end_idx} for array length {len(min_vals_dim)}")
        return np.inf # Return infinite cost for invalid segment

    t = np.arange(segment_len_points)
    actual_min = min_vals_dim[start_idx : end_idx + 1]
    actual_max = max_vals_dim[start_idx : end_idx + 1]

    # Linear interpolation for min boundary
    min_start_val = actual_min[0]
    min_end_val = actual_min[-1]
    min_slope = (min_end_val - min_start_val) / (segment_len_points - 1)
    approx_min = min_start_val + min_slope * t

    # Linear interpolation for max boundary
    max_start_val = actual_max[0]
    max_end_val = actual_max[-1]
    max_slope = (max_end_val - max_start_val) / (segment_len_points - 1)
    approx_max = max_start_val + max_slope * t

    error_min = np.sum((actual_min - approx_min)**2)
    error_max = np.sum((actual_max - approx_max)**2)

    # Return cost normalized by segment length? Optional, could affect lambda tuning.
    # return (error_min + error_max) / segment_len_points
    return weight * (error_min + error_max)


# --- Optimal Segmentation using Dynamic Programming (Operates on potentially downsampled data) ---
def find_optimal_segmentation(min_vals_dp, max_vals_dp, n_timesteps_dp, max_segments, lambda_penalty):
    """
    Finds the optimal segmentation using dynamic programming on the provided (potentially downsampled) data.

    Args:
        min_vals_dp: Min values for DP (potentially downsampled).
        max_vals_dp: Max values for DP (potentially downsampled).
        n_timesteps_dp: Number of time steps for DP.
        max_segments: Max segments allowed.
        lambda_penalty: Penalty per segment.

    Returns:
        tuple: (optimal_segment_indices_dp, optimal_num_segments, min_total_cost, raw_costs_per_segment_count)
               optimal_segment_indices_dp: List of time indices *in the downsampled scale*.
    """
    n_dims = min_vals_dp.shape[1]
    dp_start_time = time.time()
    POS_COLS = ['tx', 'ty', 'tz']
    print("Precomputing segment costs (on DP data)...")
    cost_cache = np.full((n_timesteps_dp, n_timesteps_dp), np.inf)
    for i in range(n_timesteps_dp):
        cost_cache[i, i] = 0.0
        for j in range(i + 1, n_timesteps_dp):
            segment_cost_total = 0.0
            # --- Loop modification starts here ---
            for d in range(n_dims):
                # Determine the weight for this dimension
                # Assumes SEGMENTATION_FEATURE_COLS order matches the columns in min_vals_dp/max_vals_dp
                feature_name = SEGMENTATION_FEATURE_COLS[d] # Get feature name
                current_weight = 1.0 # Default weight for position
                if feature_name not in POS_COLS: # If it's not position, assume rotation
                    current_weight = ROTATION_WEIGHT

                # Pass the determined weight to the cost function
                cost_d = calculate_segment_cost(i, j, min_vals_dp[:, d], max_vals_dp[:, d], weight=current_weight)
                # --- Loop modification ends here ---

                if not np.isfinite(cost_d):
                     segment_cost_total = np.inf
                     break
                segment_cost_total += cost_d
            cost_cache[i, j] = segment_cost_total
    # --- Precompute Costs on DP data ---
    print("Precomputing segment costs (on DP data)...")
    print(f"Segment costs precomputed in {time.time() - dp_start_time:.2f} seconds.")

    # --- DP Calculation ---
    dp = np.full((n_timesteps_dp, max_segments + 1), np.inf)
    bp = np.full((n_timesteps_dp, max_segments + 1), -1, dtype=int)

    # Initialization: Cost for 1 segment ending at time t (using DP scale indices)
    for t in range(n_timesteps_dp):
        if 0 <= t < cost_cache.shape[1]: # Check index validity
            dp[t, 1] = cost_cache[0, t]
            bp[t, 1] = 0
        else:
             dp[t, 1] = np.inf
             bp[t, 1] = -1

    print("Running dynamic programming for segmentation...")
    dp_fill_start_time = time.time()
    for m in range(2, max_segments + 1):
        for t in range(1, n_timesteps_dp): # End time of the current segment (DP scale)
            min_cost_for_t_m = np.inf
            best_prev_t_start_index = -1 # Start index of the *last* segment (DP scale)
            # Iterate through possible end points (j) of the (m-1)-th segment (DP scale)
            for j in range(m - 2, t):
                # Ensure indices j and j+1 are valid for accessing dp and cost_cache
                if j >= 0 and (j+1) < n_timesteps_dp: # Indices are relative to DP scale
                    cost_prev_segments = dp[j, m - 1]
                    # Cost of segment [j+1, t] (DP scale indices)
                    cost_last_segment = cost_cache[j + 1, t]

                    if np.isfinite(cost_prev_segments) and np.isfinite(cost_last_segment):
                        current_cost = cost_prev_segments + cost_last_segment
                        if current_cost < min_cost_for_t_m:
                            min_cost_for_t_m = current_cost
                            best_prev_t_start_index = j + 1 # Store the start index (DP scale)

            if best_prev_t_start_index != -1: # Check if a valid path was found
                 dp[t, m] = min_cost_for_t_m
                 bp[t, m] = best_prev_t_start_index

    print(f"DP table filled in {time.time() - dp_fill_start_time:.2f} seconds.")

    # --- Find Optimal Number of Segments ---
    # Costs are based on the last time step of the DP data
    final_raw_costs = dp[n_timesteps_dp - 1, 1:]
    final_costs_with_penalty = final_raw_costs + lambda_penalty * np.arange(1, max_segments + 1)

    valid_indices = np.where(np.isfinite(final_costs_with_penalty))[0]
    if len(valid_indices) == 0:
        print(f"Warning: Could not find a valid segmentation up to {max_segments} segments with finite cost.")
        if np.isfinite(dp[n_timesteps_dp - 1, 1]):
             optimal_num_segments = 1
             min_total_cost = dp[n_timesteps_dp - 1, 1] + lambda_penalty
             print("Falling back to 1 segment.")
        else:
             print("Error: Cannot find any valid segmentation.")
             return [0, n_timesteps_dp], 1, np.inf, final_raw_costs # Return DP scale indices on failure
    else:
        optimal_idx_in_valid = np.argmin(final_costs_with_penalty[valid_indices])
        optimal_num_segments = valid_indices[optimal_idx_in_valid] + 1
        min_total_cost = final_costs_with_penalty[valid_indices[optimal_idx_in_valid]]

    print(f"Optimal number of segments found: {optimal_num_segments} with total cost (incl. penalty): {min_total_cost:.4f}")

    # --- Backtrack to find segment boundaries (in DP scale) ---
    segment_boundaries_dp = [n_timesteps_dp] # End boundary (DP scale)
    current_t = n_timesteps_dp - 1
    current_m = optimal_num_segments
    while current_m > 0 and current_t >= 0:
         if current_t < bp.shape[0] and current_m < bp.shape[1]:
            start_of_last_segment = bp[current_t, current_m]
         else:
            print(f"Error: Index out of bounds during backtracking access bp[{current_t}, {current_m}]")
            start_of_last_segment = -1

         if start_of_last_segment == -1:
             print(f"Error during backtracking at t={current_t}, m={current_m}. Boundary not found.")
             if current_m == 1 and current_t >= 0 :
                  start_of_last_segment = 0
                  print("Assuming start at 0 for the first segment.")
             else:
                  print("Stopping backtracking due to error.")
                  break

         segment_boundaries_dp.append(start_of_last_segment)
         current_t = start_of_last_segment - 1
         current_m -= 1
         if current_t < 0 and current_m == 0: break

    # Clean up boundaries (DP scale)
    final_segment_indices_dp = sorted(list(set(segment_boundaries_dp)))
    if 0 not in final_segment_indices_dp: final_segment_indices_dp.insert(0,0)
    if n_timesteps_dp not in final_segment_indices_dp: final_segment_indices_dp.append(n_timesteps_dp)
    final_segment_indices_dp = sorted(list(set(final_segment_indices_dp)))

    # Return boundaries in DP scale, and the raw costs calculated
    return final_segment_indices_dp, optimal_num_segments, min_total_cost, final_raw_costs


# --- NEW: Function to Map Boundaries Back ---
def map_boundaries_to_original(segment_indices_dp, stride, n_timesteps_original):
    """Maps segment boundaries from downsampled scale back to original."""
    if stride == 1: # No downsampling occurred
        return segment_indices_dp

    segment_indices_orig = [int(round(idx * stride)) for idx in segment_indices_dp]

    # Ensure the last boundary maps correctly to the original end point
    if segment_indices_orig[-1] != n_timesteps_original:
         print(f"Adjusting final boundary from {segment_indices_orig[-1]} to {n_timesteps_original}")
         segment_indices_orig[-1] = n_timesteps_original

    # Ensure uniqueness and sort again after potential adjustment
    segment_indices_orig = sorted(list(set(segment_indices_orig)))

    print(f"Mapped DP boundaries back to original scale: {segment_indices_orig}")
    return segment_indices_orig

# --- Visualization (Unchanged plotting functions, but called with potentially mapped indices) ---

def plot_cost_vs_segments(raw_costs, max_segments, optimal_num_segments):
    """Plots the raw segmentation cost against the number of segments."""
    num_segments_axis = np.arange(1, max_segments + 1)
    plot_costs = np.full(max_segments, np.inf) # Initialize with inf
    valid_len = min(len(raw_costs), max_segments)
    plot_costs[:valid_len] = raw_costs[:valid_len] # Fill with calculated costs

    plt.figure(figsize=(10, 5))
    # Plot only finite costs
    valid_idx = np.isfinite(plot_costs)
    if np.any(valid_idx):
        plt.plot(num_segments_axis[valid_idx], plot_costs[valid_idx], marker='o', linestyle='-')

        # Highlight the optimal number of segments found
        if 1 <= optimal_num_segments <= max_segments and np.isfinite(plot_costs[optimal_num_segments-1]):
             plt.scatter(optimal_num_segments, plot_costs[optimal_num_segments - 1], color='red', s=100, zorder=5, label=f'Optimal ({optimal_num_segments} segments)')
             plt.legend()

    plt.title('Raw Segmentation Cost vs. Number of Segments')
    plt.xlabel('Number of Segments')
    plt.ylabel('Total Raw Segmentation Cost (Sum of Squared Errors)')
    plt.xticks(num_segments_axis)
    plt.grid(True, linestyle=':')
    min_finite_cost = np.min(plot_costs[valid_idx]) if np.any(valid_idx) else 0
    plt.ylim(bottom=min(0, min_finite_cost * 0.9)) # Adjust ylim based on data
    sns.set_style("whitegrid")
    plt.tight_layout()
    plt.show()


def plot_segmentation_with_trapezoids(min_vals_orig, max_vals_orig, segment_indices_orig, feature_names, title="Optimal Tube Segmentation with Approximations"):
    """
    Plots the *original* tube, segmentation boundaries mapped to original scale,
    and linear approximations based on the *original* data within those segments.
    """
    n_timesteps_orig, n_dims = min_vals_orig.shape
    time_axis_orig = np.arange(n_timesteps_orig)

    fig, axs = plt.subplots(n_dims, 1, figsize=(15, 5 * n_dims), sharex=True)
    if n_dims == 1: axs = [axs]
    fig.suptitle(title, fontsize=16)
    sns.set_style("whitegrid")

    for d in range(n_dims):
        # Plot the ORIGINAL tube
        axs[d].plot(time_axis_orig, max_vals_orig[:, d], color='lightblue', linestyle=':', linewidth=1.5, label=f'Max {feature_names[d]} (Actual)')
        axs[d].plot(time_axis_orig, min_vals_orig[:, d], color='lightcoral', linestyle=':', linewidth=1.5, label=f'Min {feature_names[d]} (Actual)')
        axs[d].fill_between(time_axis_orig, min_vals_orig[:, d], max_vals_orig[:, d], color='lightgrey', alpha=0.4, label='Tube')

        # Plot segmentation boundaries (mapped to original scale)
        plotted_boundary_label = False
        for i, boundary_idx_orig in enumerate(segment_indices_orig):
             # Don't plot at 0 or exactly at the end (n_timesteps_orig)
             if i > 0 and boundary_idx_orig < n_timesteps_orig:
                 label = 'Segment Boundary' if not plotted_boundary_label else ""
                 axs[d].axvline(x=boundary_idx_orig, color='black', linestyle='--', linewidth=1.5, label=label)
                 plotted_boundary_label = True

        # Plot the linear approximations (trapezoids) for each segment using ORIGINAL data
        plotted_approx_label = False
        for i in range(len(segment_indices_orig) - 1):
            start_idx_orig = segment_indices_orig[i]
            # End index is inclusive for the segment
            end_idx_orig = segment_indices_orig[i+1] - 1
            # Ensure indices are valid for the original data
            if end_idx_orig < start_idx_orig or start_idx_orig >= n_timesteps_orig or end_idx_orig >= n_timesteps_orig:
                 print(f"Skipping invalid original segment indices: start={start_idx_orig}, end={end_idx_orig}")
                 continue

            segment_time_axis_orig = np.arange(start_idx_orig, end_idx_orig + 1)
            segment_len_points = len(segment_time_axis_orig)

            if segment_len_points <= 1: continue

            # Get ORIGINAL start/end values for this segment
            min_start_val = min_vals_orig[start_idx_orig, d]
            min_end_val = min_vals_orig[end_idx_orig, d]
            max_start_val = max_vals_orig[start_idx_orig, d]
            max_end_val = max_vals_orig[end_idx_orig, d]

            # Calculate linear interpolation (trapezoid edges) based on original segment endpoints
            t_interp = np.arange(segment_len_points)
            # Avoid division by zero if segment length is 1
            min_slope = (min_end_val - min_start_val) / (segment_len_points - 1) if segment_len_points > 1 else 0
            approx_min = min_start_val + min_slope * t_interp

            max_slope = (max_end_val - max_start_val) / (segment_len_points - 1) if segment_len_points > 1 else 0
            approx_max = max_start_val + max_slope * t_interp

            # Plot the approximation lines for this segment on original time axis
            label_approx = 'Linear Approximation' if not plotted_approx_label else ""
            axs[d].plot(segment_time_axis_orig, approx_max, color='blue', linestyle='-', linewidth=2, label=label_approx)
            axs[d].plot(segment_time_axis_orig, approx_min, color='red', linestyle='-', linewidth=2)
            plotted_approx_label = True

        axs[d].set_ylabel(feature_names[d])
        axs[d].grid(True, linestyle=':')
        handles, labels = axs[d].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[d].legend(by_label.values(), by_label.keys(), loc='upper right')

    axs[-1].set_xlabel('Time Step (Original Aligned Scale)') # Label reflects original scale
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    overall_start_time = time.time()
    # 1. Load data
    print("--- Loading Data ---")
    original_trajs_df, loaded_files = load_selected_data(
        PARENT_FOLDER_PATH,
        load_all=LOAD_ALL_FILES,
        file_list=FILE_LIST,
        feature_columns=list(set(ALIGNMENT_BASE_FEATURE_COLS + SEGMENTATION_FEATURE_COLS))
    )
    if not original_trajs_df: exit("Exiting: No trajectories loaded.")

    # 2. Perform Alignment
    print("\n--- Aligning Trajectories ---")
    align_start_time = time.time()
    aligned_trajs_df, ref_idx = align_trajectories(
        original_trajs_df, ALIGNMENT_TYPE, ALIGNMENT_BASE_FEATURE_COLS, NORMALIZE_FOR_COMBINED
    )
    print(f"Alignment complete in {time.time() - align_start_time:.2f} seconds.")
    valid_aligned_trajs_df = [df for df in aligned_trajs_df if not df.empty]
    if not valid_aligned_trajs_df: exit("Exiting: Alignment failed or produced no valid trajectories.")
    # print(f"Alignment complete. {len(valid_aligned_trajs_df)} valid aligned trajectories.")

    # 3. Calculate Tube
    print("\n--- Calculating Tube ---")
    min_vals_orig, max_vals_orig, n_timesteps_orig = calculate_tube(valid_aligned_trajs_df, SEGMENTATION_FEATURE_COLS)
    if min_vals_orig is None or n_timesteps_orig <= 1: exit("Exiting: Failed to calculate tube or not enough time steps.")

    # --- NEW: Downsample if necessary ---
    min_vals_dp, max_vals_dp, n_timesteps_dp, stride = downsample_tube_data(
        min_vals_orig, max_vals_orig, MAX_DP_LENGTH
    )
    # ------------------------------------

    # 4. Find Optimal Segmentation using DP on potentially downsampled data
    print("\n--- Finding Optimal Segmentation (DP) ---")
    segmentation_start_time = time.time()
    # Pass the data intended for DP (potentially downsampled)
    segment_indices_dp, optimal_num_segments, min_total_cost, raw_costs_per_segment_count = find_optimal_segmentation(
        min_vals_dp, max_vals_dp, n_timesteps_dp, MAX_SEGMENTS, LAMBDA_PENALTY
    )
    print(f"Segmentation DP finished in {time.time() - segmentation_start_time:.2f} seconds.")

    if optimal_num_segments == -1: # Check if DP failed
         exit("Exiting: Segmentation optimization failed.")

    # --- NEW: Map boundaries back to original scale ---
    optimal_segment_indices_orig = map_boundaries_to_original(
        segment_indices_dp, stride, n_timesteps_orig
    )
    # -------------------------------------------------

    print(f"\nOptimal Segmentation Found:")
    print(f"  Number of Segments: {optimal_num_segments}")
    print(f"  Segment Boundaries (Original Time Scale): {optimal_segment_indices_orig}")
    print(f"  Minimum Total Cost (incl. penalty, based on DP data): {min_total_cost:.4f}")

    # 5. Plot Results
    print("\n--- Plotting Results ---")
    # Plot 1: Cost vs. Number of Segments (Costs are from DP on potentially downsampled data)
    plot_cost_vs_segments(raw_costs_per_segment_count, MAX_SEGMENTS, optimal_num_segments)

    # Plot 2: Optimal Segmentation with Trapezoid Approximations
    # Use ORIGINAL tube data and MAPPED-BACK boundaries
    plot_segmentation_with_trapezoids(
        min_vals_orig,
        max_vals_orig,
        optimal_segment_indices_orig, # Use mapped indices
        SEGMENTATION_FEATURE_COLS,
        title=f'Optimal Segmentation ({optimal_num_segments} Segments, Lambda={LAMBDA_PENALTY}, DP Length Max={MAX_DP_LENGTH})'
    )

    print(f"\n--- Script Finished in {time.time() - overall_start_time:.2f} seconds ---")

