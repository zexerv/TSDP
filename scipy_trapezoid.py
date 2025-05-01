# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import sys
from scipy.optimize import minimize # Import the optimizer

# --- Configuration ---
# !!! PLEASE UPDATE THIS PATH !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button' # <--- CHANGE THIS

LOAD_ALL_FILES = True
FILE_LIST = []
ALIGNMENT_TYPE = 'Combined'
ALIGNMENT_BASE_FEATURE_COLS = ['tx', 'ty', 'tz']
NORMALIZE_FOR_COMBINED = True
SEGMENTATION_FEATURE_COLS = ['tx', 'ty', 'tz']
MAX_SEGMENTS = 10 # Max number of segments to *try*
LAMBDA_PENALTY = 1.0 # Penalty for adding a segment

# --- L-BFGS-B Specific Configuration ---
INITIAL_GUESS_STRATEGY = 'uniform' # How to generate the first guess for boundaries
OPTIMIZER_OPTIONS = {'disp': False, 'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6} # Optimizer settings

print(f"--- Configuration ---")
print(f"Data Folder: {PARENT_FOLDER_PATH}")
print(f"Alignment Type: {ALIGNMENT_TYPE}")
print(f"Base features for Alignment: {ALIGNMENT_BASE_FEATURE_COLS}")
if ALIGNMENT_TYPE == 'Combined':
    print(f"Normalization before combined DTW: {NORMALIZE_FOR_COMBINED}")
print(f"Features for Segmentation: {SEGMENTATION_FEATURE_COLS}")
print(f"Max Segments to Try: {MAX_SEGMENTS}")
print(f"Segment Penalty (Lambda): {LAMBDA_PENALTY}")
print(f"Optimization Method: L-BFGS-B (via scipy.optimize.minimize)")
print(f"Optimizer Options: {OPTIMIZER_OPTIONS}")
print("-" * 20)


# --- Data Loading (Identical to previous script) ---
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

# --- Alignment (DTW/DDTW/Combined) (Identical to previous script) ---
def calculate_derivative(trajectory):
    if len(trajectory) < 2: return np.zeros_like(trajectory)
    return np.diff(trajectory, axis=0, prepend=trajectory[0:1,:])

def align_trajectories(trajectories_df_list, alignment_type, alignment_base_feature_cols, normalize_combined=True):
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
        print(f"  Aligning trajectory {i} (length {len(traj_base)}) to reference...")
        if len(traj_base) == 0:
             print(f"    Skipping empty trajectory {i}")
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
                 print("    Warning: Both derivative sequences zero. Using DTW fallback.")
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
                else: ref_pos_norm, traj_pos_norm = ref_pos, traj_pos # Skip norm
                if np.ptp(combined_deriv, axis=0).min() > 1e-6:
                     scaler_deriv.fit(combined_deriv)
                     ref_deriv_norm = scaler_deriv.transform(ref_deriv)
                     traj_deriv_norm = scaler_deriv.transform(traj_deriv)
                else: ref_deriv_norm, traj_deriv_norm = ref_deriv, traj_deriv # Skip norm

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


# --- Tube Calculation (Identical to previous script) ---
def calculate_tube(aligned_trajectories_df_list, feature_cols):
    """Calculates the min/max envelope (tube) for specified features."""
    if not aligned_trajectories_df_list:
        print("Warning: No aligned trajectories to calculate tube from.")
        return None, None, 0

    stacked_data = []
    target_len = len(aligned_trajectories_df_list[0]) # Assumes first is valid reference length
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

# --- Segmentation Cost Function (Identical to previous script) ---
def calculate_segment_cost(start_idx, end_idx, min_vals_dim, max_vals_dim):
    """Calculates the cost of approximating a tube segment with linear boundaries."""
    if start_idx >= end_idx: return 0.0
    segment_len_points = end_idx - start_idx + 1
    if segment_len_points <= 1: return 0.0

    # Check if indices are within bounds before slicing
    if start_idx < 0 or end_idx >= len(min_vals_dim) or end_idx >= len(max_vals_dim):
         print(f"Warning: Invalid indices for cost calculation: {start_idx}, {end_idx}")
         return np.inf # Return infinite cost for invalid segment

    t = np.arange(segment_len_points)
    actual_min = min_vals_dim[start_idx : end_idx + 1]
    actual_max = max_vals_dim[start_idx : end_idx + 1]

    min_start_val, min_end_val = actual_min[0], actual_min[-1]
    max_start_val, max_end_val = actual_max[0], actual_max[-1]

    # Avoid division by zero if segment length is 1 (handled above)
    min_slope = (min_end_val - min_start_val) / (segment_len_points - 1)
    approx_min = min_start_val + min_slope * t
    max_slope = (max_end_val - max_start_val) / (segment_len_points - 1)
    approx_max = max_start_val + max_slope * t

    error_min = np.sum((actual_min - approx_min)**2)
    error_max = np.sum((actual_max - approx_max)**2)
    return error_min + error_max

# --- L-BFGS-B Based Segmentation ---

# Objective function for scipy.optimize.minimize
def segmentation_objective(continuous_boundaries_x, n_timesteps, min_vals, max_vals, cost_cache):
    """
    Calculates the total raw segmentation cost for a given set of continuous boundary guesses.

    Args:
        continuous_boundaries_x (np.array): Array of M-1 continuous boundary locations.
        n_timesteps (int): Total number of time steps.
        min_vals (np.array): Min tube values (n_timesteps, n_dims).
        max_vals (np.array): Max tube values (n_timesteps, n_dims).
        cost_cache (np.array): Precomputed costs C(i, j).

    Returns:
        float: Total raw segmentation cost.
    """
    n_dims = min_vals.shape[1]

    # --- Discretize and Validate Boundaries ---
    # Round, clamp to valid internal indices [1, n_timesteps-2], sort, unique
    if len(continuous_boundaries_x) == 0: # Case for M=1 segment
        discrete_boundaries_t = []
    else:
        discrete_boundaries_t = np.round(continuous_boundaries_x).astype(int)
        # Clamp values to be valid *internal* split points
        discrete_boundaries_t = np.clip(discrete_boundaries_t, 1, n_timesteps - 1)
        # Sort and get unique values
        discrete_boundaries_t = sorted(list(set(discrete_boundaries_t)))

    # Form the full list of segment boundaries including start and end
    final_boundaries = [0] + discrete_boundaries_t + [n_timesteps]
    # Ensure uniqueness again after adding 0 and n_timesteps
    final_boundaries = sorted(list(set(final_boundaries)))

    # --- Calculate Total Cost ---
    total_cost = 0.0
    for k in range(len(final_boundaries) - 1):
        start_idx = final_boundaries[k]
        # End index for cost calculation is inclusive
        end_idx = final_boundaries[k+1] - 1

        # Basic check for valid segment definition
        if start_idx > end_idx or start_idx < 0 or end_idx < 0 or end_idx >= n_timesteps:
             # Penalize invalid segments heavily
             # print(f"Warning: Invalid segment created: {start_idx} to {end_idx}")
             total_cost += np.inf
             continue # Skip this segment if indices are invalid

        # Use precomputed cost if available and valid
        if start_idx < cost_cache.shape[0] and end_idx < cost_cache.shape[1] and np.isfinite(cost_cache[start_idx, end_idx]):
            total_cost += cost_cache[start_idx, end_idx]
        else:
            # Fallback: Recalculate cost if not in cache or cache value is invalid (should not happen if cache is complete)
            # print(f"Recalculating cost for segment [{start_idx}, {end_idx}]") # Debugging
            segment_cost_recalc = 0.0
            for d in range(n_dims):
                  segment_cost_recalc += calculate_segment_cost(start_idx, end_idx, min_vals[:, d], max_vals[:, d])
            total_cost += segment_cost_recalc
            # Optionally update cache if recalculated: cost_cache[start_idx, end_idx] = segment_cost_recalc

    # Handle potential non-finite costs
    if not np.isfinite(total_cost):
        return np.inf # Return a large value if something went wrong

    return total_cost


# Main segmentation function using L-BFGS-B
def find_segmentation_lbfgs(min_vals, max_vals, n_timesteps, max_segments, lambda_penalty):
    """
    Finds segmentation using L-BFGS-B for each possible number of segments.

    Returns:
        tuple: (optimal_segment_indices, optimal_num_segments, min_total_cost_with_penalty, raw_costs_per_segment_count)
    """
    n_dims = min_vals.shape[1]

    # --- Precompute Costs (Still useful!) ---
    print("Precomputing segment costs (for objective function evaluation)...")
    cost_cache = np.full((n_timesteps, n_timesteps), np.inf)
    for i in range(n_timesteps):
        cost_cache[i, i] = 0.0
        for j in range(i + 1, n_timesteps):
            segment_cost_total = 0.0
            for d in range(n_dims):
                 cost_d = calculate_segment_cost(i, j, min_vals[:, d], max_vals[:, d])
                 segment_cost_total += cost_d
            cost_cache[i, j] = segment_cost_total
    print("Segment costs precomputed.")

    # Store results for each M
    results_per_m = []
    raw_costs_list = []

    print(f"\n--- Optimizing Segmentation using L-BFGS-B (M=1 to {max_segments}) ---")
    for m_segments in range(1, max_segments + 1):
        print(f"Optimizing for M = {m_segments} segments...")
        num_boundaries_to_find = m_segments - 1

        if num_boundaries_to_find == 0:
            # Case M=1: No internal boundaries to optimize
            best_raw_cost_for_m = cost_cache[0, n_timesteps - 1]
            final_boundaries_for_m = [0, n_timesteps]
            print(f"  M=1: Raw cost = {best_raw_cost_for_m:.4f}")
        else:
            # Define bounds for the continuous variables
            bounds = [(0, n_timesteps -1)] * num_boundaries_to_find # Boundaries can be from 0 to N-1 technically before rounding

            # Define initial guess (uniformly spaced continuous points)
            # np.linspace points are: start, stop, num
            initial_guess_x0 = np.linspace(0, n_timesteps, m_segments + 1)[1:-1]

            # Run the optimizer
            optimization_result = minimize(
                segmentation_objective,
                initial_guess_x0,
                args=(n_timesteps, min_vals, max_vals, cost_cache),
                method='L-BFGS-B',
                bounds=bounds,
                options=OPTIMIZER_OPTIONS
            )

            if optimization_result.success:
                best_raw_cost_for_m = optimization_result.fun
                optimized_x = optimization_result.x
                # Convert final optimized continuous values back to discrete boundaries
                discrete_boundaries_t = np.round(optimized_x).astype(int)
                discrete_boundaries_t = np.clip(discrete_boundaries_t, 1, n_timesteps - 1)
                discrete_boundaries_t = sorted(list(set(discrete_boundaries_t)))
                final_boundaries_for_m = sorted(list(set([0] + discrete_boundaries_t + [n_timesteps])))
                print(f"  M={m_segments}: Success! Raw cost = {best_raw_cost_for_m:.4f}. Boundaries ~ {discrete_boundaries_t}")

            else:
                print(f"  M={m_segments}: Optimization failed or did not converge. Status: {optimization_result.status}, Message: {optimization_result.message}")
                # Use initial guess or a fallback? For now, assign infinite cost.
                best_raw_cost_for_m = np.inf
                final_boundaries_for_m = [] # Indicate failure

        # Store result for this M
        results_per_m.append({
            'm': m_segments,
            'raw_cost': best_raw_cost_for_m,
            'boundaries': final_boundaries_for_m
        })
        raw_costs_list.append(best_raw_cost_for_m)


    # --- Select the Best M based on Penalty ---
    best_m = -1
    min_total_cost_with_penalty = np.inf
    optimal_boundaries = []

    print("\n--- Selecting Optimal Number of Segments ---")
    for result in results_per_m:
        m = result['m']
        raw_cost = result['raw_cost']
        boundaries = result['boundaries']

        if np.isfinite(raw_cost) and boundaries: # Check if optimization was successful
            total_cost = raw_cost + lambda_penalty * m
            print(f" M={m}: Raw Cost={raw_cost:.4f}, Penalty={lambda_penalty * m:.4f}, Total Cost={total_cost:.4f}")
            if total_cost < min_total_cost_with_penalty:
                min_total_cost_with_penalty = total_cost
                best_m = m
                optimal_boundaries = boundaries
        else:
             print(f" M={m}: Skipping (Optimization failed or cost is inf)")


    if best_m == -1:
        print("Error: Could not find any valid segmentation solution.")
        # Return default/error state
        return [0, n_timesteps], 1, np.inf, raw_costs_list

    print(f"\nSelected Optimal M = {best_m} with Total Cost (incl. penalty) = {min_total_cost_with_penalty:.4f}")

    # Ensure raw_costs_list has the correct length for plotting
    padded_raw_costs = np.full(max_segments, np.inf)
    valid_len = min(len(raw_costs_list), max_segments)
    padded_raw_costs[:valid_len] = raw_costs_list[:valid_len]


    return optimal_boundaries, best_m, min_total_cost_with_penalty, padded_raw_costs


# --- Visualization (Identical to previous script) ---

def plot_cost_vs_segments(raw_costs, max_segments, optimal_num_segments):
    """Plots the raw segmentation cost against the number of segments."""
    num_segments_axis = np.arange(1, max_segments + 1)
    plot_costs = raw_costs # Already padded in the main function

    plt.figure(figsize=(10, 5))
    # Plot only finite costs
    valid_idx = np.isfinite(plot_costs)
    if np.any(valid_idx):
        plt.plot(num_segments_axis[valid_idx], plot_costs[valid_idx], marker='o', linestyle='-')

        # Highlight the optimal number of segments found (considering penalty)
        if 1 <= optimal_num_segments <= max_segments and np.isfinite(plot_costs[optimal_num_segments-1]):
             plt.scatter(optimal_num_segments, plot_costs[optimal_num_segments - 1], color='red', s=100, zorder=5, label=f'Selected Optimum ({optimal_num_segments} segments)')
             plt.legend() # Show legend only if optimum is plotted

    plt.title('Raw Segmentation Cost vs. Number of Segments (L-BFGS-B)')
    plt.xlabel('Number of Segments (M)')
    plt.ylabel('Total Raw Segmentation Cost (Sum of Squared Errors)')
    plt.xticks(num_segments_axis)
    plt.grid(True, linestyle=':')
    min_finite_cost = np.min(plot_costs[valid_idx]) if np.any(valid_idx) else 0
    plt.ylim(bottom=min(0, min_finite_cost * 0.9)) # Adjust ylim based on data
    sns.set_style("whitegrid")
    plt.tight_layout()
    plt.show()


def plot_segmentation_with_trapezoids(min_vals, max_vals, segment_indices, feature_names, title="Optimal Tube Segmentation with Approximations"):
    """Plots the tube, segmentation boundaries, and linear approximations (trapezoids)."""
    n_timesteps, n_dims = min_vals.shape
    time_axis = np.arange(n_timesteps)

    fig, axs = plt.subplots(n_dims, 1, figsize=(15, 5 * n_dims), sharex=True)
    if n_dims == 1: axs = [axs]
    fig.suptitle(title, fontsize=16)
    sns.set_style("whitegrid")

    for d in range(n_dims):
        axs[d].plot(time_axis, max_vals[:, d], color='lightblue', linestyle=':', linewidth=1.5, label=f'Max {feature_names[d]} (Actual)')
        axs[d].plot(time_axis, min_vals[:, d], color='lightcoral', linestyle=':', linewidth=1.5, label=f'Min {feature_names[d]} (Actual)')
        axs[d].fill_between(time_axis, min_vals[:, d], max_vals[:, d], color='lightgrey', alpha=0.4, label='Tube')

        plotted_boundary_label = False
        for i, boundary_idx in enumerate(segment_indices):
             if i > 0 and boundary_idx < n_timesteps:
                 label = 'Segment Boundary' if not plotted_boundary_label else ""
                 axs[d].axvline(x=boundary_idx, color='black', linestyle='--', linewidth=1.5, label=label)
                 plotted_boundary_label = True

        plotted_approx_label = False
        for i in range(len(segment_indices) - 1):
            start_idx = segment_indices[i]
            end_idx = segment_indices[i+1] - 1
            if end_idx < start_idx or start_idx >= n_timesteps or end_idx >= n_timesteps: continue

            segment_time_axis = np.arange(start_idx, end_idx + 1)
            segment_len_points = len(segment_time_axis)
            if segment_len_points <= 1: continue

            min_start_val, min_end_val = min_vals[start_idx, d], min_vals[end_idx, d]
            max_start_val, max_end_val = max_vals[start_idx, d], max_vals[end_idx, d]

            t_interp = np.arange(segment_len_points)
            # Check for division by zero if segment_len_points is 1 (handled above)
            min_slope = (min_end_val - min_start_val) / (segment_len_points - 1)
            approx_min = min_start_val + min_slope * t_interp
            max_slope = (max_end_val - max_start_val) / (segment_len_points - 1)
            approx_max = max_start_val + max_slope * t_interp

            label_approx = 'Linear Approximation' if not plotted_approx_label else ""
            axs[d].plot(segment_time_axis, approx_max, color='blue', linestyle='-', linewidth=2, label=label_approx)
            axs[d].plot(segment_time_axis, approx_min, color='red', linestyle='-', linewidth=2)
            plotted_approx_label = True

        axs[d].set_ylabel(feature_names[d])
        axs[d].grid(True, linestyle=':')
        handles, labels = axs[d].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[d].legend(by_label.values(), by_label.keys(), loc='upper right')

    axs[-1].set_xlabel('Time Step (Aligned)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
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
    aligned_trajs_df, ref_idx = align_trajectories(
        original_trajs_df, ALIGNMENT_TYPE, ALIGNMENT_BASE_FEATURE_COLS, NORMALIZE_FOR_COMBINED
    )
    valid_aligned_trajs_df = [df for df in aligned_trajs_df if not df.empty]
    if not valid_aligned_trajs_df: exit("Exiting: Alignment failed or produced no valid trajectories.")
    print(f"Alignment complete. {len(valid_aligned_trajs_df)} valid aligned trajectories.")

    # 3. Calculate Tube
    print("\n--- Calculating Tube ---")
    min_vals, max_vals, n_timesteps = calculate_tube(valid_aligned_trajs_df, SEGMENTATION_FEATURE_COLS)
    if min_vals is None or n_timesteps <= 1: exit("Exiting: Failed to calculate tube or not enough time steps.")

    # 4. Find Optimal Segmentation using L-BFGS-B
    print("\n--- Finding Segmentation (L-BFGS-B) ---")
    optimal_segment_indices, optimal_num_segments, min_total_cost, raw_costs_per_segment_count = find_segmentation_lbfgs(
        min_vals, max_vals, n_timesteps, MAX_SEGMENTS, LAMBDA_PENALTY
    )

    if optimal_num_segments == -1: # Check if optimization failed entirely
         exit("Exiting: Segmentation optimization failed for all segment counts.")

    print(f"\nL-BFGS-B Based Segmentation Chosen:")
    print(f"  Optimal Number of Segments: {optimal_num_segments}")
    print(f"  Segment Boundaries (time indices): {optimal_segment_indices}")
    print(f"  Minimum Total Cost (incl. penalty): {min_total_cost:.4f}")


    # 5. Plot Results
    print("\n--- Plotting Results ---")
    # Plot 1: Cost vs. Number of Segments
    plot_cost_vs_segments(raw_costs_per_segment_count, MAX_SEGMENTS, optimal_num_segments)
    # Plot 2: Optimal Segmentation with Trapezoid Approximations
    plot_segmentation_with_trapezoids(
        min_vals,
        max_vals,
        optimal_segment_indices,
        SEGMENTATION_FEATURE_COLS,
        title=f'Segmentation (L-BFGS-B: {optimal_num_segments} Segments, Lambda={LAMBDA_PENALTY}) with Approximations'
    )

    print("\n--- Script Finished ---")