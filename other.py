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
import time
# --- Imports for Other Algorithms ---
import ruptures as rpt  # For PELT
from hmmlearn import hmm # For HMM

# --- Configuration ---
# !!! PLEASE UPDATE THIS PATH !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/lever' # <--- From user's code

LOAD_ALL_FILES = True
FILE_LIST = []
ALIGNMENT_TYPE = 'Combined'
ALIGNMENT_BASE_FEATURE_COLS = ['tx', 'ty', 'tz', 'r11','r12','r13','r21','r22','r23','r31','r32','r33']
NORMALIZE_FOR_COMBINED = True

# --- Segmentation Configuration (Common) ---
SEGMENTATION_FEATURE_COLS = ['tx', 'ty', 'tz', 'r11','r12','r13','r21','r22','r23','r31','r32','r33']
MAX_SEGMENTS = 10 # Target/Max number of segments for DP methods
LAMBDA_PENALTY_PLA = 0.5 # Penalty for PLA DP (NEEDS TUNING)

# --- Downsampling Configuration (for PLA DP) ---
MAX_DP_LENGTH = 300 # From user's code

# --- PELT Configuration ---
PELT_MODEL = 'l2'
PELT_PENALTY = 5 # NEEDS TUNING

# --- HMM Configuration ---
HMM_N_STATES = 3 # NEEDS TUNING
HMM_N_ITER = 100
HMM_COVAR_TYPE = 'diag'

print(f"--- Configuration ---")
print(f"Data Folder: {PARENT_FOLDER_PATH}")
print(f"Alignment Type: {ALIGNMENT_TYPE}")
print(f"Alignment Features: {ALIGNMENT_BASE_FEATURE_COLS}")
print(f"Segmentation Features: {SEGMENTATION_FEATURE_COLS}")
print(f"Max Segments (DP): {MAX_SEGMENTS}")
print(f"--- PLA-DP Settings ---")
print(f"Lambda Penalty (PLA): {LAMBDA_PENALTY_PLA}")
print(f"Max DP Length: {MAX_DP_LENGTH}")
print(f"--- PELT Settings ---")
print(f"Model: {PELT_MODEL}, Penalty: {PELT_PENALTY}")
print(f"--- HMM Settings ---")
print(f"Num States: {HMM_N_STATES}, Max Iter: {HMM_N_ITER}, Covar Type: {HMM_COVAR_TYPE}")
print("-" * 20)

# --- Data Loading ---
def load_selected_data(parent_folder, load_all=True, file_list=None, feature_columns=None):
    """Loads specified trajectories and features from CSV files."""
    all_trajectories = []
    loaded_filenames = []
    required_columns = list(set(ALIGNMENT_BASE_FEATURE_COLS + SEGMENTATION_FEATURE_COLS))
    if feature_columns is None: feature_columns = required_columns

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

# --- Alignment ---
def calculate_derivative(trajectory):
    if len(trajectory) < 2: return np.zeros_like(trajectory)
    return np.diff(trajectory, axis=0, prepend=trajectory[0:1,:])

def align_trajectories(trajectories_df_list, alignment_type, alignment_base_feature_cols, normalize_combined=True):
    if not trajectories_df_list: return [], -1
    if alignment_type not in ['DTW', 'DDTW', 'Combined']: raise ValueError("Invalid alignment_type")

    first_df_cols = trajectories_df_list[0].columns.tolist()
    missing_align_cols = [col for col in alignment_base_feature_cols if col not in first_df_cols]
    if missing_align_cols: raise ValueError(f"Alignment features missing: {missing_align_cols}")

    trajectories_align_np = [df[alignment_base_feature_cols].values for df in trajectories_df_list]
    trajectories_full_np = [df.values for df in trajectories_df_list]
    required_columns = trajectories_df_list[0].columns.tolist()

    lengths = [len(t) for t in trajectories_align_np]
    if not lengths: return [], -1
    reference_index = np.argmax(lengths)
    reference_trajectory_align = trajectories_align_np[reference_index]
    reference_trajectory_full = trajectories_full_np[reference_index]
    ref_len = len(reference_trajectory_align)
    print(f"\nUsing trajectory {reference_index} (length {ref_len}) as reference for {alignment_type}.")

    aligned_trajectories_np = []
    for i, traj_align in enumerate(trajectories_align_np):
        traj_full = trajectories_full_np[i]
        if len(traj_align) == 0:
             aligned_trajectories_np.append(np.zeros_like(reference_trajectory_full))
             continue
        if i == reference_index:
            aligned_trajectories_np.append(reference_trajectory_full.copy())
            continue

        ref_data_for_dtw, traj_data_for_dtw = None, None
        if alignment_type == 'DTW':
            ref_data_for_dtw = reference_trajectory_align
            traj_data_for_dtw = traj_align
        elif alignment_type == 'DDTW':
            ref_data_for_dtw = calculate_derivative(reference_trajectory_align)
            traj_data_for_dtw = calculate_derivative(traj_align)
            if not np.any(ref_data_for_dtw) and not np.any(traj_data_for_dtw):
                 ref_data_for_dtw = reference_trajectory_align
                 traj_data_for_dtw = traj_align
        elif alignment_type == 'Combined':
            ref_deriv = calculate_derivative(reference_trajectory_align)
            traj_deriv = calculate_derivative(traj_align)
            ref_pos = reference_trajectory_align
            traj_pos = traj_align
            if normalize_combined:
                scaler_pos = MinMaxScaler()
                scaler_deriv = MinMaxScaler()
                valid_pos_cols = np.where(np.ptp(np.vstack((ref_pos, traj_pos)), axis=0) > 1e-6)[0]
                valid_deriv_cols = np.where(np.ptp(np.vstack((ref_deriv, traj_deriv)), axis=0) > 1e-6)[0]
                ref_pos_norm, traj_pos_norm = ref_pos.copy(), traj_pos.copy()
                ref_deriv_norm, traj_deriv_norm = ref_deriv.copy(), traj_deriv.copy()
                if len(valid_pos_cols) > 0:
                    scaler_pos.fit(np.vstack((ref_pos[:, valid_pos_cols], traj_pos[:, valid_pos_cols])))
                    ref_pos_norm[:, valid_pos_cols] = scaler_pos.transform(ref_pos[:, valid_pos_cols])
                    traj_pos_norm[:, valid_pos_cols] = scaler_pos.transform(traj_pos[:, valid_pos_cols])
                if len(valid_deriv_cols) > 0:
                    scaler_deriv.fit(np.vstack((ref_deriv[:, valid_deriv_cols], traj_deriv[:, valid_deriv_cols])))
                    ref_deriv_norm[:, valid_deriv_cols] = scaler_deriv.transform(ref_deriv[:, valid_deriv_cols])
                    traj_deriv_norm[:, valid_deriv_cols] = scaler_deriv.transform(traj_deriv[:, valid_deriv_cols])
                ref_data_for_dtw = np.hstack((ref_pos_norm, ref_deriv_norm))
                traj_data_for_dtw = np.hstack((traj_pos_norm, traj_deriv_norm))
            else:
                ref_data_for_dtw = np.hstack((ref_pos, ref_deriv))
                traj_data_for_dtw = np.hstack((traj_pos, traj_deriv))

        distance, path = fastdtw(ref_data_for_dtw, traj_data_for_dtw, dist=euclidean)

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


# --- Mean Trajectory Calculation ---
def calculate_mean_trajectory(aligned_trajectories_df_list, feature_cols):
    """Calculates the mean trajectory for specified features."""
    if not aligned_trajectories_df_list: return None, 0
    target_len = 0
    for df in aligned_trajectories_df_list:
        if not df.empty: target_len = len(df); break
    if target_len == 0: return None, 0

    stacked_data = []
    valid_traj_count = 0
    for i, df in enumerate(aligned_trajectories_df_list):
        if not df.empty and len(df) == target_len and all(c in df.columns for c in feature_cols):
            stacked_data.append(df[feature_cols].values)
            valid_traj_count += 1

    if not stacked_data: return None, 0
    print(f"Calculating mean trajectory from {valid_traj_count} valid trajectories.")

    try: trajectory_array = np.stack(stacked_data, axis=0)
    except ValueError as e: print(f"Error stacking trajectories for mean: {e}."); return None, 0

    mean_traj = np.mean(trajectory_array, axis=0)
    n_timesteps = mean_traj.shape[0]
    print(f"Calculated mean trajectory: {n_timesteps} time steps, {mean_traj.shape[1]} dimensions.")
    return mean_traj, n_timesteps

# --- Downsampling Function ---
def downsample_data(data_array, max_length):
    """Downsamples numpy array along axis 0."""
    n_timesteps_orig = data_array.shape[0]
    if n_timesteps_orig <= max_length: return data_array, n_timesteps_orig, 1
    stride = math.ceil(n_timesteps_orig / max_length)
    print(f"Downsampling data: Original length={n_timesteps_orig}, Target max={max_length}, Stride={stride}")
    data_ds = data_array[::stride]
    n_timesteps_ds = data_ds.shape[0]
    print(f"Downsampled length: {n_timesteps_ds}")
    return data_ds, n_timesteps_ds, stride

# --- Map Boundaries Back ---
def map_boundaries_to_original(segment_indices_dp, stride, n_timesteps_original):
    """Maps segment boundaries from downsampled scale back to original."""
    if stride == 1:
        if segment_indices_dp and segment_indices_dp[-1] != n_timesteps_original:
             segment_indices_dp[-1] = n_timesteps_original
        return segment_indices_dp
    segment_indices_orig = [int(round(idx * stride)) for idx in segment_indices_dp]
    if not segment_indices_orig: return []
    if segment_indices_orig[-1] > n_timesteps_original: segment_indices_orig[-1] = n_timesteps_original
    elif segment_indices_orig[-1] < n_timesteps_original:
         if len(segment_indices_dp) > 1:
             second_last_dp = segment_indices_dp[-2]
             if int(round(second_last_dp * stride)) < n_timesteps_original:
                  segment_indices_orig[-1] = n_timesteps_original
    segment_indices_orig = sorted(list(set(segment_indices_orig)))
    if segment_indices_orig[-1] != n_timesteps_original:
        segment_indices_orig[-1] = n_timesteps_original
        segment_indices_orig = sorted(list(set(segment_indices_orig)))
    return segment_indices_orig

# --- Algorithm 1: Piecewise Linear Approximation (PLA) via DP ---
def calculate_pla_segment_cost(segment_data):
    """Calculates the cost for PLA: sum of squared errors from the best fit line."""
    n_points, n_dims = segment_data.shape
    if n_points <= 1: return 0.0
    total_sse = 0.0
    time_indices = np.arange(n_points).reshape(-1, 1)
    X = np.hstack((np.ones((n_points, 1)), time_indices))
    for d in range(n_dims):
        y = segment_data[:, d]
        try:
            params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            # Check residuals format and rank before accessing
            if isinstance(residuals, (np.ndarray, list)) and len(residuals) > 0 and rank == X.shape[1]:
                 sse_d = residuals[0]
            elif rank == X.shape[1]: # Calculate manually if residuals not returned but fit is valid
                 y_pred = X @ params
                 sse_d = np.sum((y - y_pred)**2)
            else: # Handle rank deficiency or other issues
                 # print(f"Warning: Rank deficiency or issue in lstsq for dim {d}. Rank={rank}")
                 sse_d = np.sum((y - np.mean(y))**2) # Fallback to variance if line fit fails
            total_sse += sse_d
        except np.linalg.LinAlgError: return np.inf
    return total_sse

def find_optimal_segmentation_pla(mean_traj_dp, n_timesteps_dp, max_segments, lambda_penalty):
    """Finds the optimal PLA segmentation using dynamic programming."""
    n_dims = mean_traj_dp.shape[1]
    dp_start_time = time.time()
    print("Precomputing PLA segment costs (on DP data)...")
    cost_cache = np.full((n_timesteps_dp, n_timesteps_dp), np.inf)
    for i in range(n_timesteps_dp):
        cost_cache[i, i] = 0.0
        for j in range(i + 1, n_timesteps_dp):
            segment_data = mean_traj_dp[i : j + 1]
            cost_cache[i, j] = calculate_pla_segment_cost(segment_data)
    print(f"PLA segment costs precomputed in {time.time() - dp_start_time:.2f} seconds.")

    dp = np.full((n_timesteps_dp, max_segments + 1), np.inf)
    bp = np.full((n_timesteps_dp, max_segments + 1), -1, dtype=int)
    for t in range(n_timesteps_dp):
        if 0 <= t < cost_cache.shape[1]: dp[t, 1] = cost_cache[0, t]; bp[t, 1] = 0
        else: dp[t, 1] = np.inf; bp[t, 1] = -1

    print("Running PLA dynamic programming...")
    dp_fill_start_time = time.time()
    for m in range(2, max_segments + 1):
        for t in range(1, n_timesteps_dp):
            min_cost_for_t_m = np.inf
            best_prev_t_start_index = -1
            search_start_j = max(0, m - 2)
            for j in range(search_start_j, t):
                if j >= 0 and (j+1) < n_timesteps_dp:
                    cost_prev = dp[j, m - 1]; cost_last = cost_cache[j + 1, t]
                    if np.isfinite(cost_prev) and np.isfinite(cost_last):
                        current_cost = cost_prev + cost_last
                        if current_cost < min_cost_for_t_m:
                            min_cost_for_t_m = current_cost
                            best_prev_t_start_index = j + 1
            if best_prev_t_start_index != -1: dp[t, m] = min_cost_for_t_m; bp[t, m] = best_prev_t_start_index
    print(f"PLA DP table filled in {time.time() - dp_fill_start_time:.2f} seconds.")

    final_raw_costs = dp[n_timesteps_dp - 1, 1:]
    final_costs_with_penalty = final_raw_costs + lambda_penalty * np.arange(1, max_segments + 1)
    valid_indices = np.where(np.isfinite(final_costs_with_penalty))[0]
    if len(valid_indices) == 0:
        print(f"Warning: PLA DP - No valid segmentation. Checking M=1.")
        if np.isfinite(dp[n_timesteps_dp - 1, 1]): optimal_num_segments = 1; min_total_cost = dp[n_timesteps_dp - 1, 1] + lambda_penalty
        else: print("Error: PLA DP - Cannot find any valid segmentation."); return [0, n_timesteps_dp], -1, np.inf
    else:
        optimal_idx_in_valid = np.argmin(final_costs_with_penalty[valid_indices])
        optimal_num_segments = valid_indices[optimal_idx_in_valid] + 1
        min_total_cost = final_costs_with_penalty[valid_indices[optimal_idx_in_valid]]
    print(f"PLA - Optimal segments: {optimal_num_segments}, Min cost (incl penalty): {min_total_cost:.4f}")

    segment_boundaries_dp = [n_timesteps_dp]
    current_t, current_m = n_timesteps_dp - 1, optimal_num_segments
    while current_m > 0 and current_t >= 0:
         if current_t < bp.shape[0] and current_m < bp.shape[1]: start_of_last_segment = bp[current_t, current_m]
         else: start_of_last_segment = -1
         if start_of_last_segment == -1:
             if current_m == 1 and current_t >= 0 : start_of_last_segment = 0
             else: break
         segment_boundaries_dp.append(start_of_last_segment)
         current_t = start_of_last_segment - 1; current_m -= 1
         if current_t < 0 and current_m == 0: break
    final_segment_indices_dp = sorted(list(set(segment_boundaries_dp)))
    if 0 not in final_segment_indices_dp: final_segment_indices_dp.insert(0,0)
    if n_timesteps_dp not in final_segment_indices_dp: final_segment_indices_dp.append(n_timesteps_dp)
    final_segment_indices_dp = sorted(list(set(final_segment_indices_dp)))
    return final_segment_indices_dp, optimal_num_segments, min_total_cost

# --- Algorithm 2: PELT ---
def segment_with_pelt(data_orig, penalty, model='l2'):
    """Segments multivariate data using PELT."""
    print(f"\n--- Running PELT (Model: {model}, Penalty: {penalty}) ---")
    n_timesteps_orig, n_dims = data_orig.shape
    pelt_start_time = time.time()
    # PELT in ruptures can handle multivariate data directly for some models like 'l2'
    algo = rpt.Pelt(model=model).fit(data_orig)
    try:
        # Predict breakpoints using the specified penalty
        # Result includes the end point (n_timesteps_orig)
        bkps = algo.predict(pen=penalty)
        # Ensure start point 0 is included
        final_boundaries = sorted(list(set([0] + bkps)))
    except Exception as e:
        print(f"  Error running PELT: {e}")
        final_boundaries = [0, n_timesteps_orig] # Fallback

    print(f"PELT finished in {time.time() - pelt_start_time:.2f} seconds.")
    print(f"PELT detected {len(final_boundaries)-1} segments.")
    return final_boundaries

# --- Algorithm 3: HMM ---
def segment_with_hmm(data_orig, n_states, n_iter, covar_type):
    """Segments multivariate data using HMM state changes."""
    print(f"\n--- Running HMM (States: {n_states}, Iter: {n_iter}, Covar: {covar_type}) ---")
    n_timesteps_orig, n_dims = data_orig.shape
    hmm_start_time = time.time()
    try:
        model = hmm.GaussianHMM(n_components=n_states, covariance_type=covar_type, n_iter=n_iter, random_state=42, tol=1e-3) # Added tol
        model.fit(data_orig)
        if hasattr(model.monitor_, 'converged') and not model.monitor_.converged:
             print("  Warning: HMM fitting did not converge according to tol.")
        # If the attribute doesn't exist, we might assume it converged or reached max_iter
        elif not hasattr(model.monitor_, 'converged'):
             print("  Note: Could not check HMM convergence status via monitor_.converged attribute.")
        state_sequence = model.predict(data_orig)
        change_points = np.where(np.diff(state_sequence) != 0)[0] + 1
        final_boundaries = sorted(list(set([0] + change_points.tolist() + [n_timesteps_orig])))
        print(f"HMM finished in {time.time() - hmm_start_time:.2f} seconds.")
        print(f"HMM detected {len(final_boundaries)-1} segments based on state changes.")
        return final_boundaries
    except Exception as e:
        print(f"  Error running HMM: {e}")
        return [0, n_timesteps_orig]

# --- NEW: Plotting Function for Comparison ---
def plot_comparison_segmentation(mean_traj_orig, boundaries_dict, feature_names, title="Segmentation Comparison"):
    """
    Plots the mean trajectory and segmentation boundaries from multiple algorithms.

    Args:
        mean_traj_orig (np.array): The original mean trajectory (N x D).
        boundaries_dict (dict): Dictionary where keys are algorithm names
                                and values are lists of boundary indices (original scale).
                                e.g., {'PLA': [0, 100, 250], 'PELT': [0, 110, 250]}
        feature_names (list): List of names for each dimension.
        title (str): Overall title for the plot.
    """
    n_timesteps_orig, n_dims = mean_traj_orig.shape
    time_axis_orig = np.arange(n_timesteps_orig)

    # Define colors and linestyles for different algorithms
    # Add more if needed
    styles = {
        'PLA-DP': {'color': 'red', 'linestyle': '--', 'linewidth': 1.5},
        'PELT': {'color': 'blue', 'linestyle': ':', 'linewidth': 2.0},
        'HMM': {'color': 'green', 'linestyle': '-.', 'linewidth': 1.5},
        # Add your method's style here if comparing
        'Tube-DP': {'color': 'purple', 'linestyle': '-', 'linewidth': 1.0}
    }

    n_cols = min(n_dims, 3)
    n_rows = math.ceil(n_dims / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=16)
    axs_flat = axs.flatten()
    sns.set_style("whitegrid")

    plotted_algos = set() # To control legend entries

    for d in range(n_dims):
        ax = axs_flat[d]
        # Plot the mean trajectory
        ax.plot(time_axis_orig, mean_traj_orig[:, d], color='grey', linewidth=1.0, label='Mean Traj.' if d==0 else "")

        # Plot boundaries for each algorithm
        for algo_name, boundaries in boundaries_dict.items():
            style = styles.get(algo_name, {'color': 'black', 'linestyle': '-', 'linewidth': 1}) # Default style
            label_added = False
            for i, bkp in enumerate(boundaries):
                if i > 0 and bkp < n_timesteps_orig: # Don't plot 0 or end boundary
                    # Add label only once per algorithm per plot group
                    label = algo_name if algo_name not in plotted_algos else ""
                    ax.axvline(x=bkp, label=label, **style)
                    if label: plotted_algos.add(algo_name); label_added=True
            # If algo had no internal boundaries, maybe add label differently? (optional)

        ax.set_ylabel(feature_names[d])
        ax.grid(True, linestyle=':')
        if d==0: # Add legend only to the first plot
             ax.legend(loc='best', fontsize='medium')

    # Hide unused subplots
    for d_unused in range(n_dims, n_rows * n_cols):
        axs_flat[d_unused].set_visible(False)

    # Set common X label
    for ax_bottom in axs[-1, :]:
         if ax_bottom.get_visible():
              ax_bottom.set_xlabel('Time Step (Original Aligned Scale)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    overall_start_time = time.time()
    # 1. Load data
    print("--- Loading Data ---")
    all_needed_columns = list(set(ALIGNMENT_BASE_FEATURE_COLS + SEGMENTATION_FEATURE_COLS))
    original_trajs_df, loaded_files = load_selected_data(
        PARENT_FOLDER_PATH, load_all=LOAD_ALL_FILES, file_list=FILE_LIST, feature_columns=all_needed_columns
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
    if not valid_aligned_trajs_df: exit("Exiting: Alignment failed.")

    # 3. Calculate Mean Trajectory using ONLY segmentation features
    print("\n--- Calculating Mean Trajectory ---")
    mean_traj_orig, n_timesteps_orig = calculate_mean_trajectory(valid_aligned_trajs_df, SEGMENTATION_FEATURE_COLS)
    if mean_traj_orig is None or n_timesteps_orig <= 1: exit("Exiting: Failed to calculate mean trajectory or not enough time steps.")
    n_dims = mean_traj_orig.shape[1]

    # --- Run Segmentation Algorithms ---
    results_boundaries = {} # Store results

    # --- 1. PLA via DP ---
    print("\n--- Running PLA Segmentation (DP) ---")
    mean_traj_dp, n_timesteps_dp, stride_dp = downsample_data(mean_traj_orig, MAX_DP_LENGTH)
    pla_segment_indices_dp, pla_optimal_num_segments, pla_min_total_cost = find_optimal_segmentation_pla(
        mean_traj_dp, n_timesteps_dp, MAX_SEGMENTS, LAMBDA_PENALTY_PLA
    )
    if pla_optimal_num_segments != -1:
        pla_segment_indices_orig = map_boundaries_to_original(pla_segment_indices_dp, stride_dp, n_timesteps_orig)
        results_boundaries['PLA-DP'] = pla_segment_indices_orig
        print(f"\nPLA-DP Results:")
        print(f"  Optimal Segments: {pla_optimal_num_segments}")
        print(f"  Boundaries (Original Scale): {pla_segment_indices_orig}")
    else: print("\nPLA-DP Segmentation Failed.")

    # --- 2. PELT ---
    pelt_segment_indices_orig = segment_with_pelt(mean_traj_orig, penalty=PELT_PENALTY, model=PELT_MODEL)
    results_boundaries['PELT'] = pelt_segment_indices_orig
    print(f"\nPELT Results:")
    print(f"  Boundaries (Original Scale): {pelt_segment_indices_orig}")

    # --- 3. HMM ---
    hmm_segment_indices_orig = segment_with_hmm(mean_traj_orig, n_states=HMM_N_STATES, n_iter=HMM_N_ITER, covar_type=HMM_COVAR_TYPE)
    results_boundaries['HMM'] = hmm_segment_indices_orig
    print(f"\nHMM Results:")
    print(f"  Boundaries (Original Scale): {hmm_segment_indices_orig}")

    # --- Plot Comparison ---
    print("\n--- Plotting Comparison Results ---")
    # Ensure feature names match the columns used for the mean trajectory
    plot_comparison_segmentation(
        mean_traj_orig,
        results_boundaries,
        SEGMENTATION_FEATURE_COLS, # Pass the list of feature names
        title="Segmentation Algorithm Comparison"
    )

    print(f"\n--- Comparison Script Finished in {time.time() - overall_start_time:.2f} seconds ---")

