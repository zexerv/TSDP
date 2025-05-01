import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.linalg import solve # For solving linear system for weights
from scipy.signal import find_peaks # For finding peaks in variance change
import seaborn as sns

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button'

# --- File Loading Config ---
LOAD_ALL_FILES = True
FILE_LIST = [] # Ignored if LOAD_ALL_FILES is True

# --- Preprocessing Config ---
APPLY_DTW_ALIGNMENT = True # Apply DTW before ProMP?

# --- Feature Config ---
# Features for DTW distance calculation (if APPLY_DTW_ALIGNMENT is True)
DTW_FEATURE_COLS = ['tx', 'ty', 'tz']
# Features to learn with ProMP
PROMP_POS_COLS = ['tx', 'ty', 'tz']
INCLUDE_ROTATION_PROMP = False
PROMP_ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

# --- ProMP Config ---
N_BASIS_FUNCTIONS = 10
BASIS_FUNCTION_TYPE = 'gaussian' # 'gaussian' or 'sigmoid'
BASIS_SCALE_FACTOR = 1.5
REGULARIZATION = 1e-6

# --- Segmentation Config ---
# How many segments to find (n_segments = n_changepoints + 1)
# This is used to select the top N peaks in variance change.
N_SEGMENTS_TARGET = 4 # Corresponds to N_STATES in HMM
# Threshold and distance for find_peaks on variance change rate
PEAK_THRESHOLD = 0.01 # Minimum height of a peak (adjust based on variance scale)
PEAK_MIN_DISTANCE = 10 # Minimum steps between peaks (adjust based on trajectory length)


# --- Determine ProMP Features ---
POS_COLS = ['tx', 'ty', 'tz']
ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
all_load_features = POS_COLS + ROT_COLS

PROMP_FEATURE_COLS = PROMP_POS_COLS[:]
if INCLUDE_ROTATION_PROMP:
    PROMP_FEATURE_COLS.extend(PROMP_ROT_COLS)
N_PROMP_FEATURES = len(PROMP_FEATURE_COLS)

print(f"ProMP using {N_PROMP_FEATURES} features: {PROMP_FEATURE_COLS}")
if APPLY_DTW_ALIGNMENT:
    print(f"Using features for DTW: {DTW_FEATURE_COLS}")


# --- Data Loading ---
def load_selected_data(parent_folder, load_all=True, file_list=None):
    """Loads specified trajectories (all pose columns) from CSV files."""
    all_trajectories = []
    loaded_filenames = []
    required_columns = list(set(all_load_features + DTW_FEATURE_COLS))

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
            all_trajectories.append(df[required_columns]) # Store DataFrame
            loaded_filenames.append(filename)
        except FileNotFoundError: print(f"Warning: File not found - {file_path}")
        except KeyError as e: print(f"Error in {file_path}: {e}")
        except Exception as e: print(f"Error loading/processing {file_path}: {e}")

    if not all_trajectories: print("Warning: No valid trajectory data loaded.")
    return all_trajectories, loaded_filenames

# --- DTW Alignment ---
def align_trajectories_dtw(trajectories_df_list, dtw_feature_cols):
    """Aligns trajectories using DTW based on specified features."""
    if not trajectories_df_list: return [], None, -1
    trajectories_dtw_np = [df[dtw_feature_cols].values for df in trajectories_df_list]
    trajectories_full_np = [df.values for df in trajectories_df_list]
    required_columns = trajectories_df_list[0].columns.tolist()

    lengths = [len(t) for t in trajectories_dtw_np]
    if not lengths: return [], None, -1
    reference_index = np.argmax(lengths)
    reference_trajectory_dtw = trajectories_dtw_np[reference_index]
    reference_trajectory_full = trajectories_full_np[reference_index]
    ref_len = len(reference_trajectory_dtw)
    print(f"\nUsing trajectory {reference_index} (length {ref_len}) as reference for DTW.")

    aligned_trajectories_np = []
    for i, traj_dtw in enumerate(trajectories_dtw_np):
        traj_full = trajectories_full_np[i]
        if len(traj_dtw) == 0:
             aligned_trajectories_np.append(np.zeros_like(reference_trajectory_full)); continue
        if i == reference_index:
            aligned_trajectories_np.append(reference_trajectory_full.copy()); continue

        distance, path = fastdtw(reference_trajectory_dtw, traj_dtw, dist=euclidean)
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

# --- ProMP Implementation ---
def generate_basis_functions(basis_type, n_basis, n_time_steps, scale_factor=1.0):
    """Generates basis functions phi(t) of a specified type."""
    if n_basis <= 0: return np.ones((n_time_steps, 1)), np.array([0.0])
    if n_basis == 1: centers = np.array([0.5])
    else: centers = np.linspace(0, 1, n_basis)
    time = np.linspace(0, 1, n_time_steps)
    phi = np.zeros((n_time_steps, n_basis))
    width = (centers[1] - centers[0]) if n_basis > 1 else 1.0
    width = max(width, 1e-6)

    if basis_type == 'gaussian':
        variance = max((width * scale_factor)**2, 1e-9)
        for i in range(n_basis): phi[:, i] = np.exp(-0.5 * (time - centers[i])**2 / variance)
    elif basis_type == 'sigmoid':
        steepness_param = max(width * scale_factor, 1e-6)
        for i in range(n_basis): phi[:, i] = 0.5 * (np.tanh((time - centers[i]) / steepness_param) + 1.0)
    else: raise ValueError(f"Unknown basis type: {basis_type}")

    row_sums = np.sum(phi, axis=1, keepdims=True) + 1e-9
    return phi / row_sums, centers

def calculate_promp_weights(trajectory, basis_functions, regularization=1e-6):
    """Calculates the weights w for a single trajectory."""
    phi = basis_functions; phi_T_phi = phi.T @ phi
    reg_term = regularization * np.identity(phi_T_phi.shape[0])
    try: return solve(phi_T_phi + reg_term, phi.T @ trajectory, assume_a='pos')
    except np.linalg.LinAlgError: return np.linalg.pinv(phi) @ trajectory

# Inside the ProMP script (e.g., promp_script_flexible_basis)

# ***** Ensure this function definition replaces the old one *****
def learn_promp_distribution(aligned_trajectories_df, promp_feature_cols,
                             n_basis, basis_type, basis_scale_factor, regularization):
    """Learns the ProMP weight distribution (mean and covariance) from demonstrations."""
    # Input is now explicitly list of DataFrames
    if not aligned_trajectories_df: return None, None, None, 0, []
    # Use length from the first valid DataFrame
    valid_dfs = [df for df in aligned_trajectories_df if not df.empty]
    if not valid_dfs: return None, None, None, 0, []
    ref_len = len(valid_dfs[0])

    n_promp_features = len(promp_feature_cols)
    n_demos = len(valid_dfs) # Count only valid dfs
    if ref_len == 0 or n_promp_features == 0 or n_demos == 0: return None, None, None, 0, []

    # 1. Generate Basis Functions
    print(f"Generating {n_basis} '{basis_type}' basis functions...")
    basis_functions, _ = generate_basis_functions(
        basis_type, n_basis, ref_len, basis_scale_factor
        # Removed basis_center_threshold argument if you reverted that change
    )

    # 2. Calculate weights for each demonstration
    all_weights, valid_demo_indices = [], []
    original_indices = [i for i, df in enumerate(aligned_trajectories_df) if not df.empty] # Track original indices

    for i, traj_df in enumerate(valid_dfs): # Iterate through valid DataFrames
        if len(traj_df) != ref_len:
             print(f"Warn: Skipping demo index {original_indices[i]}, length mismatch.")
             continue
        try:
            # ***** FIX: Select columns by name, then get values *****
            # Ensure promp_feature_cols contains the correct column names (e.g., ['tx', 'ty', 'tz'])
            traj_promp_features = traj_df[promp_feature_cols].values
            # ***** END FIX *****

            # Check if selection resulted in correct shape
            if traj_promp_features.shape != (ref_len, n_promp_features):
                 print(f"Warn: Skipping demo index {original_indices[i]}, feature shape mismatch after selection. Expected ({ref_len}, {n_promp_features}), got {traj_promp_features.shape}")
                 continue

            weights = calculate_promp_weights(traj_promp_features, basis_functions, regularization)
            all_weights.append(weights.flatten())
            valid_demo_indices.append(original_indices[i]) # Store original index
        except KeyError as e:
            # This error occurs if a column name in promp_feature_cols is not in traj_df
            print(f"Warn: Skipping demo index {original_indices[i]}, missing feature column: {e}")
        except Exception as e:
            print(f"Warn: Skipping demo index {original_indices[i]}, error calc weights: {e}")


    if not all_weights: print("Error: Could not calculate weights."); return None, None, None, 0, []
    all_weights = np.array(all_weights)

    # 3. Estimate mean and covariance of weights
    if all_weights.shape[0] < 2:
         print("Warning: Need >= 2 demos for covariance. Covariance set to zero.")
         mean_weights_flat = np.mean(all_weights, axis=0)
         cov_weights_flat = np.zeros((all_weights.shape[1], all_weights.shape[1]))
    else:
        mean_weights_flat = np.mean(all_weights, axis=0)
        cov_weights_flat = np.cov(all_weights, rowvar=False) + np.identity(all_weights.shape[1]) * 1e-6
    mean_weights = mean_weights_flat.reshape(n_basis, n_promp_features)
    print(f"\nProMP learning complete. Used {len(valid_demo_indices)} valid demos.")
    return mean_weights, cov_weights_flat, basis_functions, ref_len, valid_demo_indices

# Include other necessary functions like generate_basis_functions, calculate_promp_weights etc.
# from the correct ProMP script version you ar

def calculate_promp_variance(basis_functions, cov_weights_flat, n_promp_features):
    """Calculates the variance along the trajectory."""
    ref_len, n_basis = basis_functions.shape
    variance_trajectory = np.zeros((ref_len, n_promp_features))
    for d in range(n_promp_features):
        start_idx, end_idx = d * n_basis, (d + 1) * n_basis
        if end_idx > cov_weights_flat.shape[0] or end_idx > cov_weights_flat.shape[1]: continue
        cov_wd = cov_weights_flat[start_idx:end_idx, start_idx:end_idx]
        for t in range(ref_len):
            phi_t = basis_functions[t, :]
            try: variance_trajectory[t, d] = phi_t @ cov_wd @ phi_t.T
            except Exception: variance_trajectory[t, d] = 0
    return variance_trajectory

# --- ProMP Segmentation (Variance Change Heuristic) ---
def segment_promp_variance(variance_trajectory, n_segments, peak_threshold, min_distance):
    """Segments trajectory based on peaks in the rate of change of variance."""
    if variance_trajectory is None or variance_trajectory.shape[0] < 3:
        return [] # Cannot calculate change or find peaks

    # Calculate summed variance across features
    total_variance = np.sum(variance_trajectory, axis=1)

    # Calculate rate of change of total variance (absolute value)
    variance_change_rate = np.abs(np.diff(total_variance, prepend=total_variance[0]))

    # Smooth the change rate slightly (optional, can help reduce noise)
    # window_size = 5
    # if len(variance_change_rate) >= window_size:
    #     variance_change_rate = np.convolve(variance_change_rate, np.ones(window_size)/window_size, mode='same')


    # Find peaks in the change rate
    # We want n_segments-1 change points
    n_changepoints = n_segments - 1
    if n_changepoints <= 0:
        return []

    peaks, properties = find_peaks(variance_change_rate, height=peak_threshold, distance=min_distance)

    if len(peaks) == 0:
        print("Warning: No significant peaks found in variance change rate.")
        return []

    # Select the top N peaks based on prominence or height
    if len(peaks) > n_changepoints:
        peak_heights = properties['peak_heights']
        top_peak_indices = np.argsort(peak_heights)[-n_changepoints:]
        change_points = sorted(peaks[top_peak_indices])
    else:
        change_points = sorted(peaks)

    # Return indices *before* the change (peaks indicate point of max change)
    # Adjusting index slightly might be needed depending on interpretation
    # For simplicity, return the peak index itself as the change point
    return change_points


# --- Visualization ---
def plot_promp_segmentation(trajectory_df, change_points, trajectory_filename, n_segments, feature_cols_to_plot):
    """Plots trajectory components and marks detected change points."""
    try:
        trajectory_data = trajectory_df[feature_cols_to_plot].values
    except KeyError:
        print(f"Warning: Plotting features {feature_cols_to_plot} not found in trajectory {trajectory_filename}. Skipping plot.")
        return

    n_points = len(trajectory_data)
    if n_points == 0: print(f"Skipping plot for {trajectory_filename} due to empty data."); return

    time_steps = np.arange(n_points)
    n_plot_dims = trajectory_data.shape[1]

    fig, axs = plt.subplots(n_plot_dims, 1, sharex=True, figsize=(12, 2.5 * n_plot_dims))
    if n_plot_dims == 1: axs = [axs] # Make indexable
    fig.suptitle(f'ProMP Segmentation: {trajectory_filename} ({n_segments} Segments)', fontsize=14)

    sns.set_style("whitegrid")
    colors = sns.color_palette("viridis", n_segments) # Color segments

    segment_boundaries = np.concatenate(([0], change_points, [n_points]))

    for i in range(n_plot_dims):
        axs[i].plot(time_steps, trajectory_data[:, i], label=f'{feature_cols_to_plot[i]}', color='black')
        axs[i].set_ylabel(f'{feature_cols_to_plot[i]}')

        # Color background based on segments
        plotted_segment_labels = set()
        for seg_idx in range(n_segments):
            start = segment_boundaries[seg_idx]
            end = segment_boundaries[seg_idx + 1] -1 # End index is inclusive for axvspan
            label = f'Segment {seg_idx}' if seg_idx not in plotted_segment_labels and i==0 else ""
            axs[i].axvspan(max(0, start - 0.5), min(n_points - 1, end + 0.5),
                           facecolor=colors[seg_idx], alpha=0.2, label=label, zorder=1)
            if label: plotted_segment_labels.add(seg_idx)

        # Mark change points with vertical lines
        if change_points is not None:
            for j, cp in enumerate(change_points):
                 label_cp = 'Change Point' if i == 0 and j == 0 else ""
                 axs[i].axvline(cp, color='red', linestyle='--', linewidth=1.5, label=label_cp, zorder=5)


    axs[-1].set_xlabel('Time Step (Aligned)')
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load data
    original_trajs_df, loaded_files = load_selected_data(
        PARENT_FOLDER_PATH, load_all=LOAD_ALL_FILES, file_list=FILE_LIST
    )
    if not original_trajs_df: exit("Exiting: No trajectories loaded.")

    # 2. Optional DTW Alignment
    if APPLY_DTW_ALIGNMENT:
        try: dtw_col_indices_ignored = [all_load_features.index(col) for col in DTW_FEATURE_COLS]
        except ValueError as e: exit(f"Error: DTW feature column not found: {e}")
        aligned_trajs_df, ref_idx = align_trajectories_dtw(original_trajs_df, DTW_FEATURE_COLS)
        if not aligned_trajs_df: exit("\nDTW alignment failed. Exiting.")
        print(f"\nSuccessfully aligned {len(aligned_trajs_df)} trajectories.")
        trajs_for_promp = aligned_trajs_df
    else:
        print("\nSkipping DTW alignment.")
        trajs_for_promp = original_trajs_df

    # Filter out empty dataframes just in case
    valid_trajs_for_promp = [df for df in trajs_for_promp if not df.empty]
    if not valid_trajs_for_promp: exit("No valid trajectories remain for ProMP.")


    # 3. Learn ProMP Distribution
    try: promp_col_indices = [all_load_features.index(col) for col in PROMP_FEATURE_COLS]
    except ValueError as e: exit(f"Error: ProMP feature column not found: {e}")

    mean_w, cov_w_flat, basis_funcs, ref_len, valid_demo_indices = learn_promp_distribution(
        valid_trajs_for_promp, # Use the potentially filtered list
        promp_col_indices,
        N_BASIS_FUNCTIONS,
        BASIS_FUNCTION_TYPE,
        BASIS_SCALE_FACTOR,
        REGULARIZATION
    )

    if mean_w is None: exit("ProMP learning failed.")

    # 4. Calculate Variance Trajectory (using all features modeled by ProMP)
    variance_promp_traj = calculate_promp_variance(basis_funcs, cov_w_flat, N_PROMP_FEATURES)

    # 5. Segment based on variance change
    print(f"\nSegmenting based on variance changes (target: {N_SEGMENTS_TARGET} segments)...")
    change_points = segment_promp_variance(variance_promp_traj, N_SEGMENTS_TARGET, PEAK_THRESHOLD, PEAK_MIN_DISTANCE)

    if change_points is not None:
        print(f"Detected change points (indices): {change_points}")

        # 6. Visualize Segmentation for the reference trajectory (or others)
        # Find the DF corresponding to the reference index among the valid ones
        # This logic assumes valid_demo_indices maps back to original_trajs_df correctly if DTW was applied
        # If DTW was skipped, ref_idx might not be directly usable.
        # For simplicity, plotting the first valid trajectory after potential alignment.
        if valid_trajs_for_promp:
             plot_promp_segmentation(valid_trajs_for_promp[0], change_points, f"{loaded_files[valid_demo_indices[0]] if valid_demo_indices else 'Trajectory 0'}", N_SEGMENTS_TARGET, POS_COLS)
        else:
             print("No valid trajectory to plot segmentation for.")

    else:
        print("Segmentation based on ProMP variance failed.")

