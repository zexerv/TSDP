import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.linalg import solve # For solving linear system for weights
from scipy.interpolate import interp1d # For basis functions if needed
import seaborn as sns

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button'

# --- File Loading Config ---
LOAD_ALL_FILES = True
FILE_LIST = [] # Ignored if LOAD_ALL_FILES is True

# --- Feature Config ---
# Features for DTW distance calculation (usually position)
DTW_FEATURE_COLS = ['tx', 'ty', 'tz']
# Features to learn with ProMP (Position is required)
PROMP_POS_COLS = ['tx', 'ty', 'tz']
# Optional: Include Rotation features for ProMP
INCLUDE_ROTATION_PROMP = False
PROMP_ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

# --- ProMP Config ---
N_BASIS_FUNCTIONS = 10   # Number of basis functions
BASIS_FUNCTION_TYPE = 'atan' # Options: 'gaussian', 'atan'
# Factor controlling width/steepness of basis functions
BASIS_SCALE_FACTOR = 0.05 # Adjusted for atan, experiment with this value
# ***** NEW: Threshold for basis function centers *****
BASIS_CENTER_THRESHOLD = 0.0 # e.g., 0.05 means exclude first and last 5% of time
REGULARIZATION = 1e-6    # Regularization for solving linear system for weights

# --- Determine ProMP Features ---
# Define base column lists globally for clarity
POS_COLS = ['tx', 'ty', 'tz']
ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

PROMP_FEATURE_COLS = POS_COLS[:] # Start with position
if INCLUDE_ROTATION_PROMP:
    PROMP_FEATURE_COLS.extend(PROMP_ROT_COLS)

N_PROMP_FEATURES = len(PROMP_FEATURE_COLS)

print(f"Using features for DTW: {DTW_FEATURE_COLS}")
print(f"ProMP using {N_PROMP_FEATURES} features: {PROMP_FEATURE_COLS}")
print(f"Using basis function type: {BASIS_FUNCTION_TYPE}")
print(f"Basis center threshold: {BASIS_CENTER_THRESHOLD * 100:.1f}%")


# --- Data Loading ---
def load_selected_data(parent_folder, load_all=True, file_list=None):
    """Loads specified trajectories (all pose columns) from CSV files."""
    all_trajectories = []
    loaded_filenames = []
    required_columns = POS_COLS + ROT_COLS # Load all initially

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
            full_trajectory_data = df[required_columns].values
            if np.isnan(full_trajectory_data).any(): print(f"Warning: NaN values found in {file_path}. Skipping."); continue
            if full_trajectory_data.shape[0] == 0: print(f"Warning: Trajectory empty in {file_path}. Skipping."); continue
            all_trajectories.append(full_trajectory_data)
            loaded_filenames.append(filename)
        except FileNotFoundError: print(f"Warning: File not found - {file_path}")
        except KeyError as e: print(f"Error in {file_path}: {e}")
        except Exception as e: print(f"Error loading/processing {file_path}: {e}")

    if not all_trajectories: print("Warning: No valid trajectory data loaded.")
    return all_trajectories, loaded_filenames

# --- DTW Alignment ---
def align_trajectories_dtw(trajectories, dtw_feature_indices):
    """Aligns trajectories using DTW based on specified features."""
    if not trajectories: return [], None, -1
    lengths = [len(t) for t in trajectories]
    if not lengths: return [], None, -1
    reference_index = np.argmax(lengths)
    reference_trajectory_full = trajectories[reference_index]
    ref_len = len(reference_trajectory_full)
    print(f"\nUsing trajectory {reference_index} (length {ref_len}) as reference for DTW.")
    aligned_trajectories = []
    for i, traj_full in enumerate(trajectories):
        if len(traj_full) == 0:
             aligned_trajectories.append(np.zeros_like(reference_trajectory_full)); continue
        if i == reference_index:
            aligned_trajectories.append(reference_trajectory_full.copy()); continue
        traj_dtw_features = traj_full[:, dtw_feature_indices]
        ref_dtw_features = reference_trajectory_full[:, dtw_feature_indices]
        distance, path = fastdtw(ref_dtw_features, traj_dtw_features, dist=euclidean)
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
        aligned_trajectories.append(warped_traj_full)
    return aligned_trajectories, reference_trajectory_full, reference_index

# --- ProMP Implementation ---

# ***** MODIFIED: Basis Function Generation with Threshold *****
def generate_basis_functions(basis_type, n_basis, n_time_steps, scale_factor=1.0, center_threshold=0.0):
    """
    Generates basis functions phi(t) of a specified type.
    Centers are placed between center_threshold and 1-center_threshold.
    """
    if n_basis <= 0:
        return np.ones((n_time_steps, 1)), np.array([0.0]) # Constant basis

    # Define the range for placing centers based on threshold
    center_start = center_threshold
    center_end = 1.0 - center_threshold
    if center_start >= center_end: # Ensure valid range if threshold is >= 0.5
        print(f"Warning: Basis center threshold ({center_threshold}) is too large. Placing center at 0.5.")
        center_start = 0.5
        center_end = 0.5

    # Calculate centers within the thresholded range
    if n_basis == 1:
        centers = np.array([0.5]) # Place single basis function in the middle
    else:
        centers = np.linspace(center_start, center_end, n_basis)

    time = np.linspace(0, 1, n_time_steps)
    phi = np.zeros((n_time_steps, n_basis))

    # Calculate characteristic width/scale based on spacing of centers
    if n_basis > 1:
        # Use spacing within the thresholded range
        width = (centers[1] - centers[0]) if len(centers) > 1 else (center_end - center_start)
    else:
        width = (center_end - center_start) # Width is the range itself if only one center
    width = max(width, 1e-6) # Ensure positive

    if basis_type == 'gaussian':
        variance = (width * scale_factor)**2
        variance = max(variance, 1e-9) # Ensure variance is positive
        for i in range(n_basis):
            phi[:, i] = np.exp(-0.5 * (time - centers[i])**2 / variance)
    elif basis_type == 'atan':
        # Scale factor now controls steepness (smaller = steeper)
        steepness_param = max(width * scale_factor, 1e-6) # w in atan((t-c)/w)
        for i in range(n_basis):
            # Shifted and scaled atan, range [0, 1]
            phi[:, i] = (1.0 / math.pi) * np.arctan((time - centers[i]) / steepness_param) + 0.5
    else:
        raise ValueError(f"Unknown basis function type: {basis_type}")

    # Normalize basis functions row-wise (sum to 1 at each time step)
    row_sums = np.sum(phi, axis=1, keepdims=True) + 1e-9 # Add epsilon
    phi_normalized = phi / row_sums

    return phi_normalized, centers

def calculate_promp_weights(trajectory, basis_functions, regularization=1e-6):
    """Calculates the weights w for a single trajectory using linear regression."""
    phi = basis_functions
    phi_T_phi = phi.T @ phi
    regularized_term = regularization * np.identity(phi_T_phi.shape[0])
    try:
        weights = solve(phi_T_phi + regularized_term, phi.T @ trajectory, assume_a='pos')
    except np.linalg.LinAlgError:
        print("Warning: Linear system for weights ill-conditioned. Using pseudo-inverse.")
        weights = np.linalg.pinv(phi) @ trajectory
    return weights

# ***** MODIFIED: Calls generate_basis_functions with threshold *****
def learn_promp_distribution(aligned_trajectories, promp_feature_indices,
                             n_basis, basis_type, basis_scale_factor,
                             basis_center_threshold, regularization):
    """Learns the ProMP weight distribution (mean and covariance) from demonstrations."""
    if not aligned_trajectories: return None, None, None, 0, []
    ref_len = len(aligned_trajectories[0])
    n_promp_features = len(promp_feature_indices)
    n_demos = len(aligned_trajectories)
    if ref_len == 0 or n_promp_features == 0 or n_demos == 0: return None, None, None, 0, []

    # 1. Generate Basis Functions with threshold
    basis_functions, _ = generate_basis_functions(
        basis_type, n_basis, ref_len, basis_scale_factor, basis_center_threshold
    )

    # 2. Calculate weights for each demonstration
    all_weights, valid_demo_indices = [], []
    for i, traj_full in enumerate(aligned_trajectories):
        if len(traj_full) != ref_len: continue
        traj_promp_features = traj_full[:, promp_feature_indices]
        weights = calculate_promp_weights(traj_promp_features, basis_functions, regularization)
        all_weights.append(weights.flatten())
        valid_demo_indices.append(i)
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
    print(f"\nProMP learning complete. Mean weights shape: {mean_weights.shape}, Cov weights shape: {cov_weights_flat.shape}")
    return mean_weights, cov_weights_flat, basis_functions, ref_len, valid_demo_indices

def reconstruct_trajectory(basis_functions, weights):
    """Reconstructs a trajectory given basis functions and weights."""
    return basis_functions @ weights

def sample_promp_trajectory(mean_weights_flat, cov_weights_flat, basis_functions, n_promp_features):
    """Samples a new trajectory from the ProMP distribution."""
    n_basis = basis_functions.shape[1]
    try:
        sampled_weights_flat = np.random.multivariate_normal(mean_weights_flat, cov_weights_flat)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix not positive definite. Using mean weights.")
        sampled_weights_flat = mean_weights_flat
    sampled_weights = sampled_weights_flat.reshape(n_basis, n_promp_features)
    sampled_trajectory = reconstruct_trajectory(basis_functions, sampled_weights)
    return sampled_trajectory

def calculate_promp_variance(basis_functions, cov_weights_flat, n_promp_features):
    """Calculates the variance along the trajectory for each feature dimension."""
    ref_len, n_basis = basis_functions.shape
    variance_trajectory = np.zeros((ref_len, n_promp_features))
    for d in range(n_promp_features):
        start_idx, end_idx = d * n_basis, (d + 1) * n_basis
        if end_idx > cov_weights_flat.shape[0] or end_idx > cov_weights_flat.shape[1]: continue
        cov_wd = cov_weights_flat[start_idx:end_idx, start_idx:end_idx]
        for t in range(ref_len):
            phi_t = basis_functions[t, :]
            try: variance_trajectory[t, d] = phi_t @ cov_wd @ phi_t.T
            except Exception as e: print(f"Warn: Var calc error d={d}, t={t}: {e}"); variance_trajectory[t, d] = 0
    return variance_trajectory


# --- Visualization ---
def plot_promp_results(aligned_trajectories, valid_demo_indices, promp_feature_indices, mean_trajectory, variance_trajectory, sampled_trajectories, time_vector):
    """Visualizes the ProMP mean, variance, samples, and original demonstrations."""
    n_plot_dims = min(mean_trajectory.shape[1], 3) # Plot up to 3 dimensions
    feature_labels = PROMP_FEATURE_COLS[:n_plot_dims]
    sns.set_theme(style="whitegrid")
    demo_color = "darkgrey"; demo_alpha = 0.5; demo_linewidth = 1.5
    mean_color = "black"; variance_color = "black"; sample_color = "blue"
    fig, axs = plt.subplots(n_plot_dims, 1, figsize=(12, 4 * n_plot_dims), sharex=True)
    if n_plot_dims == 1: axs = [axs] # Make indexable
    fig.suptitle('ProMP Learning Results', fontsize=16)

    # Plot original aligned demonstrations
    plotted_demo_label = False
    for i, demo_idx in enumerate(valid_demo_indices):
         if demo_idx < len(aligned_trajectories):
             traj_full = aligned_trajectories[demo_idx]
             traj_promp = traj_full[:, promp_feature_indices]
             for j in range(n_plot_dims):
                 label = 'Aligned Demos' if not plotted_demo_label and j==0 else ""
                 axs[j].plot(time_vector, traj_promp[:, j], color=demo_color, alpha=demo_alpha, linewidth=demo_linewidth, label=label)
                 if label: plotted_demo_label = True

    # Plot mean trajectory and variance
    std_dev_trajectory = np.sqrt(np.maximum(variance_trajectory, 0))
    plotted_std_label = False
    for j in range(n_plot_dims):
        axs[j].plot(time_vector, mean_trajectory[:, j], color=mean_color, linewidth=2.5, label='ProMP Mean')
        label_std = 'ProMP Std Dev' if not plotted_std_label else ""
        axs[j].fill_between(time_vector, mean_trajectory[:, j] - std_dev_trajectory[:, j],
                            mean_trajectory[:, j] + std_dev_trajectory[:, j],
                            color=variance_color, alpha=0.2, label=label_std)
        if label_std: plotted_std_label = True
        axs[j].set_ylabel(feature_labels[j])
        axs[j].grid(True, linestyle=':')

    # Plot sampled trajectories
    plotted_sample_label = False
    for s, sampled_traj in enumerate(sampled_trajectories):
         for j in range(n_plot_dims):
             label_sample = 'Samples' if not plotted_sample_label and j==0 else ""
             axs[j].plot(time_vector, sampled_traj[:, j], color=sample_color, linestyle='--', linewidth=1, alpha=0.6, label=label_sample)
             if label_sample: plotted_sample_label = True

    axs[-1].set_xlabel('Time Step (Aligned)')
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # Define base column lists globally
    POS_COLS = ['tx', 'ty', 'tz']
    ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

    # 1. Load data
    all_features_to_load = POS_COLS + ROT_COLS
    original_trajs_full, loaded_files = load_selected_data(
        PARENT_FOLDER_PATH, load_all=LOAD_ALL_FILES, file_list=FILE_LIST
    )
    if not original_trajs_full: exit("Exiting: No trajectories loaded.")

    # 2. Perform DTW Alignment
    try: dtw_col_indices = [all_features_to_load.index(col) for col in DTW_FEATURE_COLS]
    except ValueError as e: exit(f"Error: DTW feature column not found: {e}")
    aligned_trajs_full, ref_traj_full, ref_idx = align_trajectories_dtw(original_trajs_full, dtw_col_indices)
    valid_aligned_trajs_full = [t for t in aligned_trajs_full if len(t) > 0 and np.any(t)]
    if not valid_aligned_trajs_full: exit("\nNo valid aligned trajectories after DTW. Exiting.")
    print(f"\nSuccessfully aligned {len(valid_aligned_trajs_full)} trajectories.")
    ref_len = len(valid_aligned_trajs_full[0])

    # 3. Learn ProMP Distribution
    try: promp_col_indices = [all_features_to_load.index(col) for col in PROMP_FEATURE_COLS]
    except ValueError as e: exit(f"Error: ProMP feature column not found: {e}")

    # ***** MODIFIED: Pass basis center threshold *****
    mean_w, cov_w_flat, basis_funcs, _, valid_demo_indices = learn_promp_distribution(
        valid_aligned_trajs_full,
        promp_col_indices,
        N_BASIS_FUNCTIONS,
        BASIS_FUNCTION_TYPE,
        BASIS_SCALE_FACTOR,
        BASIS_CENTER_THRESHOLD, # Pass the new parameter
        REGULARIZATION
    )

    if mean_w is None: exit("ProMP learning failed.")

    # 4. Generate Mean Trajectory and Variance
    mean_promp_traj = reconstruct_trajectory(basis_funcs, mean_w)
    variance_promp_traj = calculate_promp_variance(basis_funcs, cov_w_flat, N_PROMP_FEATURES)

    # 5. Sample New Trajectories
    n_samples_to_plot = 30
    sampled_promp_trajs = []
    print(f"\nSampling {n_samples_to_plot} trajectories from ProMP...")
    mean_w_flat = mean_w.flatten()
    for _ in range(n_samples_to_plot):
        sampled_trajs = sample_promp_trajectory(mean_w_flat, cov_w_flat, basis_funcs, N_PROMP_FEATURES)
        sampled_promp_trajs.append(sampled_trajs)

    # 6. Visualize Results
    print("Generating visualizations...")
    aligned_time_vec = np.arange(ref_len)
    plot_promp_results(valid_aligned_trajs_full, valid_demo_indices, promp_col_indices, mean_promp_traj, variance_promp_traj, sampled_promp_trajs, aligned_time_vec)

    # Optional: Print learned parameters
    print("\nLearned ProMP Parameters:")
    print(f"Features used: {PROMP_FEATURE_COLS}")
    print(f"Basis function type: {BASIS_FUNCTION_TYPE}")
    print(f"Number of basis functions: {N_BASIS_FUNCTIONS}")
    # print("Mean Weights (flat):", mean_w_flat)
    # print("Covariance Weights (flat):", cov_w_flat)

