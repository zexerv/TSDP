import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import seaborn as sns
import bayesian_changepoint_detection as bcpd # Import BOCPD library
from bayesian_changepoint_detection import likelihoods # Explicitly import submodule if needed

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button'

# --- File Loading Config ---
LOAD_ALL_FILES = True
FILE_LIST = [] # Ignored if LOAD_ALL_FILES is True

# --- Preprocessing Config ---
APPLY_DTW_ALIGNMENT = True # Apply DTW before BOCPD?

# --- Feature Config ---
# Features for DTW distance calculation (if APPLY_DTW_ALIGNMENT is True)
DTW_FEATURE_COLS = ['tx', 'ty', 'tz']
# Features to use for Change Point Detection
CPD_POS_COLS = ['tx', 'ty', 'tz']
INCLUDE_ROTATION_CPD = False
CPD_ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

# --- BOCPD Algorithm Config ---
# Hazard function parameter lambda (expected run length) - Smaller means more sensitive to change
HAZARD_LAMBDA = 100 # Example value, TUNE THIS!
# Observation likelihood model parameters (assuming Gaussian for simplicity)
# These might need adjustment based on data scale and variance
# Prior parameters for the Gaussian likelihood (Normal-Gamma conjugate prior)
# See library documentation for details if needed. Using defaults often works okay.
ALPHA0 = 0.01 # Prior pseudo-count for precision (Gamma shape)
BETA0 = 0.0001 # Prior pseudo-sum-of-squares (Gamma rate)
KAPPA0 = 1.0   # Prior belief strength on mean
MU0 = 0.0      # Prior mean (set near expected data mean if known)

# Threshold for plotting change point probability
PROB_THRESHOLD = 0.5


# --- Determine CPD Features ---
POS_COLS = ['tx', 'ty', 'tz']
ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
all_load_features = POS_COLS + ROT_COLS

CPD_FEATURE_COLS = CPD_POS_COLS[:]
if INCLUDE_ROTATION_CPD:
    CPD_FEATURE_COLS.extend(CPD_ROT_COLS)
N_FEATURES_CPD = len(CPD_FEATURE_COLS)

print(f"BOCPD using {N_FEATURES_CPD} features: {CPD_FEATURE_COLS}")
if APPLY_DTW_ALIGNMENT:
    print(f"Using features for DTW: {DTW_FEATURE_COLS}")


# --- Data Loading (Identical to previous scripts) ---
def load_selected_data(parent_folder, load_all=True, file_list=None):
    """Loads specified trajectories (all pose columns) from CSV files."""
    all_trajectories = []
    loaded_filenames = []
    required_columns = list(set(all_load_features + DTW_FEATURE_COLS + CPD_FEATURE_COLS))

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


# --- DTW Alignment (Identical to previous scripts) ---
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

# --- Bayesian Online Change Point Detection ---
def detect_bocpd(trajectory_data, hazard_lambda, alpha0, beta0, kappa0, mu0):
    """
    Performs Bayesian Online Change Point Detection.

    Args:
        trajectory_data (np.ndarray): Trajectory data (n_points, n_features).
        hazard_lambda (float): Expected run length parameter for hazard function.
        alpha0, beta0, kappa0, mu0: Prior parameters for the Normal-Gamma model.

    Returns:
        np.ndarray: Array of run length probabilities R[t, r_t] for each time t.
                    Shape (n_points + 1, n_points + 1). R[t, r] is prob that run length is r at time t.
                    Returns None if detection fails.
    """
    if trajectory_data.shape[0] < 1:
        print("Warning: Trajectory too short for BOCPD.")
        return None, None # Return None for maxes too

    # Define the hazard function (constant rate)
    # Check if hazard_functions submodule exists, otherwise use direct access
    try:
        hazard_func = lambda r: bcpd.hazard_functions.constant_hazard(r, hazard_lambda)
    except AttributeError:
        print("Warning: bcpd.hazard_functions not found, trying direct access bcpd.constant_hazard")
        try:
             hazard_func = lambda r: bcpd.constant_hazard(r, hazard_lambda)
        except AttributeError:
             print("Error: Cannot find constant_hazard function.")
             return None, None


    n_points, n_features = trajectory_data.shape
    print(f"  Running BOCPD for {n_features} features individually...")

    # Store run length probabilities for each feature
    R_features = []
    maxes_list = [] # Store maxes for each dimension

    for dim in range(n_features):
        print(f"    Processing feature {dim+1}/{n_features}...")
        signal = trajectory_data[:, dim]

        # ***** FIX: Instantiate likelihood directly from bcpd *****
        # Use the Normal-Gamma observation likelihood model from the library
        try:
             observation_likelihood = bcpd.StudentT(alpha=alpha0, beta=beta0, kappa=kappa0, mu=mu0)
        except AttributeError:
             print(f"Error: Cannot find StudentT likelihood directly in bcpd module.")
             print("Please ensure the 'bayesian-changepoint-detection' library is installed correctly and check its API for likelihood models.")
             return None, None
        # ***** END FIX *****


        # Run BOCPD
        try:
            # Check for correct function name (might be detect_changepoints or offline_changepoint_detection)
            if hasattr(bcpd, 'detect_changepoints'):
                 R, maxes = bcpd.detect_changepoints(signal, hazard_func, observation_likelihood)
            elif hasattr(bcpd, 'offline_changepoint_detection'):
                 R, maxes = bcpd.offline_changepoint_detection(signal, hazard_func, observation_likelihood)
            else:
                 print("Error: Cannot find a suitable changepoint detection function in bcpd.")
                 return None, None

            R_features.append(R)
            maxes_list.append(maxes) # Store maxes from this dimension
        except Exception as e:
            print(f"Error running BOCPD on feature {dim}: {e}")
            # Handle error, maybe append None or skip? For now, return None
            return None, None


    # Combine results (averaging log probabilities)
    if not R_features: # Check if any feature processing succeeded
        print("Error: BOCPD failed for all features.")
        return None, None

    # Ensure all R matrices have the same shape before combining
    expected_shape = R_features[0].shape
    if not all(R.shape == expected_shape for R in R_features):
        print("Error: R matrices from different features have inconsistent shapes.")
        # Find max dimensions for padding or handle error
        max_rows = max(R.shape[0] for R in R_features)
        max_cols = max(R.shape[1] for R in R_features)
        # Basic handling: return error for now
        return None, None


    log_R_sum = np.zeros_like(R_features[0])
    valid_counts = np.zeros_like(R_features[0])

    for R_dim in R_features:
         log_R_dim = np.log(R_dim + 1e-10)
         log_R_sum += log_R_dim
         valid_counts += (R_dim > 0)

    avg_log_R = np.full_like(log_R_sum, -np.inf)
    mask = valid_counts > 0
    avg_log_R[mask] = log_R_sum[mask] / valid_counts[mask]
    R_combined = np.exp(avg_log_R)
    row_sums = R_combined.sum(axis=1, keepdims=True)
    # Ensure row_sums has compatible shape for division
    R_combined = np.divide(R_combined, row_sums, out=np.zeros_like(R_combined), where=row_sums > 1e-9) # Avoid division by zero


    # Combine maxes (e.g., take the mode or average - mode might be better)
    # For simplicity, just return maxes from the first dimension for now
    # A more robust combination might be needed depending on application
    combined_maxes = maxes_list[0] if maxes_list else None

    print("  BOCPD calculation finished.")
    return R_combined, combined_maxes

# --- Visualization ---
def plot_bocpd_results(trajectory_df, R, maxes, prob_threshold, trajectory_filename, feature_cols_to_plot):
    """Plots trajectory, run length probabilities, and detected change points."""
    try:
        trajectory_data = trajectory_df[feature_cols_to_plot].values
    except KeyError:
        print(f"Warning: Plotting features {feature_cols_to_plot} not found in {trajectory_filename}. Skipping.")
        return
    n_points = len(trajectory_data)
    if n_points == 0: print(f"Skipping plot for {trajectory_filename} due to empty data."); return

    time_steps = np.arange(n_points)
    n_plot_dims = trajectory_data.shape[1]

    # Create figure with 2 main sections: Trajectory + Run Length Probability Map
    fig = plt.figure(figsize=(12, 6 + n_plot_dims * 1.5))
    gs = fig.add_gridspec(n_plot_dims + 1, 1) # +1 row for probability map

    sns.set_style("whitegrid")
    ax_prob = fig.add_subplot(gs[n_plot_dims, 0]) # Probability map at the bottom

    # Plot trajectory dimensions
    axs_traj = []
    for i in range(n_plot_dims):
        ax = fig.add_subplot(gs[i, 0], sharex=ax_prob) # Share X with prob map
        axs_traj.append(ax)
        ax.plot(time_steps, trajectory_data[:, i], color='black', label=feature_cols_to_plot[i])
        ax.set_ylabel(feature_cols_to_plot[i])
        if i < n_plot_dims -1:
             plt.setp(ax.get_xticklabels(), visible=False) # Hide x-ticks for upper plots
        ax.grid(True, linestyle=':')

    # Plot Run Length Probability Map
    # R has shape (T+1, T+1), we plot R[1:, 1:] - probability of run length r at time t
    # Limit run length display for clarity if T is large
    T = n_points
    max_run_plot = min(T + 1, 200) # Limit y-axis for run length plot
    im = ax_prob.imshow(np.rot90(R[1:T+1, :max_run_plot]), aspect='auto', cmap='gray_r',
                        extent=[0, T, 0, max_run_plot], interpolation='none')
    ax_prob.set_xlabel("Time Step")
    ax_prob.set_ylabel("Run Length (r_t)")
    ax_prob.set_title("Run Length Probability P(r_t | x_{1:t})")
    fig.colorbar(im, ax=ax_prob, label="Probability")

    # Mark MAP change points (where run length probability drops significantly, often near 0)
    # Use maxes (argmax of R[t,:]) to find most likely run length at each time
    # Change points occur when maxes[t] is small (e.g., 0 or 1) after being large
    # Heuristic: Find points where maxes drops significantly
    change_points_bocpd = np.where(np.diff(maxes) < -int(HAZARD_LAMBDA * 0.5))[0] # Example heuristic
    print(f"  Heuristically derived change points from BOCPD MAP: {change_points_bocpd}")

    # Mark change points on trajectory plots
    if change_points_bocpd is not None and len(change_points_bocpd) > 0:
        for ax_t in axs_traj:
            for j, cp in enumerate(change_points_bocpd):
                 label = 'BOCPD Change Point' if ax_t == axs_traj[0] and j == 0 else ""
                 ax_t.axvline(cp, color='red', linestyle='--', linewidth=1.5, label=label)

    # Add legend to the first trajectory plot
    handles, labels = axs_traj[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs_traj[0].legend(by_label.values(), by_label.keys(), loc='best')

    fig.suptitle(f'BOCPD Segmentation: {trajectory_filename} (Lambda={HAZARD_LAMBDA})', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
        trajs_for_cpd = aligned_trajs_df
        time_label = "Time Step (Aligned)"
    else:
        print("\nSkipping DTW alignment.")
        trajs_for_cpd = original_trajs_df
        time_label = "Time Step (Original)"


    # 3. Perform BOCPD for each trajectory
    print(f"\nPerforming Bayesian Online Change Point Detection...")

    for i, traj_df in enumerate(trajs_for_cpd):
        filename = loaded_files[i] if i < len(loaded_files) else f"Trajectory {i}"
        print(f"\n--- Processing: {filename} ---")
        try:
            traj_cpd_features = traj_df[CPD_FEATURE_COLS].values
        except KeyError as e:
            print(f"  Skipping: Missing CPD feature column: {e}")
            continue

        if traj_cpd_features.shape[0] < 2:
            print("  Skipping BOCPD due to short trajectory length.")
            continue

        # Run BOCPD
        R_combined, maxes = detect_bocpd(
            traj_cpd_features,
            hazard_lambda=HAZARD_LAMBDA,
            alpha0=ALPHA0,
            beta0=BETA0,
            kappa0=KAPPA0,
            mu0=MU0
        )

        if R_combined is not None:
            # Plot the results
            plot_bocpd_results(traj_df, R_combined, maxes, PROB_THRESHOLD, filename, POS_COLS) # Plot position dims
        else:
            print("  BOCPD failed for this trajectory.")

