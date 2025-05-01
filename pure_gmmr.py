import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.mixture import GaussianMixture
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
# Features to use for GMM/GMR (Position is required)
GMM_POS_COLS = ['tx', 'ty', 'tz']
# Optional: Include Rotation features for GMM/GMR
INCLUDE_ROTATION_GMM = False
GMM_ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

# --- GMM/GMR Config ---
N_GMM_COMPONENTS = 5     # Number of Gaussian components for the GMM
GMM_COVARIANCE_TYPE = 'full' # 'full', 'diag', 'tied', 'spherical'
RANDOM_STATE_GMM = 42    # For reproducibility

# --- Determine GMM Features ---
# Define base column lists globally for clarity
POS_COLS = ['tx', 'ty', 'tz']
ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

GMM_FEATURE_COLS = POS_COLS[:] # Start with position
if INCLUDE_ROTATION_GMM:
    GMM_FEATURE_COLS.extend(ROT_COLS)

# The actual features used by GMM will be [time] + GMM_FEATURE_COLS
GMM_INPUT_DIM = 1 + len(GMM_FEATURE_COLS) # +1 for the time dimension

print(f"Using features for DTW: {DTW_FEATURE_COLS}")
print(f"GMM/GMR using {GMM_INPUT_DIM} features: ['time'] + {GMM_FEATURE_COLS}")


# --- Data Loading ---
def load_selected_data(parent_folder, load_all=True, file_list=None):
    """Loads specified trajectories (all pose columns) from CSV files."""
    all_trajectories = []
    loaded_filenames = []
    # Define required columns inside the function using the global lists
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

# --- Data Preparation for GMM ---
def prepare_gmm_data(aligned_trajectories, ref_len, gmm_feature_indices):
    """Prepares data for GMM training by adding normalized time."""
    if not aligned_trajectories:
        return None, None

    # Normalized time vector [0, 1]
    norm_time = np.linspace(0, 1, ref_len).reshape(-1, 1)

    gmm_data_list = []
    for traj in aligned_trajectories:
        if len(traj) == ref_len: # Ensure traj has the correct length
             # Select specified features and prepend time
             features = traj[:, gmm_feature_indices]
             time_augmented_features = np.hstack((norm_time, features))
             gmm_data_list.append(time_augmented_features)
        else:
             print(f"Warning: Skipping trajectory with unexpected length {len(traj)} during GMM data prep.")


    if not gmm_data_list:
         print("Error: No valid trajectories left after filtering for GMM data prep.")
         return None, None

    # Concatenate all data points
    concatenated_gmm_data = np.concatenate(gmm_data_list, axis=0)

    return concatenated_gmm_data, norm_time.flatten() # Return time vector as well

# --- GMM Training ---
def train_gmm(data, n_components, cov_type, random_state):
    """Trains a Gaussian Mixture Model."""
    print(f"\nTraining GMM with {n_components} components...")
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type=cov_type,
                          random_state=random_state,
                          warm_start=False, # Start fresh each time
                          verbose=1, verbose_interval=10)
    try:
        gmm.fit(data)
        print(f"GMM Training complete. Converged: {gmm.converged_}")
        return gmm
    except Exception as e:
        print(f"Error during GMM fitting: {e}")
        return None

# --- Gaussian Mixture Regression (GMR) ---
def calculate_gmr(gmm_model, time_vector):
    """
    Performs Gaussian Mixture Regression to predict pose given time.

    Args:
        gmm_model: Trained sklearn GaussianMixture model.
        time_vector: 1D array of time points to predict poses for.

    Returns:
        tuple: (predicted_means, predicted_covariances)
               predicted_means: Array (len(time_vector), n_pose_features) of expected poses.
               predicted_covariances: Array (len(time_vector), n_pose_features, n_pose_features)
                                     of pose covariances.
    """
    n_components = gmm_model.n_components
    # Ensure means_ attribute exists and has correct dimensions
    if not hasattr(gmm_model, 'means_') or gmm_model.means_ is None or gmm_model.means_.ndim < 2:
        raise ValueError("GMM model is not fitted properly or means_ attribute is missing/invalid.")
    n_total_features = gmm_model.means_.shape[1]
    if n_total_features <= 1:
        raise ValueError(f"GMM model has {n_total_features} features, needs at least 2 (time + 1 pose dim).")

    n_pose_features = n_total_features - 1 # All features except time
    n_times = len(time_vector)

    predicted_means = np.zeros((n_times, n_pose_features))
    predicted_covariances = np.zeros((n_times, n_pose_features, n_pose_features))

    # Calculate responsibilities (beta)
    beta = np.zeros((n_times, n_components))
    for k in range(n_components):
        mu_t_k = gmm_model.means_[k, 0]
        try:
            if gmm_model.covariance_type == 'full':
                sigma_t_k = gmm_model.covariances_[k, 0, 0]
            elif gmm_model.covariance_type == 'diag':
                sigma_t_k = gmm_model.covariances_[k, 0]
            elif gmm_model.covariance_type == 'tied':
                sigma_t_k = gmm_model.covariances_[0, 0]
            elif gmm_model.covariance_type == 'spherical':
                # For spherical, covariance_ is scalar per component
                sigma_t_k = gmm_model.covariances_[k]
            else: # Should not happen with sklearn GMM
                raise TypeError(f"Unsupported covariance type: {gmm_model.covariance_type}")

            # Ensure sigma_t_k is positive
            sigma_t_k = max(sigma_t_k, 1e-9) # Add epsilon for stability

            norm_factor = 1.0 / np.sqrt(2 * np.pi * sigma_t_k)
            time_diff_sq = (time_vector - mu_t_k)**2
            prob_time_given_k = norm_factor * np.exp(-0.5 * time_diff_sq / sigma_t_k)
            beta[:, k] = prob_time_given_k * gmm_model.weights_[k]
        except IndexError:
             print(f"Warning: Index error accessing covariance for component {k}. Cov shape: {gmm_model.covariances_.shape if hasattr(gmm_model, 'covariances_') else 'N/A'}")
             beta[:, k] = 0 # Assign zero probability if error occurs
        except TypeError:
             # Handle case where spherical covariance might be returned differently
             if gmm_model.covariance_type == 'spherical' and isinstance(gmm_model.covariances_[k], (int, float, np.number)):
                 sigma_t_k = max(gmm_model.covariances_[k], 1e-9)
                 norm_factor = 1.0 / np.sqrt(2 * np.pi * sigma_t_k)
                 time_diff_sq = (time_vector - mu_t_k)**2
                 prob_time_given_k = norm_factor * np.exp(-0.5 * time_diff_sq / sigma_t_k)
                 beta[:, k] = prob_time_given_k * gmm_model.weights_[k]
             else:
                 print(f"Warning: Type error accessing covariance for component {k}. Cov type: {type(gmm_model.covariances_)}")
                 beta[:, k] = 0


    beta_sum = np.sum(beta, axis=1, keepdims=True)
    # Handle cases where sum is zero (all components have zero prob for that time)
    valid_sum_mask = (beta_sum > 1e-9).flatten() # Use flatten() for direct boolean indexing

    # Ensure beta remains 2D even if only one time step is valid
    if np.any(valid_sum_mask): # Check if there are any valid sums
         # ***** FIX: Reshape beta_sum slice for broadcasting *****
         beta[valid_sum_mask, :] = beta[valid_sum_mask, :] / beta_sum[valid_sum_mask]
         # ***** END FIX *****
    # For time steps where sum is zero, responsibilities remain zero

    # Calculate predicted mean and covariance
    for t, time_val in enumerate(time_vector):
        if not valid_sum_mask[t]: # Skip if no component responsible
             predicted_means[t] = np.nan
             predicted_covariances[t] = np.nan
             continue

        mean_t = np.zeros(n_pose_features)
        cov_t = np.zeros((n_pose_features, n_pose_features))

        for k in range(n_components):
            beta_tk = beta[t, k]
            if beta_tk < 1e-9: continue

            mu_k = gmm_model.means_[k, :]
            mu_t_k = mu_k[0]
            mu_p_k = mu_k[1:]

            try:
                if gmm_model.covariance_type == 'full':
                    sigma_k = gmm_model.covariances_[k, :, :]
                    sigma_pt_k = sigma_k[1:, 0:1]; sigma_tp_k = sigma_k[0:1, 1:]
                    sigma_tt_k_inv = 1.0 / max(sigma_k[0, 0], 1e-9)
                    sigma_pp_k = sigma_k[1:, 1:]
                elif gmm_model.covariance_type == 'diag':
                    sigma_k_diag = gmm_model.covariances_[k, :]
                    sigma_pt_k = np.zeros((n_pose_features, 1)); sigma_tp_k = np.zeros((1, n_pose_features))
                    sigma_tt_k_inv = 1.0 / max(sigma_k_diag[0], 1e-9)
                    sigma_pp_k = np.diag(sigma_k_diag[1:])
                elif gmm_model.covariance_type == 'tied':
                    sigma_k = gmm_model.covariances_
                    sigma_pt_k = sigma_k[1:, 0:1]; sigma_tp_k = sigma_k[0:1, 1:]
                    sigma_tt_k_inv = 1.0 / max(sigma_k[0, 0], 1e-9)
                    sigma_pp_k = sigma_k[1:, 1:]
                elif gmm_model.covariance_type == 'spherical':
                    # Ensure sigma_k_val is scalar
                    if isinstance(gmm_model.covariances_[k], (int, float, np.number)):
                        sigma_k_val = gmm_model.covariances_[k]
                    elif isinstance(gmm_model.covariances_[k], np.ndarray) and gmm_model.covariances_[k].ndim == 0:
                         sigma_k_val = gmm_model.covariances_[k].item()
                    else: # Should not happen if fitted correctly
                         raise TypeError(f"Unexpected spherical covariance type/shape for component {k}")
                    sigma_pt_k = np.zeros((n_pose_features, 1)); sigma_tp_k = np.zeros((1, n_pose_features))
                    sigma_tt_k_inv = 1.0 / max(sigma_k_val, 1e-9)
                    sigma_pp_k = np.eye(n_pose_features) * sigma_k_val

                # GMR Equations
                if sigma_pt_k.shape != (n_pose_features, 1) or sigma_tp_k.shape != (1, n_pose_features):
                     print(f"Warning: Dimension mismatch in covariance calculation for component {k} at time {t}. Skipping component.")
                     continue

                term_mu = (sigma_pt_k * sigma_tt_k_inv * (time_val - mu_t_k)).flatten()
                mu_p_tk = mu_p_k + term_mu

                term_cov = (sigma_pt_k * sigma_tt_k_inv) @ sigma_tp_k
                sigma_p_tk = sigma_pp_k - term_cov

                mean_t += beta_tk * mu_p_tk
                cov_t += beta_tk * (sigma_p_tk + np.outer(mu_p_tk, mu_p_tk))

            except IndexError:
                 print(f"Warning: Index error accessing covariance/mean for component {k} at time {t}. Skipping component.")
            except Exception as e:
                 print(f"Warning: Error calculating GMR for component {k} at time {t}: {e}. Skipping component.")


        cov_t -= np.outer(mean_t, mean_t)
        predicted_means[t] = mean_t
        predicted_covariances[t] = cov_t

    return predicted_means, predicted_covariances

# --- Sampling ---
def sample_gmr_trajectory(gmm_model, time_vector):
    """Samples a single trajectory from the GMR model."""
    means, covariances = calculate_gmr(gmm_model, time_vector)
    n_times, n_pose_features = means.shape
    sampled_trajectory = np.zeros((n_times, n_pose_features))

    for t in range(n_times):
        if np.isnan(means[t]).any() or np.isnan(covariances[t]).any():
             print(f"Warning: NaN encountered in GMR output at time step {t}. Cannot sample.")
             sampled_trajectory[t:] = np.nan # Propagate NaN
             break # Stop sampling for this trajectory

        try:
            cov_t = covariances[t]
            cov_t = (cov_t + cov_t.T) / 2 # Ensure symmetry
            cov_t += np.eye(n_pose_features) * 1e-7 # Add jitter
            sampled_trajectory[t] = np.random.multivariate_normal(means[t], cov_t)
        except np.linalg.LinAlgError:
            print(f"Warning: Covariance matrix not positive definite at time step {t}. Using mean only.")
            sampled_trajectory[t] = means[t]
        except ValueError as e:
             print(f"Warning: Error sampling at time step {t}: {e}. Using mean only.")
             sampled_trajectory[t] = means[t]

    return sampled_trajectory


# --- Visualization ---
def plot_gmm_gmr_results(gmm_model, gmm_data, time_vector, gmr_means, gmr_covs, n_samples=3):
    """Visualizes the GMM components, GMR result, and sampled trajectories."""
    n_components = gmm_model.n_components
    n_pose_features = gmr_means.shape[1]
    n_plot_dims = min(n_pose_features, 3) # Plot up to 3 pose dimensions

    feature_labels = GMM_FEATURE_COLS[:n_plot_dims] # Labels for pose dimensions

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("husl", n_components) # Colors for GMM components
    gmr_color = "black"
    sample_color = "grey"

    # --- Plot 1: GMM Components and Data (Time vs. Pose Dims) ---
    fig1, axs1 = plt.subplots(n_plot_dims, 1, figsize=(12, 4 * n_plot_dims), sharex=True)
    if n_plot_dims == 1: axs1 = [axs1] # Make indexable
    fig1.suptitle(f'GMM Components ({n_components}) in Time-Pose Space', fontsize=16)

    # Plot all data points
    for j in range(n_plot_dims):
        axs1[j].scatter(gmm_data[:, 0], gmm_data[:, j+1], s=5, alpha=0.1, color='grey', label='Data Points')

    # Plot GMM ellipses
    for k in range(n_components):
        mean = gmm_model.means_[k]
        cov = gmm_model.covariances_ if gmm_model.covariance_type == 'tied' else gmm_model.covariances_[k]

        for j in range(n_plot_dims):
            ax = axs1[j]
            plot_dims = [0, j + 1]
            mean_2d = mean[plot_dims]

            try:
                if gmm_model.covariance_type == 'full':
                    cov_2d = cov[np.ix_(plot_dims, plot_dims)]
                elif gmm_model.covariance_type == 'diag':
                    cov_2d = np.diag(cov[plot_dims])
                elif gmm_model.covariance_type == 'tied':
                     cov_2d = cov[np.ix_(plot_dims, plot_dims)]
                elif gmm_model.covariance_type == 'spherical':
                     # Ensure cov is treated as scalar variance here
                     # Check if cov is scalar or array before indexing
                     if isinstance(cov, (int, float, np.number)): cov_val = cov
                     elif isinstance(cov, np.ndarray) and cov.ndim == 0: cov_val = cov.item()
                     elif isinstance(cov, np.ndarray) and cov.ndim >= 1: cov_val = cov[0] # Assume same variance
                     else: raise TypeError(f"Unexpected type for spherical covariance: {type(cov)}")
                     cov_2d = np.diag([cov_val, cov_val]) # Use same variance for time and pose dim

                v, w = np.linalg.eigh(cov_2d)
                if np.any(v < 1e-9): continue
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[:, 0]
                angle = 180. * np.arctan2(u[1], u[0]) / np.pi
                ell = patches.Ellipse(mean_2d, v[0], v[1], angle=angle, color=colors[k], alpha=0.6, fill=False, linewidth=2, label=f'Comp {k}' if j==0 else "")
                ax.add_patch(ell)
            except Exception as e:
                print(f"Could not plot ellipse for component {k}, dim {j+1}: {e}")

            ax.set_ylabel(feature_labels[j])
            ax.grid(True, linestyle=':')

    axs1[-1].set_xlabel('Normalized Time')
    # Consolidate legend
    handles, labels = axs1[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig1.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.show()


    # --- Plot 2: GMR Mean, Variance, and Samples ---
    fig2, axs2 = plt.subplots(n_plot_dims, 1, figsize=(12, 4 * n_plot_dims), sharex=True)
    if n_plot_dims == 1: axs2 = [axs2] # Make indexable
    fig2.suptitle('GMR Predicted Trajectory and Samples', fontsize=16)

    # Filter out NaN values from GMR results before plotting
    valid_indices = ~np.isnan(gmr_means).any(axis=1)
    time_vector_valid = time_vector[valid_indices]
    gmr_means_valid = gmr_means[valid_indices]
    gmr_covs_valid = gmr_covs[valid_indices]


    for j in range(n_plot_dims):
        ax = axs2[j]
        # Plot original data points (optional)
        ax.scatter(gmm_data[:, 0], gmm_data[:, j+1], s=5, alpha=0.05, color='lightgrey', label='_nolegend_')

        # Plot GMR mean
        ax.plot(time_vector_valid, gmr_means_valid[:, j], color=gmr_color, linewidth=2.5, label='GMR Mean')

        # Plot GMR variance
        # Ensure gmr_covs_valid has the correct shape before indexing
        if gmr_covs_valid.ndim == 3 and gmr_covs_valid.shape[1] > j and gmr_covs_valid.shape[2] > j:
            std_dev = np.sqrt(gmr_covs_valid[:, j, j] + 1e-9)
            ax.fill_between(time_vector_valid, gmr_means_valid[:, j] - std_dev, gmr_means_valid[:, j] + std_dev,
                            color=gmr_color, alpha=0.2, label='GMR Std Dev')
        else:
            print(f"Warning: Could not plot standard deviation for dimension {j}. Covariance shape: {gmr_covs_valid.shape}")


        # Plot sampled trajectories
        for s in range(n_samples):
            sampled_traj = sample_gmr_trajectory(gmm_model, time_vector)
            # Plot only valid (non-NaN) parts of sampled trajectories
            valid_sample_indices = ~np.isnan(sampled_traj).any(axis=1)
            # Ensure dimension j exists in sampled_traj
            if sampled_traj.shape[1] > j:
                ax.plot(time_vector[valid_sample_indices], sampled_traj[valid_sample_indices, j],
                        color=sample_color, linestyle='--', linewidth=1, alpha=0.7, label='Samples' if s==0 else "")
            else:
                 print(f"Warning: Dimension {j} not found in sampled trajectory.")


        ax.set_ylabel(feature_labels[j])
        ax.grid(True, linestyle=':')
        if j == 0:
             ax.legend(loc='best')

    axs2[-1].set_xlabel('Normalized Time')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # Define base column lists globally (needed for load_selected_data)
    POS_COLS = ['tx', 'ty', 'tz']
    ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

    # 1. Load data (loading all pose features initially)
    all_features_to_load = POS_COLS + ROT_COLS
    original_trajs_full, loaded_files = load_selected_data(
        PARENT_FOLDER_PATH, load_all=LOAD_ALL_FILES, file_list=FILE_LIST
    )
    if not original_trajs_full: exit("Exiting: No trajectories loaded.")

    # 2. Perform DTW Alignment using specified DTW features
    try: dtw_col_indices = [all_features_to_load.index(col) for col in DTW_FEATURE_COLS]
    except ValueError as e: exit(f"Error: DTW feature column not found: {e}")
    aligned_trajs_full, ref_traj_full, ref_idx = align_trajectories_dtw(original_trajs_full, dtw_col_indices)
    valid_aligned_trajs_full = [t for t in aligned_trajs_full if len(t) > 0 and np.any(t)]
    if not valid_aligned_trajs_full: exit("\nNo valid aligned trajectories after DTW. Exiting.")
    print(f"\nSuccessfully aligned {len(valid_aligned_trajs_full)} trajectories.")
    ref_len = len(valid_aligned_trajs_full[0])

    # 3. Prepare data for GMM (Select GMM features + add time)
    try: gmm_col_indices = [all_features_to_load.index(col) for col in GMM_FEATURE_COLS]
    except ValueError as e: exit(f"Error: GMM feature column not found: {e}")

    concatenated_gmm_data, norm_time_vec = prepare_gmm_data(valid_aligned_trajs_full, ref_len, gmm_col_indices)
    if concatenated_gmm_data is None: exit("Failed to prepare GMM data.")
    if concatenated_gmm_data.shape[1] != GMM_INPUT_DIM:
         exit(f"Error: GMM input data has wrong shape {concatenated_gmm_data.shape}. Expected {GMM_INPUT_DIM} features.")
    if np.isnan(concatenated_gmm_data).any(): exit("Error: GMM input data contains NaN values.")

    # 4. Train the GMM
    gmm = train_gmm(concatenated_gmm_data, N_GMM_COMPONENTS, GMM_COVARIANCE_TYPE, RANDOM_STATE_GMM)

    # 5. Perform GMR and Visualize
    if gmm:
        try:
            print("\nCalculating GMR...")
            gmr_mean_traj, gmr_cov_traj = calculate_gmr(gmm, norm_time_vec)

            print("Generating visualizations...")
            plot_gmm_gmr_results(gmm, concatenated_gmm_data, norm_time_vec, gmr_mean_traj, gmr_cov_traj, n_samples=5)

            # Optional: Print GMM parameters
            print("\nLearned GMM Parameters:")
            print(f"Features used: ['time'] + {GMM_FEATURE_COLS}")
            print("Weights:", np.round(gmm.weights_, 3))
            print("Means (Time + Features):", np.round(gmm.means_, 3))

        except Exception as e:
            print(f"Error during GMR or plotting: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
    else:
        print("GMM model training failed.")

