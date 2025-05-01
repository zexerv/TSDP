import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For legend handles if needed
import math
import os
from hmmlearn import hmm
from scipy import linalg # Potentially needed for covariance checks
import seaborn as sns # For improved aesthetics and kdeplot
# Required for DTW
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/switch'

# --- File Loading Config ---
LOAD_ALL_FILES = True
FILE_LIST = [] # Ignored if LOAD_ALL_FILES is True

# --- Feature Config ---
# Features for DTW distance calculation (usually position)
DTW_FEATURE_COLS = ['tx', 'ty', 'tz']

# Base features for HMM (Position is usually always included)
HMM_BASE_POS = True # Always include tx, ty, tz
HMM_BASE_ROT = False # Include r11..r33 from original aligned data

# --- NEW: Variance Feature Config ---
INCLUDE_VARIANCE = True  # Parent flag for variance features
# If INCLUDE_VARIANCE is True, these flags apply:
VAR_POS_X = True
VAR_POS_Y = True
VAR_POS_Z = True
VAR_ROT = False # Include variance of r11..r33 (only if HMM_BASE_ROT is also True)

# --- NEW: Normalized Time Feature Config ---
INCLUDE_NORM_TIME = True # Include normalized time as a feature

# --- HMM Config ---
N_STATES = 4            # Number of hidden states (phases) for the HMM
N_MIX = 1               # Number of Gaussian mixtures per state
N_ITER_HMM = 50         # Max iterations for HMM training
COVARIANCE_TYPE = 'diag'# 'diag' or 'full'.
RANDOM_STATE_HMM = 42   # For reproducible results


# --- Determine HMM Features based on Flags ---
POS_COLS = ['tx', 'ty', 'tz']
ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
HMM_FEATURE_COLS = []
HMM_FEATURE_LABELS = [] # For printing and clarity

if HMM_BASE_POS:
    HMM_FEATURE_COLS.extend(POS_COLS)
    HMM_FEATURE_LABELS.extend(['tx', 'ty', 'tz'])

if HMM_BASE_ROT:
    HMM_FEATURE_COLS.extend(ROT_COLS)
    HMM_FEATURE_LABELS.extend(['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33'])

# Store indices of base features within the initially loaded data (pos+rot)
all_load_features = POS_COLS + ROT_COLS
base_hmm_indices_in_loaded = [all_load_features.index(col) for col in HMM_FEATURE_COLS]

# Add variance features if requested
variance_feature_indices = [] # Store indices of variance features in the final HMM input
if INCLUDE_VARIANCE:
    if VAR_POS_X:
        HMM_FEATURE_LABELS.append('var_tx')
        variance_feature_indices.append(all_load_features.index('tx')) # Index in original loaded data
    if VAR_POS_Y:
        HMM_FEATURE_LABELS.append('var_ty')
        variance_feature_indices.append(all_load_features.index('ty'))
    if VAR_POS_Z:
        HMM_FEATURE_LABELS.append('var_tz')
        variance_feature_indices.append(all_load_features.index('tz'))
    if VAR_ROT and HMM_BASE_ROT: # Only add rot variance if base rot is included
        HMM_FEATURE_LABELS.extend([f'var_{col}' for col in ROT_COLS])
        variance_feature_indices.extend([all_load_features.index(col) for col in ROT_COLS])

# Add normalized time feature if requested
if INCLUDE_NORM_TIME:
    HMM_FEATURE_LABELS.append('norm_time')

N_FEATURES_HMM = len(HMM_FEATURE_LABELS)

print(f"Using features for DTW: {DTW_FEATURE_COLS}")
print(f"HMM using {N_FEATURES_HMM} features: {HMM_FEATURE_LABELS}")


# --- Data Loading ---
def load_selected_data(parent_folder, load_all=True, file_list=None):
    """Loads specified trajectories (all pose columns) from CSV files."""
    all_trajectories = []
    loaded_filenames = []
    # Always load position and rotation columns initially
    required_columns = POS_COLS + ROT_COLS

    if load_all:
        try:
            all_files_in_dir = [f for f in os.listdir(parent_folder) if os.path.isfile(os.path.join(parent_folder, f))]
            filenames = [f for f in all_files_in_dir if f.endswith('.csv')]
            try:
                 filenames.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError:
                 filenames.sort()
            print(f"Found {len(filenames)} CSV files to load: {filenames}")
            if not filenames:
                 print(f"Warning: No CSV files found in {parent_folder}")
                 return [], []
        except FileNotFoundError:
            print(f"Error: Parent folder not found - {parent_folder}")
            return [], []
    elif file_list:
        filenames = file_list
        print(f"Loading specified files: {filenames}")
    else:
        print("Error: No files specified to load.")
        return [], []

    for filename in filenames:
        file_path = os.path.join(parent_folder, filename)
        # print(f"  Loading {file_path}...") # Reduce verbosity
        try:
            df = pd.read_csv(file_path)
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                 raise KeyError(f"Missing columns: {missing_cols}")

            full_trajectory_data = df[required_columns].values

            if np.isnan(full_trajectory_data).any():
                print(f"Warning: NaN values found in {file_path}. Skipping file.")
                continue
            if full_trajectory_data.shape[0] == 0:
                 print(f"Warning: Trajectory data is empty in {file_path}. Skipping file.")
                 continue

            all_trajectories.append(full_trajectory_data)
            loaded_filenames.append(filename)

        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
        except KeyError as e:
            print(f"Error in {file_path}: {e}")
        except Exception as e:
            print(f"Error loading or processing {file_path}: {e}")

    if not all_trajectories:
         print("Warning: No valid trajectory data loaded.")

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
        # print(f"  Aligning trajectory {i} (length {len(traj_full)}) to reference...") # Reduce verbosity
        if len(traj_full) == 0:
             aligned_trajectories.append(np.zeros_like(reference_trajectory_full))
             continue
        if i == reference_index:
            aligned_trajectories.append(reference_trajectory_full.copy())
            continue
        traj_dtw_features = traj_full[:, dtw_feature_indices]
        ref_dtw_features = reference_trajectory_full[:, dtw_feature_indices]
        distance, path = fastdtw(ref_dtw_features, traj_dtw_features, dist=euclidean)
        # print(f"    DTW distance: {distance:.4f}") # Reduce verbosity
        path_array = np.array(path)
        warped_traj_full = np.zeros_like(reference_trajectory_full)
        ref_indices_in_path = path_array[:, 0]
        traj_indices_in_path = path_array[:, 1]
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

# --- Feature Augmentation ---
# ***** MODIFIED FUNCTION SIGNATURE *****
def augment_features(aligned_trajectories, ref_len, base_indices, var_indices, include_variance, include_norm_time):
    """Adds variance and normalized time features to aligned trajectories."""
    if not aligned_trajectories:
        return []

    n_trajectories = len(aligned_trajectories)
    n_base_features = len(base_indices)
    # Use the passed argument here
    n_var_features = len(var_indices) if include_variance else 0
    n_time_features = 1 if include_norm_time else 0
    n_total_hmm_features = n_base_features + n_var_features + n_time_features

    # Stack trajectories for variance calculation (n_traj, ref_len, n_all_features)
    stacked_trajs = np.stack(aligned_trajectories)

    # Calculate variance across trajectories for selected features
    variance_features = np.zeros((ref_len, n_var_features))
    if include_variance and n_var_features > 0:
        # Calculate variance for the specified original feature indices
        variance_features = np.var(stacked_trajs[:, :, var_indices], axis=0)
        # Add small epsilon to variance to avoid issues with zero variance
        variance_features += 1e-9

    # Calculate normalized time
    norm_time_feature = np.linspace(0, 1, ref_len).reshape(-1, 1) if include_norm_time else np.empty((ref_len, 0))

    # Augment each trajectory
    augmented_trajectories = []
    for i in range(n_trajectories):
        base_data = aligned_trajectories[i][:, base_indices]
        # Concatenate features: base_pose/rot, variance, norm_time
        augmented_traj = np.concatenate([
            base_data,
            variance_features, # Already shaped (ref_len, n_var_features)
            norm_time_feature  # Already shaped (ref_len, n_time_features)
        ], axis=1)

        # Ensure the final shape matches expected HMM features
        if augmented_traj.shape[1] != n_total_hmm_features:
             print(f"Warning: Feature dimension mismatch for trajectory {i}. Expected {n_total_hmm_features}, got {augmented_traj.shape[1]}")
             # Handle error or skip trajectory
             continue
        augmented_trajectories.append(augmented_traj)

    return augmented_trajectories


# --- HMM Training ---
def train_hmm(data, n_states, n_mix, n_iter, cov_type, random_state):
    """Trains a GMM-HMM model on concatenated data (assumes same length)."""
    print(f"\nTraining GMM-HMM with {n_states} states and {n_mix} mixture components...")
    model = hmm.GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type=cov_type,
                       n_iter=n_iter, tol=1e-3, random_state=random_state, verbose=True,
                       params='stmcw', init_params='stmcw')
    try:
        model.fit(data) # No lengths needed
        print("HMM Training complete.")
        if hasattr(model, 'monitor_') and not model.monitor_.converged:
            print("Warning: HMM training did not converge.")
        return model
    except ValueError as e: print(f"Error during HMM fitting: {e}"); return None
    except Exception as e: print(f"Unexpected error during HMM fitting: {e}"); return None

# --- GMM Contour Visualization ---
def plot_gmm_contours(ax, data, labels, plot_dims, model, colors, title):
    """Plots data points (position only) and GMM contours."""
    dim1, dim2 = plot_dims
    dim_labels = ['tx', 'ty', 'tz']
    # Ensure data has at least 3 columns before slicing for position
    if data.shape[1] < 3:
        print("Warning: Cannot plot GMM contours, input data has fewer than 3 dimensions.")
        return [], []
    pos_data = data[:, :3] # Extract X, Y, Z for plotting (first 3 features)

    sns.set_style("whitegrid")
    plot_handles, plot_labels = [], []
    for state in range(model.n_components):
        state_indices = np.where(labels == state)[0]
        if len(state_indices) > 0:
            state_pos_data = pos_data[state_indices]
            scatter = ax.scatter(state_pos_data[:, dim1], state_pos_data[:, dim2], s=15, alpha=0.5,
                       color=colors[state], label=f'State {state} Data')
            if f'State {state} Data' not in plot_labels:
                plot_handles.append(scatter); plot_labels.append(f'State {state} Data')
            try:
                # Check if enough points and variance for KDE
                if len(state_indices) > 5 and state_pos_data.shape[1] > max(plot_dims) and np.linalg.matrix_rank(np.cov(state_pos_data[:, plot_dims].T)) == 2:
                     sns.kdeplot(x=state_pos_data[:, dim1], y=state_pos_data[:, dim2], ax=ax,
                                 color=colors[state], levels=4, linewidths=1.5)
                     proxy = plt.Line2D([0], [0], linestyle='-', color=colors[state], lw=1.5, label=f'State {state} Density')
                     if f'State {state} Density' not in plot_labels:
                          plot_handles.append(proxy); plot_labels.append(f'State {state} Density')
                else: # Fallback: plot mean
                     if state < model.n_components and model.n_mix > 0:
                          mean_hmm = model.means_[state, 0, :] # Mean of first component, all features
                          if len(mean_hmm) > max(plot_dims): # Check if plot dimensions exist in mean
                              mean_plot = ax.scatter(mean_hmm[dim1], mean_hmm[dim2], marker='X', s=100, color=colors[state], edgecolor='black', zorder=11, label=f'State {state} Mean')
                              if f'State {state} Mean' not in plot_labels:
                                   plot_handles.append(mean_plot); plot_labels.append(f'State {state} Mean')
                          else:
                              print(f"Warning: Cannot plot mean for State {state}, mean dimension ({len(mean_hmm)}) too small for plot dimensions ({plot_dims}).")
            except Exception as e: print(f"Error plotting KDE/Mean for State {state}: {e}")
    ax.set_xlabel(f'{dim_labels[dim1]}'); ax.set_ylabel(f'{dim_labels[dim2]}')
    ax.set_title(title); ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    return plot_handles, plot_labels

# --- State vs Aligned Time Plotting ---
def plot_state_vs_time(ax, all_labels, ref_len, n_states, colors):
    """Plots predicted state against aligned time steps."""
    if ref_len == 0: # Avoid division by zero if ref_len is 0
        print("Warning: Reference length is 0, cannot plot state vs time.")
        return
    n_aligned_traj = len(all_labels) // ref_len
    if n_aligned_traj == 0 and len(all_labels) > 0: # Handle case where labels might not be multiple of ref_len
         print(f"Warning: Number of labels ({len(all_labels)}) not multiple of ref_len ({ref_len}). Plotting points individually.")
         n_aligned_traj = 1 # Assume single sequence for plotting purposes
         time_repeated = np.arange(len(all_labels)) # Use index instead of aligned time
         state_jitter = all_labels + np.random.normal(0, 0.08, size=all_labels.shape)
         ax.set_xlabel("Data Point Index")
    elif n_aligned_traj > 0:
        aligned_time = np.arange(ref_len)
        state_jitter = all_labels + np.random.normal(0, 0.08, size=all_labels.shape)
        time_repeated = np.tile(aligned_time, n_aligned_traj)
        ax.set_xlabel("Time Step (Aligned)")
    else: # No labels
        print("No labels to plot for state vs time.")
        return


    for state in range(n_states):
        state_mask = (all_labels == state)
        if np.any(state_mask):
            ax.scatter(time_repeated[state_mask], state_jitter[state_mask],
                       color=colors[state], alpha=0.3, s=5, label=f'State {state}')

    ax.set_ylabel("Predicted State Index")
    ax.set_title("State Occurrence vs. Aligned Trajectory Time")
    ax.set_yticks(np.arange(n_states)); ax.set_yticklabels([str(i) for i in range(n_states)])
    ax.grid(True, linestyle=':', alpha=0.7, axis='y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=3)

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load data (loading all pose features initially)
    original_trajs_full, loaded_files = load_selected_data(
        PARENT_FOLDER_PATH, load_all=LOAD_ALL_FILES, file_list=FILE_LIST
    )
    if not original_trajs_full: exit("Exiting: No trajectories loaded.")

    # 2. Perform DTW Alignment using specified DTW features
    all_load_features = POS_COLS + ROT_COLS # Reference list for indices
    try:
        dtw_col_indices = [all_load_features.index(col) for col in DTW_FEATURE_COLS]
    except ValueError as e: exit(f"Error: DTW feature column not found: {e}")

    aligned_trajs_full, ref_traj_full, ref_idx = align_trajectories_dtw(
        original_trajs_full, dtw_col_indices
    )
    valid_aligned_trajs_full = [t for t in aligned_trajs_full if len(t) > 0 and np.any(t)]
    if not valid_aligned_trajs_full: exit("\nNo valid aligned trajectories after DTW. Exiting.")
    print(f"\nSuccessfully aligned {len(valid_aligned_trajs_full)} trajectories.")
    ref_len = len(valid_aligned_trajs_full[0]) # Get length after alignment

    # 3. Augment features (Variance and Time)
    print("Augmenting features...")
    # ***** MODIFIED FUNCTION CALL *****
    hmm_input_trajs = augment_features(
        valid_aligned_trajs_full,
        ref_len,
        base_hmm_indices_in_loaded,
        variance_feature_indices,
        INCLUDE_VARIANCE, # Pass the global flag
        INCLUDE_NORM_TIME
    )
    if not hmm_input_trajs: exit("Feature augmentation failed.")

    concatenated_hmm_data = np.concatenate(hmm_input_trajs)

    # Check HMM data validity
    if concatenated_hmm_data.shape[1] != N_FEATURES_HMM:
         exit(f"Error: Final HMM data has wrong shape {concatenated_hmm_data.shape}. Expected {N_FEATURES_HMM} features.")
    if np.isnan(concatenated_hmm_data).any():
        exit("Error: Final HMM data contains NaN values.")

    # 4. Train the HMM on augmented aligned data
    hmm_model = train_hmm(concatenated_hmm_data, N_STATES, N_MIX, N_ITER_HMM, COVARIANCE_TYPE, RANDOM_STATE_HMM)

    # 5. Predict states for all aligned data points
    if hmm_model:
        try:
            all_predicted_states = hmm_model.predict(concatenated_hmm_data)
            print("\nPredicted states for all aligned data points.")

            # --- Visualization ---
            sns.set_theme(style="whitegrid")
            state_colors = sns.color_palette("viridis", N_STATES)

            # 6. Visualize GMM Contours (based on original position features)
            print("Generating GMM contour visualizations...")
            fig_gmm, axs_gmm = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
            fig_gmm.suptitle(f'GMM State Distributions (Post-DTW, {N_STATES} States, Features Used: {N_FEATURES_HMM})', fontsize=16)
            # Pass the augmented data, plot_gmm_contours knows to use first 3 cols for plot
            h1, l1 = plot_gmm_contours(axs_gmm[0], concatenated_hmm_data, all_predicted_states, [0, 1], hmm_model, state_colors, "XY Projection")
            h2, l2 = plot_gmm_contours(axs_gmm[1], concatenated_hmm_data, all_predicted_states, [0, 2], hmm_model, state_colors, "XZ Projection")
            h3, l3 = plot_gmm_contours(axs_gmm[2], concatenated_hmm_data, all_predicted_states, [1, 2], hmm_model, state_colors, "YZ Projection")
            all_handles = h1+h2+h3; all_labels = l1+l2+l3
            by_label = dict(zip(all_labels, all_handles))
            fig_gmm.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize='medium')
            plt.tight_layout(rect=[0, 0.03, 0.9, 0.95]); plt.show()

            # 7. Visualize State vs Aligned Time
            print("Generating State vs Aligned Time plot...")
            fig_time, ax_time = plt.subplots(1, 1, figsize=(12, 6))
            plot_state_vs_time(ax_time, all_predicted_states, ref_len, N_STATES, state_colors)
            plt.tight_layout(rect=[0, 0.03, 0.85, 0.95]); plt.show()

            # Print learned parameters (optional)
            print("\nLearned HMM Parameters (Post-DTW, Augmented Features):")
            print(f"Features used for HMM: {HMM_FEATURE_LABELS}")
            print("Initial Probabilities (pi):"); print(hmm_model.startprob_)
            print("\nTransition Matrix (A):"); print(np.round(hmm_model.transmat_, 3))
            print("\nGMM Weights per State:"); print(np.round(hmm_model.weights_, 3))
            print("\nGMM Means per State (showing first 3 dims - Pos):")
            if hmm_model.means_.shape[-1] >= 3: print(np.round(hmm_model.means_[:, :, :3], 3))
            else: print(np.round(hmm_model.means_, 3))

        except Exception as e: print(f"Error during prediction or plotting: {e}")
    else: print("HMM model training failed.")

