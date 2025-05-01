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
from scipy.optimize import minimize # For optimization

# --- Configuration ---
# !!! PLEASE UPDATE THIS PATH !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/linear_switch'

LOAD_ALL_FILES = True
FILE_LIST = []
ALIGNMENT_TYPE = 'Combined'
ALIGNMENT_BASE_FEATURE_COLS = ['tx', 'ty', 'tz', 'r11','r12','r13','r21','r22','r23','r31','r32','r33']
NORMALIZE_FOR_COMBINED = True

# --- Segmentation Configuration ---
# Features to use for segmentation (will fit model to the mean of these)
SEGMENTATION_FEATURE_COLS = ['tx', 'ty', 'tz', 'r11','r12','r13','r21','r22','r23','r31','r32','r33']
# --- Basis Function Fitting Settings ---
NUM_BASIS_FUNCTIONS = 6 # Fixed number of atan/tanh functions (M)
ROTATION_WEIGHT = 0.001 # Weight for rotation features in cost
BASIS_FUNCTION_TYPE = 'tanh' # 'tanh' or 'atan'

# --- Downsampling Configuration (for fitting) ---
MAX_FIT_LENGTH = 300 # Max length for optimization data

# --- Optimization Settings ---
OPTIMIZER_METHOD = 'L-BFGS-B'
OPTIMIZER_OPTIONS = {'disp': True, 'maxiter': 5000, 'ftol': 1e-4, 'gtol': 1e-5}

# --- Helper lists/sets ---
POS_COLS_SET = {'tx', 'ty', 'tz'} # Use a set for faster lookup

print(f"--- Configuration ---")
print(f"Data Folder: {PARENT_FOLDER_PATH}")
print(f"Alignment Type: {ALIGNMENT_TYPE}")
print(f"Alignment Features: {ALIGNMENT_BASE_FEATURE_COLS}")
print(f"Segmentation Features: {SEGMENTATION_FEATURE_COLS}")
print(f"--- Basis Function Fitting Settings ---")
print(f"Number of Basis Functions (M): {NUM_BASIS_FUNCTIONS}")
print(f"Basis Function Type: {BASIS_FUNCTION_TYPE}")
print(f"Rotation Weight (Cost): {ROTATION_WEIGHT}")
print(f"Max Fit Length (Downsampling): {MAX_FIT_LENGTH}")
print(f"Optimizer Method: {OPTIMIZER_METHOD}")
print("-" * 20)


# --- Data Loading & Alignment (Reused) ---
# (Functions load_selected_data, calculate_derivative, align_trajectories assumed here)
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
    print(f"Downsampling data for fitting: Original length={n_timesteps_orig}, Target max={max_length}, Stride={stride}")
    data_ds = data_array[::stride]
    n_timesteps_ds = data_ds.shape[0]
    print(f"Downsampled length: {n_timesteps_ds}")
    return data_ds, n_timesteps_ds, stride

# --- Basis Function Model and Optimization ---

def unpack_params(params, n_dims, n_basis):
    """Unpacks the flattened parameter vector."""
    M = n_basis
    D = n_dims
    # Order: t_c (M), A (D*M), S (D*M), B (D*M)
    tc = params[0:M]
    A = params[M : M + D*M].reshape(D, M)
    S = params[M + D*M : M + 2*D*M].reshape(D, M)
    B = params[M + 2*D*M : M + 3*D*M].reshape(D, M)
    return tc, A, S, B

def basis_function_model(params, t_axis, n_dims, n_basis, basis_type='tanh'):
    """Calculates the trajectory estimate using sum of basis functions."""
    tc, A, S, B = unpack_params(params, n_dims, n_basis)
    N = len(t_axis)
    y_hat = np.zeros((N, n_dims))

    # Ensure t_axis is column vector for broadcasting
    t_col = t_axis.reshape(-1, 1) # Shape (N, 1)

    for m in range(n_basis):
        # Calculate argument for basis function: S[d, m] * (t - tc[m])
        # tc[m] is scalar, S[:, m] is shape (D,), t_col is (N, 1)
        # Resulting arg shape should be (N, D)
        arg = S[:, m] * (t_col - tc[m]) # Broadcasting happens here

        if basis_type == 'tanh':
            basis_val = np.tanh(arg)
        elif basis_type == 'atan':
            # Scale atan to be roughly comparable to tanh range (-1 to 1)
            basis_val = (2 / np.pi) * np.arctan(arg)
        else:
            raise ValueError("Unsupported basis_function_type")

        # Add contribution: A[d, m] * basis_val + B[d, m]
        # A[:, m] shape (D,), B[:, m] shape (D,)
        # basis_val shape (N, D)
        y_hat += A[:, m] * basis_val + B[:, m]

    return y_hat

def objective_function_basis(params, mean_traj_fit, t_axis_fit, n_dims, n_basis, basis_type, weight_vector):
    """Calculates the weighted SSE between mean trajectory and basis model."""
    # Calculate the model prediction
    y_hat = basis_function_model(params, t_axis_fit, n_dims, n_basis, basis_type)

    # Calculate weighted squared error
    error = mean_traj_fit - y_hat
    weighted_error = error * np.sqrt(weight_vector) # Apply sqrt of weight for SSE calculation
    sse = np.sum(weighted_error**2)

    # Add a small penalty if tc values are too close or out of order? (Optional)
    tc, _, _, _ = unpack_params(params, n_dims, n_basis)
    sorted_tc = np.sort(tc)
    diff_penalty = np.sum(np.exp(-np.diff(sorted_tc)*5)) # Penalize small differences
    order_penalty = np.sum(np.maximum(0, tc - np.roll(tc, -1))[:-1]) # Penalize out-of-order slightly

    # Ensure cost is finite
    if not np.isfinite(sse):
        return np.inf

    # Combine SSE with penalties (tune penalty weights if used)
    cost = sse + diff_penalty * 1e-2 + order_penalty * 1e-2
    return cost
# --- Map Boundaries Back ---
def map_boundaries_to_original(segment_indices_dp, stride, n_timesteps_original):
    """Maps segment boundaries/centers from downsampled scale back to original."""
    # If input is None or empty (e.g., from failed optimization), return empty list
    if segment_indices_dp is None or len(segment_indices_dp) == 0:
        return []
    if stride == 1:
        # Ensure list elements are integers even if no stride
        segment_indices_orig = [int(round(idx)) for idx in segment_indices_dp]
        # Optional: Adjust end point if needed, though less likely needed for centers
        # if segment_indices_orig and segment_indices_orig[-1] != n_timesteps_original:
        #      segment_indices_orig[-1] = n_timesteps_original
        return segment_indices_orig

    segment_indices_orig = [int(round(idx * stride)) for idx in segment_indices_dp]

    # Ensure uniqueness and sort after mapping
    segment_indices_orig = sorted(list(set(segment_indices_orig)))

    # Optional: Clamp values to be within [0, n_timesteps_original]
    segment_indices_orig = [max(0, min(idx, n_timesteps_original)) for idx in segment_indices_orig]
    segment_indices_orig = sorted(list(set(segment_indices_orig))) # Re-sort/unique after clamp

    # print(f"Mapped DP boundaries/centers back to original scale: {segment_indices_orig}")
    return segment_indices_orig

def fit_basis_model(mean_traj_fit, n_timesteps_fit, n_dims, n_basis, basis_type, weight_vector):
    """Optimizes parameters for the basis function model."""
    print(f"\n--- Fitting {n_basis} '{basis_type}' Basis Functions ---")
    fit_start_time = time.time()

    # --- Initial Guess (x0) ---
    M = n_basis
    D = n_dims
    param_size = M + 3 * D * M

    # tc: Uniformly spaced in the *downsampled* time range
    initial_tc = np.linspace(0, n_timesteps_fit - 1, M + 2)[1:-1]
    # A, S, B: Initialize based on `data range
    data_range = np.ptp(mean_traj_fit, axis=0) # Peak-to-peak range for each dimension
    initial_A = np.tile(data_range / (2 * M), (M, 1)).T # Distribute half-range over basis funcs (D, M)
    initial_S = np.full((D, M), 1.0)  # Keep initial steepness simple
    # Initial B: Start near the overall mean, distributed
    initial_B = np.tile(np.mean(mean_traj_fit, axis=0) / M, (M, 1)).T # (D, M)

    # Handle dimensions with zero range (avoid NaN/errors)
    initial_A[np.isnan(initial_A)] = 0.1
    initial_A[data_range == 0, :] = 0.1 

    x0 = np.concatenate([
        initial_tc.flatten(),
        initial_A.flatten(),
        initial_S.flatten(),
        initial_B.flatten()
    ])

    # --- Bounds ---
    bounds = []
    # Bounds for tc (shared centers) - within downsampled time range
    bounds.extend([(0, n_timesteps_fit - 1)] * M)
    # Bounds for A (amplitude) - allow negative/positive
    bounds.extend([(-np.inf, np.inf)] * (D * M))
    # Bounds for S (steepness) - must be positive
    bounds.extend([(1e-6, np.inf)] * (D * M)) # Small positive lower bound
    # Bounds for B (bias) - allow negative/positive
    bounds.extend([(-np.inf, np.inf)] * (D * M))

    # --- Optimization ---
    t_axis_fit = np.arange(n_timesteps_fit)
    result = minimize(
        objective_function_basis,
        x0,
        args=(mean_traj_fit, t_axis_fit, n_dims, n_basis, basis_type, weight_vector),
        method=OPTIMIZER_METHOD,
        bounds=bounds,
        options=OPTIMIZER_OPTIONS
    )

    print(f"Optimization finished in {time.time() - fit_start_time:.2f} seconds.")
    if result.success:
        print(f"Optimization successful. Final cost: {result.fun:.4f}")
        optimized_params = result.x
        # Extract the optimized tc values (centers)
        optimized_tc, _, _, _ = unpack_params(optimized_params, n_dims, n_basis)
        # Sort the centers as they are the segmentation points
        segment_centers_dp_scale = sorted(optimized_tc)
        return segment_centers_dp_scale, optimized_params, result.fun
    else:
        print(f"Optimization failed: {result.message}")
        return None, None, np.inf


# --- Plotting for Basis Function Fit ---
def plot_basis_fit_segmentation(mean_traj_orig, optimized_params, segment_centers_orig,
                                feature_names, basis_type, title="Basis Function Fit & Segmentation"):
    """Plots the original mean trajectory, the fitted model, and segment centers."""
    n_timesteps_orig, n_dims = mean_traj_orig.shape
    time_axis_orig = np.arange(n_timesteps_orig)
    n_basis = len(segment_centers_orig) # Number of centers found

    if optimized_params is None:
        print("Cannot plot fit: Optimization failed.")
        return

    # Calculate the fitted model on the ORIGINAL time axis
    y_hat_orig = basis_function_model(optimized_params, time_axis_orig, n_dims, n_basis, basis_type)

    n_cols = min(n_dims, 3)
    n_rows = math.ceil(n_dims / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=16)
    axs_flat = axs.flatten()
    sns.set_style("whitegrid")

    for d in range(n_dims):
        ax = axs_flat[d]
        # Plot the original mean trajectory
        ax.plot(time_axis_orig, mean_traj_orig[:, d], color='grey', linewidth=1.5, label='Mean Traj.', alpha=0.7)
        # Plot the fitted model
        ax.plot(time_axis_orig, y_hat_orig[:, d], color='purple', linewidth=2.0, label=f'Fitted {basis_type} Sum')

        # Plot segmentation boundaries (centers tc)
        plotted_boundary_label = False
        for i, center_orig in enumerate(segment_centers_orig):
             # Don't plot if center is outside reasonable range (e.g., due to mapping)
             if 0 < center_orig < n_timesteps_orig:
                 label = 'Segment Center ($t_c$)' if not plotted_boundary_label else ""
                 ax.axvline(x=center_orig, color='black', linestyle='--', linewidth=1.5, label=label)
                 plotted_boundary_label = True

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

    # --- Downsample Mean Trajectory for Fitting ---
    mean_traj_fit, n_timesteps_fit, stride = downsample_data(mean_traj_orig, MAX_FIT_LENGTH)

    # --- Prepare Weight Vector ---
    weight_vector = np.ones(n_dims)
    for d, name in enumerate(SEGMENTATION_FEATURE_COLS):
        if name not in POS_COLS_SET:
            weight_vector[d] = ROTATION_WEIGHT
    print(f"Using weight vector: {weight_vector}")

    # --- Fit Basis Function Model ---
    segment_centers_dp_scale, optimized_params, final_cost = fit_basis_model(
        mean_traj_fit, n_timesteps_fit, n_dims, NUM_BASIS_FUNCTIONS,
        BASIS_FUNCTION_TYPE, weight_vector
    )

    if segment_centers_dp_scale is not None:
        # --- Map segment centers back to original scale ---
        segment_centers_orig = map_boundaries_to_original(
            segment_centers_dp_scale, stride, n_timesteps_orig
        )
        # Add start and end points for consistency if needed for interpretation
        segment_boundaries_orig = sorted(list(set([0] + segment_centers_orig + [n_timesteps_orig])))


        print(f"\nBasis Function Segmentation Results:")
        print(f"  Basis Function Centers (Original Scale): {segment_centers_orig}")
        # print(f"  Resulting Boundaries (incl. 0, end): {segment_boundaries_orig}")
        print(f"  Final Optimized Cost: {final_cost:.4f}")

        # --- Plot Results ---
        print("\n--- Plotting Fit and Segmentation ---")
        plot_basis_fit_segmentation(
            mean_traj_orig,
            optimized_params,
            segment_centers_orig, # Plot the centers
            SEGMENTATION_FEATURE_COLS,
            BASIS_FUNCTION_TYPE,
            title=f'{NUM_BASIS_FUNCTIONS} x {BASIS_FUNCTION_TYPE} Fit & Segment Centers (RotW={ROTATION_WEIGHT}, FitLen={MAX_FIT_LENGTH})'
        )
    else:
        print("\nSegmentation failed because basis function fitting did not succeed.")


    print(f"\n--- Script Finished in {time.time() - overall_start_time:.2f} seconds ---")

