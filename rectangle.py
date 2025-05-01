# -*- coding: utf-8 -*-
"""
Trajectory Segmentation using Iterative Refinement and Optimization 
of Trapezoidal Bounds (Optimizing Time Boundaries Only).

This script takes aligned trajectory data (obtained from pvtw.py), calculates
the min/max envelope of the demonstrations, and represents this envelope using 
a sequence of trapezoids. The vertical bounds of each trapezoid are fixed by 
the envelope at its start/end times. The optimization focuses solely on finding
the optimal time boundaries between segments.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Still useful for filling area
import os 

# --- Import the user's module for DTW/Alignment ---
try:
    import pvtw
except ImportError:
    print("Error: Could not import 'pvtw.py'. Make sure the file is in the same directory.")
    exit()

# --- Configuration for Data Loading and Alignment (using pvtw) ---
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/switch'
LOAD_ALL_FILES = True
FILE_LIST = [] 
ALIGNMENT_TYPE = 'Combined' 
ALIGNMENT_BASE_FEATURE_COLS = ['tx', 'ty', 'tz'] 
NORMALIZE_FOR_COMBINED = True 

# --- Configuration for Trapezoid Segmentation Algorithm ---
# Features to use for calculating the envelope and segmentation cost
SEGMENTATION_FEATURE_COLS = ['tx', 'ty', 'tz'] 

MAX_SEGMENTS = 10     # Maximum number of trapezoid segments to try
COST_CONVERGENCE_THRESHOLD = 1e-5 
INITIAL_SEGMENTS = 1 # Start with one segment
OPTIMIZER_METHOD = 'Nelder-Mead' 

# --- Cost Function Weights ---
# Weight for penalty 1 (Trapezoid area outside envelope)
LAMBDA_PENALTY1 = 1.0 
# Weight for penalty 2 (Envelope area outside trapezoid)
LAMBDA_PENALTY2 = 40.0 
# Weight for complexity penalty (number of segments)
LAMBDA_COMPLEXITY = 0.02 # <<< MODIFIED AS REQUESTED
# Weight for penalty encouraging segments to have minimum duration
LAMBDA_TIME_PROXIMITY = 1
MIN_SEGMENT_DURATION = 40 # Minimum desired time steps per segment
# LAMBDA_BOUND_CROSSING = 10.0 # <<< REMOVED (Bounds fixed by envelope)

# --- Core Segmentation Functions ---

def calculate_envelope(aligned_trajs_df_list, feature_cols):
    """Calculates the min/max envelope across all demonstrations."""
    if not aligned_trajs_df_list:
        return None, None, 0, 0
    try:
        all_demos_np = np.stack([df[feature_cols].values for df in aligned_trajs_df_list], axis=0)
    except ValueError as e:
         print(f"Error stacking trajectories. Do they have the same length and columns? {e}")
         lengths = [len(df) for df in aligned_trajs_df_list]
         if len(set(lengths)) > 1:
             print(f"Error: Trajectories have different lengths after alignment: {lengths}")
         return None, None, 0, 0
    except KeyError as e:
        print(f"Error: Feature columns {feature_cols} not found in all dataframes. {e}")
        return None, None, 0, 0

    T = all_demos_np.shape[1]
    N_dims = all_demos_np.shape[2]
    min_envelope = np.min(all_demos_np, axis=0) # Shape: (T, N_dims)
    max_envelope = np.max(all_demos_np, axis=0) # Shape: (T, N_dims)
    return min_envelope, max_envelope, T, N_dims

def get_trapezoid_bounds_at_time(t, t_start, t_end, y_min_start, y_max_start, y_min_end, y_max_end):
    """Calculates the interpolated trapezoid bounds at a specific time t."""
    # This function remains the same
    if t < t_start or t > t_end:
        return np.nan, np.nan 
    duration = t_end - t_start
    if duration <= 1e-9: # Use small epsilon for float comparison
        return y_min_start, y_max_start
    alpha = (t - t_start) / duration
    y_min_t = y_min_start + alpha * (y_min_end - y_min_start)
    y_max_t = y_max_start + alpha * (y_max_end - y_max_start)
    return y_min_t, y_max_t

def calculate_trapezoid_cost_timeopt(params_time, N_segments, N_dims, T, 
                                     min_envelope, max_envelope, 
                                     lambda_penalty1, lambda_penalty2, lambda_comp,
                                     lambda_time_prox, min_segment_duration): # Removed lambda_bound_crossing
    """
    Calculates the total cost for trapezoid segments where only time boundaries
    are optimized. Vertical bounds are determined by the envelope.

    Args:
        params_time (np.ndarray): Flat array of optimizable time parameters:
                                  [t_end_0, ..., t_end_{N-2}] # Internal time boundaries ONLY
        N_segments (int): The number of trapezoid segments.
        N_dims (int): Number of spatial dimensions.
        T (int): Total number of time steps.
        min_envelope, max_envelope: Envelope arrays (T, N_dims).
        lambda_*: Cost weights.

    Returns:
        float: The total calculated cost.
    """
    if N_segments == 0: return np.inf
    
    # --- Reconstruct Time Boundaries ---
    time_boundaries = np.zeros(N_segments + 1)
    time_boundaries[0] = 0
    time_boundaries[-1] = T - 1 
    if N_segments > 1:
        # params_time contains only the internal time boundaries
        time_boundaries[1:-1] = params_time 
        # Ensure time boundaries are sorted (should be handled by bounds, but safety)
        time_boundaries[1:-1] = np.sort(time_boundaries[1:-1])

    total_penalty1 = 0
    total_penalty2 = 0
    total_time_penalty = 0
    # total_bound_cross_penalty = 0 # Removed

    for i in range(N_segments):
        t_start = time_boundaries[i]
        t_end = time_boundaries[i+1] 
        
        t_start_f = float(t_start)
        t_end_f = float(t_end)
        
        t_start_idx = int(round(t_start_f))
        t_end_idx = int(round(t_end_f)) 
        
        t_start_idx = max(0, t_start_idx)
        t_end_idx = min(T, t_end_idx) 
        
        segment_duration_f = t_end_f - t_start_f

        if segment_duration_f <= 1e-6: 
             continue 

        # Time proximity penalty
        if segment_duration_f < min_segment_duration:
             total_time_penalty += (min_segment_duration - segment_duration_f)**2

        # --- Determine Vertical Bounds from Envelope ---
        # Get indices for start/end times, ensure they are within [0, T-1]
        t_start_bound_idx = min(max(0, t_start_idx), T - 1)
        t_end_bound_idx = min(max(0, int(round(t_end_f))), T - 1) # Use rounded end index

        y_min_start_env = min_envelope[t_start_bound_idx, :] # Shape (N_dims,)
        y_max_start_env = max_envelope[t_start_bound_idx, :] # Shape (N_dims,)
        y_min_end_env = min_envelope[t_end_bound_idx, :]     # Shape (N_dims,)
        y_max_end_env = max_envelope[t_end_bound_idx, :]     # Shape (N_dims,)
        # --- End Determine Vertical Bounds ---

        # Iterate through time steps within the segment
        for t_idx in range(t_start_idx, t_end_idx):
            t_current_f = float(t_idx) 
            
            for d in range(N_dims):
                # Get interpolated trapezoid bounds at this time step using envelope bounds
                y_min_trap_t, y_max_trap_t = get_trapezoid_bounds_at_time(
                    t_current_f, t_start_f, t_end_f, 
                    y_min_start_env[d], y_max_start_env[d], 
                    y_min_end_env[d], y_max_end_env[d]
                )

                # Get envelope values at this time step
                min_env_t = min_envelope[t_idx, d]
                max_env_t = max_envelope[t_idx, d]

                # Calculate penalties
                total_penalty1 += max(0, min_env_t - y_min_trap_t) + max(0, y_max_trap_t - max_env_t)
                total_penalty2 += max(0, y_min_trap_t - min_env_t) + max(0, max_env_t - y_max_trap_t)

    # Normalization factor
    norm_factor = T * N_dims if T * N_dims > 0 else 1
    complexity_penalty = N_segments
    
    total_cost = (lambda_penalty1 * total_penalty1 / norm_factor + 
                  lambda_penalty2 * total_penalty2 / norm_factor + 
                  lambda_comp * complexity_penalty +
                  lambda_time_prox * total_time_penalty) # Removed bound crossing penalty
                  # lambda_bound_crossing * total_bound_cross_penalty) 

    return total_cost


def optimize_trapezoids_timeopt(segments_initial, T, N_dims,
                                min_envelope, max_envelope, 
                                lambda_penalty1, lambda_penalty2, lambda_comp,
                                lambda_time_prox, min_segment_duration, 
                                optimizer_method): # Removed lambda_bound_crossing
    """Optimizes the time boundaries of the given trapezoid segments."""
    
    N_segments = len(segments_initial)
    if N_segments <= 1: # No internal time boundaries to optimize
        print("Optimization skipped: Only 1 segment.")
        return segments_initial 
    
    num_time_params = N_segments - 1

    # --- Prepare Initial Guess and Bounds (Time Only) ---
    initial_params = []
    bounds = []
    
    # 1. Time parameters (internal t_end values)
    initial_time_params = [seg['t_end'] for seg in segments_initial[:-1]]
    initial_params.extend(initial_time_params)
    
    # Time bounds
    last_time_bound = 0.0
    eps = 1.0 # Use 1 time step epsilon
    for i in range(num_time_params):
        # Use the time from the *next* segment's definition for upper bound reference
        t_end_next_i = segments_initial[i+1]['t_end'] 
        lower_t = last_time_bound + eps
        upper_t = t_end_next_i - eps
        if lower_t >= upper_t: 
            mid = (last_time_bound + t_end_next_i) / 2.0
            lower_t = mid - eps
            upper_t = mid + eps
            print(f"Warning: Adjusting tight time bounds for t_end_{i}")
        bounds.append((lower_t, upper_t))
        last_time_bound = initial_time_params[i] 
        # Clip initial guess for time parameter i
        initial_params[i] = np.clip(initial_params[i], lower_t + eps, upper_t - eps)

    # 2. Bound parameters are NOT optimized
    
    initial_params = np.array(initial_params)

    # --- Run Optimizer ---
    print(f"Starting optimization for N={N_segments} segments (Time Only) using {optimizer_method}...")
    result = minimize(
        calculate_trapezoid_cost_timeopt, # Use the modified cost function
        initial_params, # Contains only time parameters
        args=(N_segments, N_dims, T, 
              min_envelope, max_envelope, 
              lambda_penalty1, lambda_penalty2, lambda_comp,
              lambda_time_prox, min_segment_duration), # Pass only relevant args
        method=optimizer_method,
        bounds=bounds, # Contains only time bounds
        options={'disp': False, 'maxiter': 500, 'ftol': 1e-7, 'gtol': 1e-5} # May need fewer iterations now
    )

    if not result.success:
        print(f"Warning: Optimization potentially failed for N={N_segments}.")
        print(f"Message: {result.message}")
    else:
         print(f"Optimization successful for N={N_segments}. Func evals: {result.nfev}, Iters: {result.nit}")

    # --- Reconstruct Optimized Trapezoids ---
    optimized_segments = []
    optimized_time_params = result.x
    
    # Extract optimized time boundaries
    time_boundaries = np.zeros(N_segments + 1)
    time_boundaries[0] = 0
    time_boundaries[-1] = T - 1
    if N_segments > 1:
        time_boundaries[1:-1] = optimized_time_params
        time_boundaries[1:-1] = np.sort(time_boundaries[1:-1]) # Ensure sorted

    # Reconstruct segments using optimized times and envelope bounds
    for i in range(N_segments):
        t_start_opt = time_boundaries[i]
        t_end_opt = time_boundaries[i+1]
        
        # Get indices for start/end times for fetching bounds
        t_start_idx = min(max(0, int(round(t_start_opt))), T - 1)
        t_end_idx = min(max(0, int(round(t_end_opt))), T - 1)

        # Fetch bounds from envelope
        y_min_start_opt = min_envelope[t_start_idx, :]
        y_max_start_opt = max_envelope[t_start_idx, :]
        y_min_end_opt = min_envelope[t_end_idx, :]
        y_max_end_opt = max_envelope[t_end_idx, :]

        optimized_segments.append({
            't_start': t_start_opt,
            't_end': t_end_opt,
            'y_min_start': y_min_start_opt,
            'y_max_start': y_max_start_opt,
            'y_min_end': y_min_end_opt,
            'y_max_end': y_max_end_opt
        })

    return optimized_segments

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- 1. Load, Align Data, Calculate Envelope ---
    print("-" * 30)
    print("Step 1: Load, Align, Calculate Envelope")
    print("-" * 30)
    # ... (Loading and Alignment code identical) ...
    original_trajs_df, loaded_files = pvtw.load_selected_data(
        PARENT_FOLDER_PATH, load_all=LOAD_ALL_FILES, file_list=FILE_LIST)
    if not original_trajs_df: exit()
    
    aligned_trajs_df, ref_idx = pvtw.align_trajectories(
        original_trajs_df, ALIGNMENT_TYPE, ALIGNMENT_BASE_FEATURE_COLS, NORMALIZE_FOR_COMBINED)
    if not aligned_trajs_df: exit()

    valid_aligned_trajs_df = [
        df for df in aligned_trajs_df 
        if not df.empty and all(c in df.columns for c in SEGMENTATION_FEATURE_COLS)
    ] 
    if not valid_aligned_trajs_df: exit()
    print(f"Using {len(valid_aligned_trajs_df)} valid trajectories.")

    min_envelope, max_envelope, T, N_dims = calculate_envelope(valid_aligned_trajs_df, SEGMENTATION_FEATURE_COLS)
    if min_envelope is None: exit()
    print(f"Calculated envelope: T={T}, N_dims={N_dims}")

    # --- 2. Iterative Trapezoid Segmentation (Time Optimization Only) ---
    print("-" * 30)
    print("Step 2: Iterative Trapezoid Segmentation (Time Optimization Only)")
    print("-" * 30)

    # Initialize with one segment (bounds fixed by envelope)
    y_min_global_start = min_envelope[0, :]
    y_max_global_start = max_envelope[0, :]
    y_min_global_end = min_envelope[T-1, :]
    y_max_global_end = max_envelope[T-1, :]
    current_segments = [{
        't_start': 0.0, 't_end': float(T - 1),
        'y_min_start': y_min_global_start, 'y_max_start': y_max_global_start,
        'y_min_end': y_min_global_end, 'y_max_end': y_max_global_end
    }]

    results_history = []
    last_total_cost = np.inf

    # Calculate initial cost (N=1, no time params)
    initial_cost = calculate_trapezoid_cost_timeopt( # Use timeopt cost function
        np.array([]), N_segments=1, N_dims=N_dims, T=T, # Pass empty array for time params
        min_envelope=min_envelope, max_envelope=max_envelope,
        lambda_penalty1=LAMBDA_PENALTY1, lambda_penalty2=LAMBDA_PENALTY2, lambda_comp=LAMBDA_COMPLEXITY,
        lambda_time_prox=LAMBDA_TIME_PROXIMITY, min_segment_duration=MIN_SEGMENT_DURATION
        # Removed lambda_bound_crossing
    )
    print(f"Initial Cost (N=1): {initial_cost:.4f}")
    results_history.append({'N': 1, 'cost': initial_cost, 'segments': current_segments})
    last_total_cost = initial_cost

    # Loop to add segments and optimize
    for n_target in range(INITIAL_SEGMENTS + 1, MAX_SEGMENTS + 1):
        print(f"\n--- Iteration: Target N = {n_target} ---")
        N_prev = len(current_segments)
        if N_prev >= n_target: continue 

        # --- Add Segment (Split longest duration segment) ---
        durations = [seg['t_end'] - seg['t_start'] for seg in current_segments]
        split_idx = np.argmax(durations)
        seg_to_split = current_segments[split_idx]

        t_split = seg_to_split['t_start'] + durations[split_idx] / 2.0
        t_split_idx = int(round(t_split))
        t_split_idx = min(max(1, t_split_idx), T-2) 

        # Get envelope values at split time for initializing new bounds (still needed for split)
        y_min_at_split = min_envelope[t_split_idx, :]
        y_max_at_split = max_envelope[t_split_idx, :]

        # Create two new segments with bounds derived from envelope at start/split/end
        t_start_idx_split = min(max(0, int(round(seg_to_split['t_start']))), T - 1)
        t_end_idx_split = min(max(0, int(round(seg_to_split['t_end']))), T - 1)

        seg1 = {'t_start': seg_to_split['t_start'], 't_end': t_split, 
                'y_min_start': min_envelope[t_start_idx_split,:], 'y_max_start': max_envelope[t_start_idx_split,:],
                'y_min_end': y_min_at_split, 'y_max_end': y_max_at_split} 
        seg2 = {'t_start': t_split, 't_end': seg_to_split['t_end'], 
                'y_min_start': y_min_at_split, 'y_max_start': y_max_at_split, 
                'y_min_end': min_envelope[t_end_idx_split,:], 'y_max_end': max_envelope[t_end_idx_split,:]}

        segments_initial_for_opt = current_segments[:split_idx] + [seg1, seg2] + current_segments[split_idx+1:]
        N_current = len(segments_initial_for_opt)
        print(f"Split longest segment. N before optimization: {N_current}")

        # --- Optimize Segment Times Only ---
        optimized_segments = optimize_trapezoids_timeopt( # Use timeopt optimizer
            segments_initial_for_opt, T, N_dims,
            min_envelope, max_envelope, 
            LAMBDA_PENALTY1, LAMBDA_PENALTY2, LAMBDA_COMPLEXITY,
            LAMBDA_TIME_PROXIMITY, MIN_SEGMENT_DURATION, 
            OPTIMIZER_METHOD
        )

        current_segments = optimized_segments
        N_final = len(current_segments)

        # --- Calculate Final Cost for this N ---
        final_time_params = []
        if N_final > 1:
             final_time_params.extend([seg['t_end'] for seg in current_segments[:-1]])
        
        current_total_cost = calculate_trapezoid_cost_timeopt( # Use timeopt cost function
            np.array(final_time_params), N_segments=N_final, N_dims=N_dims, T=T,
            min_envelope=min_envelope, max_envelope=max_envelope,
            lambda_penalty1=LAMBDA_PENALTY1, lambda_penalty2=LAMBDA_PENALTY2, lambda_comp=LAMBDA_COMPLEXITY,
            lambda_time_prox=LAMBDA_TIME_PROXIMITY, min_segment_duration=MIN_SEGMENT_DURATION
        )

        print(f"==> Final Cost for N={N_final}: {current_total_cost:.4f}")
        results_history.append({'N': N_final, 'cost': current_total_cost, 'segments': current_segments})

        # --- Check Convergence ---
        relative_cost_change = abs(last_total_cost - current_total_cost) / (abs(last_total_cost) + 1e-9)
        print(f"Relative cost change: {relative_cost_change:.6f}")
        if relative_cost_change < COST_CONVERGENCE_THRESHOLD and N_final > INITIAL_SEGMENTS:
            print(f"\nConvergence threshold ({COST_CONVERGENCE_THRESHOLD}) reached. Stopping.")
            break 
        last_total_cost = current_total_cost

    # --- 3. Select Best Result ---
    print("-" * 30)
    print("Step 3: Selecting Final Result")
    print("-" * 30)
    if not results_history: exit()
    best_result = min(results_history, key=lambda x: x['cost'])
    print(f"Selected result with N = {best_result['N']} segments.")
    print(f"Final Minimum Cost: {best_result['cost']:.4f}")
    final_segments = best_result['segments']

    # --- 4. Visualization ---
    # Visualization code remains identical, as it plots based on the
    # 't_start', 't_end', 'y_min_start', etc. fields in the final_segments list.
    print("-" * 30)
    print("Step 4: Visualization")
    print("-" * 30)
    plt.style.use('seaborn-v0_8-whitegrid') 
    
    n_plot_dims = N_dims
    dim_labels = SEGMENTATION_FEATURE_COLS
    fig, axes = plt.subplots(n_plot_dims, 1, figsize=(14, 4 * n_plot_dims), sharex=True)
    if n_plot_dims == 1: axes = [axes] 

    time_axis = np.arange(T)

    print("Plotting envelope and optimized trapezoids...")
    for d in range(n_plot_dims):
        ax = axes[d]
        # Plot min/max envelope
        ax.fill_between(time_axis, min_envelope[:, d], max_envelope[:, d], 
                        color='lightgray', alpha=0.6, label='Demo Envelope')
        ax.plot(time_axis, min_envelope[:, d], color='dimgray', linestyle='--', linewidth=1, label='_nolegend_')
        ax.plot(time_axis, max_envelope[:, d], color='dimgray', linestyle='--', linewidth=1, label='_nolegend_')

        # Plot optimized trapezoids by drawing their boundary lines
        for i, seg in enumerate(final_segments):
            t_start = seg['t_start']
            t_end = seg['t_end']
            
            num_line_points = max(2, int(round(t_end - t_start)) + 1) 
            t_plot = np.linspace(t_start, t_end, num_line_points)
            
            y_min_line = np.linspace(seg['y_min_start'][d], seg['y_min_end'][d], num_line_points)
            y_max_line = np.linspace(seg['y_max_start'][d], seg['y_max_end'][d], num_line_points)

            label_trap = 'Optimized Trapezoid Bounds' if i == 0 else '_nolegend_'
            ax.plot(t_plot, y_min_line, color='red', linestyle='-', linewidth=1.5, label=label_trap)
            ax.plot(t_plot, y_max_line, color='red', linestyle='-', linewidth=1.5, label='_nolegend_')
            ax.fill_between(t_plot, y_min_line, y_max_line, color='red', alpha=0.2, label='_nolegend_')

        ax.set_ylabel(dim_labels[d], fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('Aligned Time Step', fontsize=12)
    fig.suptitle(f'Demonstration Envelope and Optimized Trapezoids (N={best_result["N"]})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
    
    # --- Plot Cost vs Number of Segments (N) ---
    if len(results_history) > 1:
        print("Plotting Cost vs. N...")
        fig_cost, ax_cost = plt.subplots(figsize=(8, 5))
        ns = [r['N'] for r in results_history]
        costs = [r['cost'] for r in results_history]
        ax_cost.plot(ns, costs, marker='o', linestyle='-', color='crimson', label='Total Cost')
        ax_cost.set_xlabel('Number of Segments (N)', fontsize=12)
        ax_cost.set_ylabel('Total Cost', fontsize=12)
        ax_cost.set_title('Cost vs. Number of Segments', fontsize=14)
        ax_cost.grid(True, linestyle='--', alpha=0.6)
        ax_cost.legend()
        plt.tight_layout()

    print("\nDisplaying plots...")
    plt.show()
    print("\nTrapezoid segmentation (time-optimized) process finished.")

