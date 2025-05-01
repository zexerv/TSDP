# -*- coding: utf-8 -*-
"""
Trajectory Segmentation using Iterative Refinement and Optimization.

This script takes aligned trajectory data (obtained from pvtw.py) and 
represents it as a simplified sequence of connected line segments (keyframes).
It iteratively adds keyframes and optimizes their positions and times 
to minimize a cost function reflecting data fit and complexity.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os # Used for path joining, though pvtw handles most file ops

# --- Import the user's module for DTW/Alignment ---
# Ensure pvtw.py is in the same directory or Python path
try:
    import pvtw
except ImportError:
    print("Error: Could not import 'pvtw.py'. Make sure the file is in the same directory.")
    exit()

# --- Configuration for Data Loading and Alignment (using pvtw) ---
# !!! IMPORTANT: Update this path to your actual data folder !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button'

# --- Options for pvtw.load_selected_data ---
LOAD_ALL_FILES = True
FILE_LIST = [] # Ignored if LOAD_ALL_FILES is True

# --- Options for pvtw.align_trajectories ---
# Options: 'DTW' (position only), 'DDTW' (derivative only), 'Combined' (pos + deriv)
ALIGNMENT_TYPE = 'Combined' 
# Features used by pvtw to calculate alignment distance
ALIGNMENT_BASE_FEATURE_COLS = ['tx', 'ty', 'tz'] 
# Normalization flag passed to pvtw.align_trajectories for 'Combined' type
NORMALIZE_FOR_COMBINED = True 

# --- Configuration for Segmentation Algorithm ---
# Features to use for calculating segmentation cost (deviation from demos)
# Often the same as ALIGNMENT_BASE_FEATURE_COLS, but could be different
SEGMENTATION_FEATURE_COLS = ['tx', 'ty', 'tz'] 

MAX_POINTS = 15        # Maximum number of keyframes (segments = K-1)
COST_CONVERGENCE_THRESHOLD = 1e-5 # Stop if relative cost reduction is below this
INITIAL_POINTS = 2     # Start with start and end points
OPTIMIZER_METHOD = 'L-BFGS-B' # Optimizer supporting bounds (essential for time)
# OPTIMIZER_METHOD = 'SLSQP' # Another option supporting bounds and constraints

# --- Cost Function Weights ---
# Weight for penalty encouraging points to stay separated in time
LAMBDA_PROXIMITY = 0.01  
# Minimum desired time steps between points (penalty incurred below this)
MIN_TIME_DIFF = 5       
# Weight for penalty based on the number of points (controls complexity)
LAMBDA_COMPLEXITY = 0.2 

# --- Dimension Scaling for Segmentation Cost ---
# Apply specific scales to dimensions in the cost function?
# If False, assumes dimensions are equally important (or already scaled appropriately).
# If True, uses the scales defined below.
SCALE_DIMS_IN_COST = False   
# Define scales if SCALE_DIMS_IN_COST is True (example)
# These scales multiply the squared error for each dimension.
DIM_SCALES = {'tx': 1.0, 'ty': 1.0, 'tz': 2.0} # Example: Make 'tz' error contribute more

# --- Core Segmentation Functions ---

def interpolate_segmented_trajectory(points, T):
    """
    Interpolates a segmented trajectory defined by key points over T steps.

    Args:
        points (np.ndarray): Array of keyframes, shape (K, 1 + N_dims). 
                             Column 0 is time, others are spatial dimensions.
        T (int): Total number of time steps for the output trajectory.

    Returns:
        np.ndarray: The interpolated trajectory, shape (T, N_dims).
    """
    if points.shape[0] < 2:
        print("Warning: Cannot interpolate with fewer than 2 points. Returning constant.")
        if points.shape[0] == 1:
             return np.repeat(points[:, 1:], T, axis=0)
        else: # 0 points
             # Need to know N_dims. This case should ideally not happen if called correctly.
             # Returning zeros as a fallback. Determine N_dims if possible.
             print("Error: interpolate_segmented_trajectory called with 0 points.")
             return np.zeros((T, 1)) # Adjust dimension if known

    times = points[:, 0]
    values = points[:, 1:] # Shape (K, N_dims)
    N_dims = values.shape[1]
    
    # Ensure times are sorted (optimizer bounds should handle this, but safety check)
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx, :]

    # Use linear interpolation. interp1d requires unique time points.
    unique_times, unique_indices = np.unique(times, return_index=True)
    
    if len(unique_times) < 2: 
        # Handle cases where optimization might collapse points to the same time
        print("Warning: Fewer than 2 unique time points after sorting/unique. Returning constant.")
        # Use the value corresponding to the first unique time point
        return np.repeat(values[unique_indices[0]:unique_indices[0]+1, :], T, axis=0) 
        
    # Create the interpolation function
    f_interp = interp1d(
        unique_times, 
        values[unique_indices, :], 
        axis=0, 
        kind='linear', 
        bounds_error=False, 
        # Fill with start/end values for times outside the keyframe range
        fill_value=(values[unique_indices[0], :], values[unique_indices[-1], :])
    )
                        
    # Query at the original discrete time steps
    t_query = np.linspace(0, T - 1, T) 
    interpolated_values = f_interp(t_query)
    
    return interpolated_values # Shape (T, N_dims)

def calculate_cost(params, K, N_dims, T, all_demos_np, 
                   fixed_start_val, fixed_end_val, 
                   lambda_prox, min_time_diff, lambda_comp, 
                   dim_scales_array, segment_feature_indices):
    """
    Calculates the total cost for the optimizer. 
    The cost includes data fit error, time proximity penalty, and complexity penalty.

    Args:
        params (np.ndarray): Flat array containing the variables to optimize:
                             [t_2, ..., t_{K-1}, x_2, y_2, z_2, ..., x_{K-1}, y_{K-1}, z_{K-1}]
                             (Internal times followed by internal spatial values).
        K (int): Current number of keyframes.
        N_dims (int): Number of spatial dimensions being optimized/segmented.
        T (int): Total number of time steps in the demonstrations.
        all_demos_np (list): List of demonstration trajectories (np.ndarray, shape (T, N_all_features)).
                             Note: Contains ALL features, but cost uses only selected ones.
        fixed_start_val (np.ndarray): Spatial values of the fixed start point (shape N_dims).
        fixed_end_val (np.ndarray): Spatial values of the fixed end point (shape N_dims).
        lambda_prox (float): Weight for the proximity penalty.
        min_time_diff (float): Minimum desired time separation between points.
        lambda_comp (float): Weight for the complexity penalty.
        dim_scales_array (np.ndarray): Array of scales for each dimension (shape (1, N_dims)).
        segment_feature_indices (list): Indices of the columns in all_demos_np used for cost calculation.


    Returns:
        float: The total calculated cost.
    """
    N_demos = len(all_demos_np)
    if N_demos == 0: return 0.0 # Or raise error
    if K <= 1: return np.inf # Cannot calculate cost meaningfully

    N_internal = K - 2
    
    # --- Reconstruct the 'points' array from params and fixed points ---
    points = np.zeros((K, 1 + N_dims))
    
    # Fixed start and end points
    points[0, 0] = 0             # t_1 = 0
    points[0, 1:] = fixed_start_val 
    points[K - 1, 0] = T - 1       # t_K = T-1
    points[K - 1, 1:] = fixed_end_val 

    # Fill internal points from optimization parameters
    if K > 2:
      internal_times = params[:N_internal]
      internal_values = params[N_internal:].reshape((N_internal, N_dims))
      points[1:K-1, 0] = internal_times
      points[1:K-1, 1:] = internal_values
      
    # --- Sort points by time ---
    # Crucial if the optimizer doesn't strictly enforce bounds, or for interpolation.
    # However, optimizing based on sorted points makes gradients complex for the optimizer.
    # Relying on optimizer bounds is generally preferred. We sort here primarily
    # for the interpolation and proximity penalty calculation steps.
    sort_idx = np.argsort(points[:, 0])
    points_sorted = points[sort_idx, :]

    # --- 1. Data Fit Cost ---
    # Interpolate the trajectory based on the *potentially* re-ordered points
    interp_traj = interpolate_segmented_trajectory(points_sorted, T) # Shape (T, N_dims)
    
    total_data_fit_cost = 0
    for demo_full in all_demos_np: # demo_full shape (T, N_all_features)
        # Select only the features relevant for segmentation cost
        # demo_segment_features = demo_full[:, segment_feature_indices] # Shape (T, N_dims)
        # Select columns by integer location using .iloc and convert to numpy array
        demo_segment_features = demo_full.iloc[:, segment_feature_indices].values # Shape (T, N_dims)
        diff = interp_traj - demo_segment_features # Shape (T, N_dims)
        
        # Apply dimension scaling before squaring error
        scaled_diff_sq = (diff**2) * dim_scales_array # Broadcasting dim_scales_array (1, N_dims)
        
        cost_per_demo = np.sum(scaled_diff_sq) # Sum squared errors over time and dimensions
        total_data_fit_cost += cost_per_demo
        
    # Average cost over all demonstrations
    total_data_fit_cost /= N_demos 

    # --- 2. Proximity Penalty ---
    # Calculate based on the sorted points to ensure correct time differences
    time_diffs = np.diff(points_sorted[:, 0])
    # Penalize time differences smaller than min_time_diff
    # Use squared penalty for smooth gradient
    prox_penalty = np.sum(np.maximum(0, min_time_diff - time_diffs)**2)

    # --- 3. Complexity Penalty ---
    # Simple penalty based on the number of points
    comp_penalty = K

    # --- Total Cost ---
    total_cost = total_data_fit_cost + lambda_prox * prox_penalty + lambda_comp * comp_penalty
    
    # Optional: Print cost breakdown during optimization for debugging
    # print(f"K={K}, DataFit={total_data_fit_cost:.3f}, ProxPen={prox_penalty:.3f}, CompPen={comp_penalty:.3f}, Total={total_cost:.3f}")

    return total_cost

def optimize_points(points_initial, T, all_demos_np, 
                    lambda_prox, min_time_diff, lambda_comp, 
                    dim_scales_array, segment_feature_indices, optimizer_method):
    """
    Optimizes the internal keyframes (times and spatial values).

    Args:
        points_initial (np.ndarray): Initial guess for keyframes (K, 1 + N_dims).
        T (int): Total number of time steps.
        all_demos_np (list): List of demonstration trajectories (np.ndarray).
        lambda_prox, min_time_diff, lambda_comp: Cost function parameters.
        dim_scales_array (np.ndarray): Dimension scales for cost calculation.
        segment_feature_indices (list): Indices of features used for cost.
        optimizer_method (str): Name of the scipy optimizer to use (e.g., 'L-BFGS-B').


    Returns:
        np.ndarray: The optimized keyframes array (K, 1 + N_dims).
    """
    K = points_initial.shape[0]
    N_dims = points_initial.shape[1] - 1
    
    if K <= 2: 
        print("Optimization skipped: Only 2 points (start/end).")
        return points_initial # Nothing to optimize

    N_internal = K - 2
    
    # --- Prepare Initial Guess and Bounds for Optimizer ---
    # Parameters to optimize: [t_2, ..., t_{K-1}, x_2, y_2, z_2, ..., x_{K-1}, y_{K-1}, z_{K-1}]
    initial_params = np.concatenate([
        points_initial[1:K-1, 0],                   # Internal times: t_2, ..., t_{K-1}
        points_initial[1:K-1, 1:].flatten()         # Internal values flattened: x_2, y_2, z_2, ...
    ])

    # Bounds are crucial, especially for time to maintain order t_1 < t_2 < ... < t_K
    bounds = []
    eps = 1e-3 # Small epsilon to prevent times being exactly equal

    # Time bounds for t_i (where i is index in internal points, so actual point is i+1)
    last_time_bound = 0.0 # This represents t_1
    for i in range(N_internal):
        # Lower bound for t_{i+2} is t_{i+1} + eps
        lower_t_bound = last_time_bound + eps
        
        # Upper bound for t_{i+2} is t_{i+3} - eps (or T-1 for the last internal point)
        # We use the initial times to set the upper bound relative to the *next* point's initial time
        upper_t_bound = points_initial[i+2, 0] - eps
        
        # Ensure bounds are valid (lower < upper)
        if lower_t_bound >= upper_t_bound:
             # This can happen if initial points are too close. Adjust slightly.
             mid_point = (last_time_bound + points_initial[i+2, 0]) / 2.0
             lower_t_bound = mid_point - eps
             upper_t_bound = mid_point + eps
             print(f"Warning: Adjusting tight time bounds for internal point {i+1}")

        bounds.append((lower_t_bound, upper_t_bound))
        
        # Update the lower bound for the *next* iteration
        # Use the *optimized* time from this iteration if possible? No, stick to initial structure for bounds.
        # We rely on the optimizer respecting the *current* point's bounds.
        # The lower bound for point i+3 depends on point i+2's *optimized* value.
        # L-BFGS-B handles this implicitly via box constraints.
        # We set the lower bound based on the previous point's *initial* time for simplicity.
        last_time_bound = points_initial[i+1, 0] # Use initial time t_{i+2} as reference for next lower bound

        # --- Clip initial guess to be strictly within bounds ---
        initial_params[i] = np.clip(initial_params[i], lower_t_bound + eps, upper_t_bound - eps)


    # Value bounds (spatial dimensions) - Example: -100 to 100. Adjust based on your data range.
    # Determine min/max from data for better bounds if needed.
    min_val = -100.0 
    max_val = 100.0
    value_bounds = [(min_val, max_val)] * (N_internal * N_dims)
    bounds.extend(value_bounds)

    # Extract fixed start/end values needed by the cost function wrapper
    fixed_start_val = points_initial[0, 1:]
    fixed_end_val = points_initial[-1, 1:]

    # --- Run the Optimizer ---
    print(f"Starting optimization for K={K} using {optimizer_method}...")
    result = minimize(
        calculate_cost,
        initial_params,
        args=(K, N_dims, T, all_demos_np, 
              fixed_start_val, fixed_end_val, 
              lambda_prox, min_time_diff, lambda_comp, 
              dim_scales_array, segment_feature_indices),
        method=optimizer_method,
        bounds=bounds,
        options={'disp': False, 'maxiter': 500, 'ftol': 1e-7, 'gtol': 1e-5} # Adjust options as needed
        # disp=True shows convergence messages
        # maxiter might need increasing for complex problems
        # ftol/gtol control tolerance for termination
    )

    if not result.success:
        print(f"Warning: Optimization potentially failed or did not fully converge for K={K}.")
        print(f"Message: {result.message}")
    else:
        print(f"Optimization successful for K={K}. Function evaluations: {result.nfev}, Iterations: {result.nit}")


    # --- Reconstruct the optimized points array ---
    optimized_points = np.zeros_like(points_initial)
    # Keep original start and end points
    optimized_points[0, :] = points_initial[0, :]
    optimized_points[K-1, :] = points_initial[K-1, :]
    
    # Fill in optimized internal points
    if K > 2:
      optimized_params = result.x
      optimized_points[1:K-1, 0] = optimized_params[:N_internal] # Optimized times
      optimized_points[1:K-1, 1:] = optimized_params[N_internal:].reshape((N_internal, N_dims)) # Optimized values
      
    # --- Final Sort by Time ---
    # Ensure the final result is strictly sorted by time, even if optimizer was near bounds.
    sort_idx = np.argsort(optimized_points[:, 0])
    optimized_points_sorted = optimized_points[sort_idx, :]

    # --- Check for time collapses after optimization ---
    time_diffs_final = np.diff(optimized_points_sorted[:, 0])
    if np.any(time_diffs_final < 1e-6):
         print(f"Warning: Some points have nearly identical times after optimization for K={K}.")
         # Could implement merging here if desired, but for now just warn.

    return optimized_points_sorted


# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- 1. Load and Align Data using pvtw ---
    print("-" * 30)
    print("Step 1: Loading and Aligning Data using pvtw")
    print("-" * 30)
    print(f"Loading data from: {PARENT_FOLDER_PATH}")
    
    # Call pvtw function to load data
    original_trajs_df, loaded_files = pvtw.load_selected_data(
        PARENT_FOLDER_PATH,
        load_all=LOAD_ALL_FILES,
        file_list=FILE_LIST
        # Assuming pvtw.load_selected_data loads all necessary columns
        # based on its internal logic or default behavior.
    )
    
    if not original_trajs_df:
        print("Exiting: No trajectories loaded via pvtw.")
        exit()
    else:
        print(f"Loaded {len(original_trajs_df)} trajectories using pvtw: {loaded_files}")

    # Call pvtw function to align trajectories
    print(f"\nPerforming {ALIGNMENT_TYPE} alignment using pvtw...")
    print(f"Alignment features: {ALIGNMENT_BASE_FEATURE_COLS}")
    if ALIGNMENT_TYPE == 'Combined':
        print(f"Using normalization for combined alignment: {NORMALIZE_FOR_COMBINED}")
        
    aligned_trajs_df, ref_idx = pvtw.align_trajectories(
        original_trajs_df,
        ALIGNMENT_TYPE,
        ALIGNMENT_BASE_FEATURE_COLS,
        NORMALIZE_FOR_COMBINED
    )

    if not aligned_trajs_df:
        print("Exiting: Alignment via pvtw failed or produced no valid trajectories.")
        exit()
    else:
        print(f"Alignment completed. Reference trajectory index: {ref_idx}")
        print(f"Number of aligned trajectories: {len(aligned_trajs_df)}")


    # --- 2. Prepare Data for Segmentation ---
    print("-" * 30)
    print("Step 2: Preparing Data for Segmentation")
    print("-" * 30)
    
    # Filter out any potentially empty DataFrames and ensure required columns exist
    valid_aligned_trajs_df = [
        df for df in aligned_trajs_df 
        if not df.empty and all(c in df.columns for c in SEGMENTATION_FEATURE_COLS)
    ] 
    
    if not valid_aligned_trajs_df:
        print(f"Exiting: No valid aligned trajectories found with required columns {SEGMENTATION_FEATURE_COLS}.")
        exit()
    else:
         print(f"Using {len(valid_aligned_trajs_df)} valid trajectories for segmentation.")
         print(f"Segmentation features: {SEGMENTATION_FEATURE_COLS}")

    # Convert the relevant columns to a list of NumPy arrays
    aligned_trajs_np = [df[SEGMENTATION_FEATURE_COLS].values for df in valid_aligned_trajs_df]

    # Get dimensions and indices
    T = aligned_trajs_np[0].shape[0] # Number of time steps (should be same for all)
    N_dims_segment = aligned_trajs_np[0].shape[1] # Number of features used for segmentation cost
    N_demos = len(aligned_trajs_np)
    
    # We need the original full data for cost calculation if SEGMENTATION_FEATURE_COLS
    # is different from what's stored in aligned_trajs_np (though usually they are the same).
    # For simplicity, assume aligned_trajs_np contains exactly the features needed.
    # If cost needs features NOT in SEGMENTATION_FEATURE_COLS, adjust data prep.
    
    # Find the indices corresponding to SEGMENTATION_FEATURE_COLS in the DataFrame columns
    # Assumes all DataFrames in valid_aligned_trajs_df have the same columns
    all_df_columns = list(valid_aligned_trajs_df[0].columns)
    try:
        segment_feature_indices = [all_df_columns.index(col) for col in SEGMENTATION_FEATURE_COLS]
    except ValueError as e:
        print(f"Error: One of the segmentation features {SEGMENTATION_FEATURE_COLS} not found in DataFrame columns {all_df_columns}.")
        print(e)
        exit()

    # Stack demos into a single array for easier mean calculation (optional)
    # Note: Cost function currently iterates through the list `aligned_trajs_np`
    all_demos_stack = np.stack([df.values for df in valid_aligned_trajs_df], axis=0) # Shape (N_demos, T, N_all_features)

    # Prepare dimension scales array for cost function
    if SCALE_DIMS_IN_COST:
        dim_scales_array = np.array([DIM_SCALES.get(col, 1.0) for col in SEGMENTATION_FEATURE_COLS]).reshape(1, N_dims_segment)
        print(f"Using dimension scaling in cost: {dim_scales_array}")
    else:
        dim_scales_array = np.ones((1, N_dims_segment)) # Equal scaling
        print("Using equal dimension scaling (1.0) in cost.")


    # --- 3. Iterative Segmentation and Optimization ---
    print("-" * 30)
    print("Step 3: Iterative Segmentation and Optimization")
    print("-" * 30)
    
    # Initialize with start and end points (K=2)
    # Calculate mean start/end positions using only the segmentation features
    p_start_val = np.mean(all_demos_stack[:, 0, segment_feature_indices], axis=0)   
    p_end_val = np.mean(all_demos_stack[:, T-1, segment_feature_indices], axis=0) 
    
    current_points = np.array([
        [0, *p_start_val],          # Time 0, mean start position
        [T - 1, *p_end_val]       # Time T-1, mean end position
    ], dtype=float) # Shape (2, 1 + N_dims_segment)
    
    last_total_cost = np.inf
    results_history = [] # Store results (K, cost, points) for each iteration

    # Calculate initial cost (K=2)
    initial_cost = calculate_cost([], K=2, N_dims=N_dims_segment, T=T, all_demos_np=valid_aligned_trajs_df, # Pass DFs here
                                   fixed_start_val=p_start_val, fixed_end_val=p_end_val, 
                                   lambda_prox=LAMBDA_PROXIMITY, min_time_diff=MIN_TIME_DIFF, 
                                   lambda_comp=LAMBDA_COMPLEXITY, dim_scales_array=dim_scales_array,
                                   segment_feature_indices=segment_feature_indices) # Pass indices
    print(f"Initial Cost (K=2): {initial_cost:.4f}")
    results_history.append({'K': 2, 'cost': initial_cost, 'points': current_points.copy()})
    last_total_cost = initial_cost

    # Loop to add points and optimize
    for k_target in range(INITIAL_POINTS + 1, MAX_POINTS + 1): 
        
        print(f"\n--- Iteration: Target K = {k_target} ---")
        
        # --- 3a. Add Points (Bisect Segments) ---
        K_prev = current_points.shape[0]
        if K_prev >= k_target: # Should not happen here, but safety check
             print(f"Warning: Already have {K_prev} points. Skipping addition for K={k_target}.")
             continue
             
        # We will add K_prev - 1 new points by bisecting each existing segment
        new_points_list = [] 

        for i in range(K_prev - 1):
            p1 = current_points[i] # Start point of segment i
            p2 = current_points[i+1] # End point of segment i
            
            # Add the start point of the segment
            new_points_list.append(p1)
            
            # Calculate new midpoint
            t_new = (p1[0] + p2[0]) / 2.0
            
            # Interpolate spatial value linearly at t_new
            time_diff_segment = p2[0] - p1[0]
            ratio = (t_new - p1[0]) / time_diff_segment if time_diff_segment != 0 else 0.5
            val_new = p1[1:] + ratio * (p2[1:] - p1[1:])
            
            p_new = np.array([t_new, *val_new])
            
            # Add the new midpoint
            new_points_list.append(p_new) 

        # Add the very last point of the original trajectory
        new_points_list.append(current_points[-1])

        points_with_added = np.array(new_points_list)
        
        # --- Ensure unique times before optimization ---
        # Floating point precision might make theoretically different times equal
        unique_times, unique_idx = np.unique(points_with_added[:, 0], return_index=True)
        points_initial_for_opt = points_with_added[np.sort(unique_idx)]

        K_current = points_initial_for_opt.shape[0]
        print(f"Added points by bisection. K before optimization: {K_current}")
        
        if K_current <= 2:
            print("Cannot optimize with <= 2 unique points after bisection.")
            current_points = points_initial_for_opt
            continue # Should not happen if T is reasonably large

        # --- 3b. Optimize All Internal Points ---
        optimized_points = optimize_points(
            points_initial_for_opt, T, valid_aligned_trajs_df, # Pass DFs here
            LAMBDA_PROXIMITY, MIN_TIME_DIFF, LAMBDA_COMPLEXITY, 
            dim_scales_array, segment_feature_indices, OPTIMIZER_METHOD # Pass indices
        )

        current_points = optimized_points # Update points for next iteration
        K_final = current_points.shape[0]
        # print(f"Optimization finished. Final K for this iteration: {K_final}")
        # print("Optimized Points:\n", current_points)


        # --- 3c. Calculate Final Cost for this K ---
        # Need to reconstruct the 'params' array from optimized internal points to call cost function
        final_params = np.concatenate([
             current_points[1:K_final-1, 0],                   # t_2, ..., t_{K-1}
             current_points[1:K_final-1, 1:].flatten()         # x_2, y_2, z_2, ...
        ]) if K_final > 2 else np.array([])

        current_total_cost = calculate_cost(
             final_params, K_final, N_dims_segment, T, valid_aligned_trajs_df, # Pass DFs
             current_points[0, 1:], current_points[-1, 1:], # Use actual optimized start/end if they were opt vars (they are not here)
             LAMBDA_PROXIMITY, MIN_TIME_DIFF, LAMBDA_COMPLEXITY, 
             dim_scales_array, segment_feature_indices # Pass indices
        )

        print(f"==> Final Cost for K={K_final}: {current_total_cost:.4f}")
        results_history.append({'K': K_final, 'cost': current_total_cost, 'points': current_points.copy()})

        # --- Check Convergence ---
        relative_cost_change = abs(last_total_cost - current_total_cost) / (abs(last_total_cost) + 1e-9)
        print(f"Relative cost change: {relative_cost_change:.6f}")
        if relative_cost_change < COST_CONVERGENCE_THRESHOLD and K_final > INITIAL_POINTS:
            print(f"\nConvergence threshold ({COST_CONVERGENCE_THRESHOLD}) reached. Stopping iteration.")
            break 

        last_total_cost = current_total_cost


    # --- 4. Select Best Result ---
    print("-" * 30)
    print("Step 4: Selecting Final Result")
    print("-" * 30)
    
    if not results_history:
         print("Error: No results were generated.")
         exit()
         
    # Simple selection: Use the result from the last successful iteration
    best_result = min(results_history, key=lambda x: x['cost'])

    # Alternative: Implement Elbow method or other model selection criteria here
    # e.g., find K where cost reduction significantly slows down.
    
    print(f"Selected result from iteration with K = {best_result['K']}")
    print(f"Final Cost: {best_result['cost']:.4f}")
    final_segmented_points = best_result['points']
    print("Final Keyframes (Time, Features...):")
    print(final_segmented_points)


    # --- 5. Visualization ---
    print("-" * 30)
    print("Step 5: Visualization")
    print("-" * 30)
    
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style

    # --- Plot 1: Aligned Trajectories and Final Segmentation ---
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    n_plot_dims = N_dims_segment # Plot all segmented dimensions
    dim_labels = SEGMENTATION_FEATURE_COLS
    # Use distinct colors for dimensions
    # colors = plt.cm.viridis(np.linspace(0, 0.8, n_plot_dims)) 
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] # Standard matplotlib colors


    # Plot original aligned demos (lightly in the background)
    print("Plotting aligned demonstrations...")
    for i, demo_df in enumerate(valid_aligned_trajs_df):
        demo_np = demo_df[SEGMENTATION_FEATURE_COLS].values
        for j in range(n_plot_dims):
            # Plot only the first few demos or use high transparency if many demos
            alpha_demo = 0.15 if N_demos > 10 else 0.3
            linewidth_demo = 0.8
            ax1.plot(np.arange(T), demo_np[:, j], color='gray', alpha=alpha_demo, linewidth=linewidth_demo, label='_nolegend_')

    # Interpolate the final segmented trajectory
    final_interp = interpolate_segmented_trajectory(final_segmented_points, T)

    # Plot final segmented trajectory and keyframes
    print("Plotting final segmented trajectory...")
    K_final_plot = final_segmented_points.shape[0]
    for j in range(n_plot_dims):
        # Plot the interpolated line segments
        ax1.plot(np.arange(T), final_interp[:, j], color=colors[j % len(colors)], 
                 linewidth=2.5, label=f'Segmented {dim_labels[j]}')
        # Plot the keyframes as points
        ax1.scatter(final_segmented_points[:, 0], final_segmented_points[:, j+1], 
                    c=[colors[j % len(colors)]], marker='o', s=60, zorder=5, 
                    edgecolors='black', label=f'_nolegend_') # Keyframes

    ax1.set_title(f'Aligned Demonstrations and Final Segmented Trajectory (K={K_final_plot})', fontsize=16)
    ax1.set_xlabel('Aligned Time Step', fontsize=12)
    ax1.set_ylabel('Feature Value', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.savefig("segmentation_result.png") # Optional: Save the plot
    
    # --- Plot 2: Cost vs Number of Points (K) ---
    if len(results_history) > 1:
        print("Plotting Cost vs. K...")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ks = [r['K'] for r in results_history]
        costs = [r['cost'] for r in results_history]
        
        # Separate cost components if desired (requires storing them in history)
        # data_fit_costs = [r['data_fit_cost'] for r in results_history] 
        # prox_costs = [r['prox_cost'] for r in results_history]
        # comp_costs = [r['comp_cost'] for r in results_history]
        # ax2.plot(ks, data_fit_costs, marker='.', linestyle='--', label='Data Fit Cost')
        # ax2.plot(ks, prox_costs, marker='.', linestyle='--', label='Proximity Penalty')
        # ax2.plot(ks, comp_costs, marker='.', linestyle='--', label='Complexity Penalty')

        ax2.plot(ks, costs, marker='o', linestyle='-', color='crimson', label='Total Cost')
        
        ax2.set_xlabel('Number of Keyframes (K)', fontsize=12)
        ax2.set_ylabel('Total Cost', fontsize=12)
        ax2.set_title('Cost vs. Number of Keyframes', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        plt.tight_layout()
        # plt.savefig("cost_vs_k.png") # Optional: Save the plot
    else:
        print("Skipping Cost vs. K plot (only one data point).")

    print("\nDisplaying plots...")
    plt.show()

    print("\nSegmentation process finished.")

