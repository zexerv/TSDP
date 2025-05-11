# proof/main_proof.py
import time
import os
import numpy as np 
import pandas as pd

# Import functions from our proof modules
import config_proof as config # Ensure this is the multi-dim config
from data_loader_proof import load_selected_dimensions_from_csv
from alignment_proof import md_dtw_align_trajectories
from tube_calculator_proof import calculate_multidim_tube
from visualization_proof import (plot_1d_tube, plot_all_prefix_sums, 
                                 plot_segment_ols_fit, plot_optimal_segmentation_with_fits,
                                 plot_experiment_summary_metrics)
from prefix_sum_proof import (calculate_all_prefix_sums_multidim, # Updated
                              calculate_C_LS_multidim_weighted) # Updated
from segmentation_pelt_proof import pelt_segmentation 

def downsample_tube_data_multidim(min_vals_NxD, max_vals_NxD, factor):
    """
    Downsamples NxD tube data by averaging 'factor' points along the time axis (axis 0).
    Returns new min_vals_NxD, max_vals_NxD, and the applied factor.
    """
    if not isinstance(min_vals_NxD, np.ndarray) or not isinstance(max_vals_NxD, np.ndarray) \
       or min_vals_NxD.ndim != 2 or max_vals_NxD.ndim != 2:
        print("Error (downsample_multidim): Inputs must be 2D NumPy arrays.")
        return min_vals_NxD, max_vals_NxD, 1 # Return original if invalid input type

    if factor <= 1:
        return min_vals_NxD, max_vals_NxD, 1 # No downsampling needed or invalid factor
    
    N_orig, D = min_vals_NxD.shape
    if N_orig == 0: 
        return np.empty((0,D)), np.empty((0,D)), factor

    N_new = N_orig // factor
    
    if N_new == 0: 
        # print(f"Warning (downsample_multidim): Factor {factor} too large for N_orig {N_orig}. Using N_new=1 if possible.")
        if N_orig > 0: return min_vals_NxD[:1,:], max_vals_NxD[:1,:], N_orig 
        return np.empty((0,D)), np.empty((0,D)), factor

    min_vals_ds_NxD = np.zeros((N_new, D), dtype=min_vals_NxD.dtype)
    max_vals_ds_NxD = np.zeros((N_new, D), dtype=max_vals_NxD.dtype)
    
    for i in range(N_new):
        start = i * factor
        end = start + factor
        min_vals_ds_NxD[i, :] = np.mean(min_vals_NxD[start:end, :], axis=0)
        max_vals_ds_NxD[i, :] = np.mean(max_vals_NxD[start:end, :], axis=0)
        
    # print(f"Downsampled data from ({N_orig}x{D}) to ({N_new}x{D}) with factor={factor}") # Less verbose
    return min_vals_ds_NxD, max_vals_ds_NxD, factor


def run_pelt_experiment_instance_multidim(
    experiment_id,
    current_min_tube_NxD, current_max_tube_NxD, 
    lambda_val, resolution_factor_val,
    # calculate_C_LS_func_ref will be calculate_C_LS_multidim_weighted
    log_R_detailed_history=True):
    """
    Runs a single instance of PELT experiment for multi-dimensional data.
    """
    N_prime, D_prime = current_min_tube_NxD.shape
    if N_prime == 0:
        print(f"  {experiment_id}: Data length N' is 0. Skipping PELT.")
        return None

    print(f"\n--- Running PELT Experiment Instance: {experiment_id} ---")
    print(f"    N' (processed) = {N_prime}, D' = {D_prime}, Lambda = {lambda_val:.4f}, ResolutionFactor = {resolution_factor_val}")
    
    t_ps_start = time.time()
    current_prefix_sums_md = calculate_all_prefix_sums_multidim(current_min_tube_NxD, current_max_tube_NxD)
    t_ps_end = time.time()
    # print(f"    Multi-dim prefix sums calculated in {t_ps_end - t_ps_start:.3f}s.")

    if not current_prefix_sums_md:
        print(f"    Multi-dim prefix sum calculation failed for N'={N_prime}. Skipping PELT.")
        return None

    pelt_run_start_time = time.time()
    
    optimal_endpoints, F_N_cost, max_R_size, R_history = pelt_segmentation(
        N=N_prime,
        lambda_penalty=lambda_val,
        calculate_C_LS_func=calculate_C_LS_multidim_weighted, # Use the MD weighted func
        all_prefix_sums=current_prefix_sums_md,
        original_Y_min_NxD=current_min_tube_NxD, # Pass NxD data
        original_Y_max_NxD=current_max_tube_NxD, # Pass NxD data
        log_R_history=log_R_detailed_history
    )
    pelt_run_end_time = time.time()
    pelt_runtime = pelt_run_end_time - pelt_run_start_time
    
    print(f"    PELT finished in {pelt_runtime:.3f}s.")
    num_segments_found = len(optimal_endpoints) - 1 if optimal_endpoints else 0
    
    R_profile = {}
    if R_history:
        R_profile['start_R_size'] = R_history[0][1] if R_history else None
        R_profile['mid_R_size'] = R_history[len(R_history)//2][1] if R_history else None
        R_profile['end_R_size'] = R_history[-1][1] if R_history else None

    if config.ENABLE_OPTIMAL_PELT_PLOT:
        plot_filename_pelt = (f"pelt_MD_{config.TARGET_DIMENSION if D_prime == 1 else 'MultiD'}_res{resolution_factor_val}_"
                              f"lambda{lambda_val:.3f}_Nprime{N_prime}.png")
        plot_optimal_segmentation_with_fits(
            T_min_values_NxD=current_min_tube_NxD,
            T_max_values_NxD=current_max_tube_NxD,
            optimal_segment_endpoints=optimal_endpoints, 
            all_prefix_sums=current_prefix_sums_md, # Pass MD prefix sums
            lambda_penalty=lambda_val, 
            total_cost_F_N=F_N_cost,
            # target_dimension_name is now implicit in the multi-dim plot (plots selected dims)
            N_prime_effective=N_prime,
            resolution_factor=resolution_factor_val,
            output_filename=plot_filename_pelt
        )

    return {
        "Experiment_ID": experiment_id,
        "Resolution_Factor": resolution_factor_val,
        "N_prime": N_prime,
        "D_prime": D_prime,
        "Lambda": lambda_val,
        "PELT_Runtime_s": pelt_runtime,
        "Num_Segments": num_segments_found,
        "F_N_prime_Cost": F_N_cost,
        "Max_R_Candidates_Size": max_R_size,
        "R_Profile": R_profile,
        "Optimal_Endpoints_Count": len(optimal_endpoints)
    }

def run_multidim_pelt_experiments():
    """
    Orchestrates PELT experiments for multi-dimensional data,
    varying lambda and resolution.
    """
    overall_start_time = time.time()
    print("--- Starting Multi-Dimensional PELT Experiment Pipeline ---")

    # --- 0. Setup: Load full multi-dimensional data once ---
    print("\n--- Initial Setup: Loading and Preparing Full Original Multi-Dimensional Dataset ---")
    raw_multidim_trajs, loaded_fnames = load_selected_dimensions_from_csv(
        parent_folder=config.PARENT_FOLDER_PATH, 
        dimensions_to_use=config.DIMENSIONS_TO_USE
    )
    if not raw_multidim_trajs: print("No multi-D trajectories loaded. Exiting."); return
    print(f"Loaded {len(raw_multidim_trajs)} raw multi-D trajectories. Dimensions: {config.DIMENSIONS_TO_USE}")
    
    aligned_multidim_trajs, ref_idx = md_dtw_align_trajectories(raw_multidim_trajs)
    if not aligned_multidim_trajs: print("Multi-D alignment failed. Exiting."); return
    print(f"Aligned {len(aligned_multidim_trajs)} multi-D trajectories.")
        
    full_min_tube_NxD_orig, full_max_tube_NxD_orig, N_full_orig, D_full_orig = calculate_multidim_tube(aligned_multidim_trajs)
    if full_min_tube_NxD_orig is None or N_full_orig == 0:
        print("Full multi-D tube calculation failed or tube is empty. Exiting."); return
    print(f"Full original multi-D dataset prepared: N_original = {N_full_orig}, D_original = {D_full_orig}")

    all_experiment_results = []
    exp_counter = 1

    # Define experimental parameters
    # resolution_factors_to_test = [1, 2, 4] # Example: Original, N/2, N/4
    resolution_factors_to_test = [config.DEFAULT_RESOLUTION_FACTOR] # For a single run, or iterate
    if hasattr(config, 'EXPERIMENT_RESOLUTION_FACTORS'):
        resolution_factors_to_test = config.EXPERIMENT_RESOLUTION_FACTORS
    
    # lambda_values_to_test = [0.01, 0.1, 0.2, 0.5, 1.0, 5.0] # As requested
    lambda_values_to_test = [config.DEFAULT_LAMBDA_PELT] # For a single run, or iterate
    if hasattr(config, 'EXPERIMENT_LAMBDA_VALUES'):
         lambda_values_to_test = config.EXPERIMENT_LAMBDA_VALUES


    for res_factor in resolution_factors_to_test:
        print(f"\n===== Processing Resolution Factor: {res_factor} =====")
        current_min_tube_NxD_proc, current_max_tube_NxD_proc, _ = downsample_tube_data_multidim(
            full_min_tube_NxD_orig, full_max_tube_NxD_orig, res_factor
        )
        N_processed = current_min_tube_NxD_proc.shape[0]
        if N_processed == 0:
            print(f"  Skipping res_factor {res_factor} due to 0 length after downsampling.")
            continue

        for lambda_val in lambda_values_to_test:
            exp_id = f"MD_Exp{exp_counter}_Res{res_factor}_Lambda{lambda_val:.3f}"
            
            result = run_pelt_experiment_instance_multidim(
                exp_id, current_min_tube_NxD_proc, current_max_tube_NxD_proc,
                lambda_val, res_factor,
                log_R_detailed_history=True # Get detailed R_history for analysis
            )
            if result: 
                all_experiment_results.append(result)
            exp_counter += 1
            print("-" * 70) 

    # --- Output Management: Print Results Table ---
    print("\n\n--- Multi-Dimensional PELT Experiment Results Summary ---")
    if all_experiment_results:
        results_df = pd.DataFrame(all_experiment_results)
        display_cols = ["Experiment_ID", "Resolution_Factor", "N_prime", "D_prime", "Lambda", 
                        "PELT_Runtime_s", "Num_Segments", 
                        "Max_R_Candidates_Size", "R_Profile", "F_N_prime_Cost"]
        actual_display_cols = [col for col in display_cols if col in results_df.columns]
        
        print(results_df[actual_display_cols].to_string(index=False))
        
        results_output_dir = config.RESULTS_DIR_PROOF
        if not os.path.exists(results_output_dir):
            try: os.makedirs(results_output_dir)
            except OSError as e: print(f"Could not create results directory {results_output_dir}: {e}")
        
        if os.path.exists(results_output_dir):
            csv_filename = os.path.join(results_output_dir, f"multidim_pelt_experiments_summary.csv")
            try:
                results_df.to_csv(csv_filename, index=False)
                print(f"\nResults summary also saved to: {csv_filename}")
            except Exception as e:
                print(f"Error saving results summary to CSV: {e}")
        
        # --- Plot Summary Metrics ---
        # The target_dimension_name for summary plots is a bit ambiguous for multi-D.
        # We can use a generic name or the first dimension.
        summary_plot_dim_name = config.DIMENSIONS_TO_USE[0] if config.DIMENSIONS_TO_USE else "MultiDim"
        plot_experiment_summary_metrics(results_df, summary_plot_dim_name)
            
    else:
        print("No results were collected from the multi-dimensional PELT experiments.")
    
    # --- Optional: Other visualizations (controlled by config) ---
    # These would need to be adapted if used, e.g., plot_1d_tube for a specific dimension.
    # For now, the main visualization is plot_optimal_segmentation_with_fits called per experiment.
    
    overall_duration = time.time() - overall_start_time
    print(f"\n--- Multi-Dimensional PELT Experiment Pipeline Finished in {overall_duration:.2f} seconds ---")

if __name__ == "__main__":
    # Ensure config_proof.py is set up for multi-dimensional data
    # (DIMENSIONS_TO_USE, POSITION_DIMENSIONS, ROTATION_DIMENSIONS, weights)
    # And define EXPERIMENT_RESOLUTION_FACTORS and EXPERIMENT_LAMBDA_VALUES in config
    # if you want to run a batch, otherwise it uses defaults (ResFactor=1, Lambda=0.1)
    
    # Example: Add these to config_proof.py for a batch run:
    # config.EXPERIMENT_RESOLUTION_FACTORS = [1, 2]
    # config.EXPERIMENT_LAMBDA_VALUES = [0.01, 0.1, 0.5, 1.0]
    
    # For a single default run based on config:
    if not hasattr(config, 'EXPERIMENT_RESOLUTION_FACTORS'):
        config.EXPERIMENT_RESOLUTION_FACTORS = [config.DEFAULT_RESOLUTION_FACTOR]
    if not hasattr(config, 'EXPERIMENT_LAMBDA_VALUES'):
        config.EXPERIMENT_LAMBDA_VALUES = [config.DEFAULT_LAMBDA_PELT]

    run_multidim_pelt_experiments()
