# proof/plot_cost_landscape_3segments_proof.py
import time
import os
import numpy as np 
import matplotlib.pyplot as plt

# Import functions from our proof modules
import config_proof as config 
from data_loader_proof import load_selected_dimensions_from_csv 
from alignment_proof import md_dtw_align_trajectories 
from tube_calculator_proof import calculate_multidim_tube 
from prefix_sum_proof import (calculate_all_prefix_sums_multidim, 
                              calculate_C_LS_multidim_weighted) 

def _downsample_series_by_averaging(series, factor):
    """Simple downsampling by averaging."""
    if factor <= 1 or not isinstance(series, np.ndarray) or series.size == 0:
        return series
    N_orig = len(series)
    N_new = N_orig // factor
    if N_new == 0: 
        return series[:1] if N_orig > 0 else np.array([]) 
    downsampled = np.zeros(N_new, dtype=series.dtype)
    for i in range(N_new):
        start = i * factor; end = start + factor
        downsampled[i] = np.mean(series[start:end])
    return downsampled

def _downsample_tube_data_multidim(min_vals_NxD, max_vals_NxD, factor):
    """Simple downsampling for NxD tube data by averaging along time axis."""
    if factor <= 1: return min_vals_NxD, max_vals_NxD
    N_orig, D = min_vals_NxD.shape
    if N_orig == 0: return np.empty((0,D)), np.empty((0,D))
    N_new = N_orig // factor
    if N_new == 0: 
        if N_orig > 0: return min_vals_NxD[:1,:], max_vals_NxD[:1,:]
        return np.empty((0,D)), np.empty((0,D))
    min_vals_ds_NxD = np.zeros((N_new, D), dtype=min_vals_NxD.dtype)
    max_vals_ds_NxD = np.zeros((N_new, D), dtype=max_vals_NxD.dtype)
    for i in range(N_new):
        start = i * factor; end = start + factor
        min_vals_ds_NxD[i, :] = np.mean(min_vals_NxD[start:end, :], axis=0)
        max_vals_ds_NxD[i, :] = np.mean(max_vals_NxD[start:end, :], axis=0)
    return min_vals_ds_NxD, max_vals_ds_NxD

def generate_and_plot_cost_landscape_multidim():
    overall_start_time = time.time()
    print("--- Starting 2D Cost Landscape Generation for 3 Segments (Multi-Dimensional) ---")

    LAMBDA_FOR_LANDSCAPE = config.DEFAULT_LAMBDA_PELT 
    DOWNSAMPLING_FACTOR_LANDSCAPE = getattr(config, 'LANDSCAPE_RESOLUTION_FACTOR', 1)
    if DOWNSAMPLING_FACTOR_LANDSCAPE > 1:
        print(f"Using DOWNSAMPLING_FACTOR_LANDSCAPE = {DOWNSAMPLING_FACTOR_LANDSCAPE}")

    SCRIPT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_cost_landscape_multidim")
    if not os.path.exists(SCRIPT_RESULTS_DIR):
        try: os.makedirs(SCRIPT_RESULTS_DIR)
        except OSError as e: print(f"Error creating directory {SCRIPT_RESULTS_DIR}: {e}."); SCRIPT_RESULTS_DIR = "." 

    print("\n--- Step 1: Loading and Preparing Full Original Multi-Dimensional Dataset ---")
    raw_multidim_trajs, _ = load_selected_dimensions_from_csv(
        parent_folder=config.PARENT_FOLDER_PATH, dimensions_to_use=config.DIMENSIONS_TO_USE)
    if not raw_multidim_trajs: print("No multi-D trajectories loaded. Exiting."); return
    
    aligned_multidim_trajs, _ = md_dtw_align_trajectories(raw_multidim_trajs)
    if not aligned_multidim_trajs: print("Multi-D alignment failed. Exiting."); return
    
    min_tube_vals_NxD_orig, max_tube_vals_NxD_orig, N_orig, D_orig = calculate_multidim_tube(aligned_multidim_trajs)
    if min_tube_vals_NxD_orig is None or N_orig == 0:
        print("Full multi-D tube calculation failed or tube is empty. Exiting."); return
    print(f"Full original multi-D dataset prepared: N_original = {N_orig}, D = {D_orig}")

    min_tube_vals_NxD_processed, max_tube_vals_NxD_processed = _downsample_tube_data_multidim(
        min_tube_vals_NxD_orig, max_tube_vals_NxD_orig, DOWNSAMPLING_FACTOR_LANDSCAPE)
    N_actual = min_tube_vals_NxD_processed.shape[0] 
    D_actual = min_tube_vals_NxD_processed.shape[1] 
    if N_actual == 0: print("Error: Downsampling resulted in empty tube data. Exiting."); return
    print(f"Dataset for landscape: N_actual = {N_actual}, D_actual = {D_actual} (ResFactor={DOWNSAMPLING_FACTOR_LANDSCAPE})")
    if N_actual < 3: print(f"Error: N_actual={N_actual} is too short for 3 segments. Exiting."); return

    print("\n--- Step 2: Calculating Multi-Dimensional Prefix Sums ---")
    all_prefix_sums_md = calculate_all_prefix_sums_multidim(min_tube_vals_NxD_processed, max_tube_vals_NxD_processed)
    if not all_prefix_sums_md: print("Multi-dimensional prefix sum calculation failed. Exiting."); return
    print("Successfully calculated multi-dimensional prefix sums.")

    print(f"\n--- Step 3: Calculating TotalCost(s1, s2) for 3 Segments (N_actual={N_actual}, Lambda={LAMBDA_FOR_LANDSCAPE:.4f}) ---")
    cost_grid = np.full((N_actual + 1, N_actual + 1), np.nan) 
    min_total_cost_found = np.inf
    optimal_s1_s2 = (None, None)
    calculation_count = 0
    total_calculations_expected = ((N_actual - 1) * (N_actual - 2)) // 2 if N_actual >=3 else 0

    for s1_1based in range(1, N_actual - 1): 
        for s2_1based in range(s1_1based + 1, N_actual): 
            cost1_total_weighted, _, _ = calculate_C_LS_multidim_weighted(
                1, s1_1based, all_prefix_sums_md, 
                min_tube_vals_NxD_processed, max_tube_vals_NxD_processed)
            cost2_total_weighted, _, _ = calculate_C_LS_multidim_weighted(
                s1_1based + 1, s2_1based, all_prefix_sums_md, 
                min_tube_vals_NxD_processed, max_tube_vals_NxD_processed)
            cost3_total_weighted, _, _ = calculate_C_LS_multidim_weighted(
                s2_1based + 1, N_actual, all_prefix_sums_md, 
                min_tube_vals_NxD_processed, max_tube_vals_NxD_processed)

            if np.isinf(cost1_total_weighted) or np.isinf(cost2_total_weighted) or np.isinf(cost3_total_weighted):
                current_total_cost_with_lambda = np.inf
            else:
                current_total_cost_with_lambda = (cost1_total_weighted + 
                                                  cost2_total_weighted + 
                                                  cost3_total_weighted + 
                                                  (3 * LAMBDA_FOR_LANDSCAPE))
            cost_grid[s2_1based, s1_1based] = current_total_cost_with_lambda
            if current_total_cost_with_lambda < min_total_cost_found:
                min_total_cost_found = current_total_cost_with_lambda
                optimal_s1_s2 = (s1_1based, s2_1based)
            calculation_count += 1
            if calculation_count > 0 and calculation_count % 100000 == 0 : 
                 print(f"  Calculated {calculation_count}/{total_calculations_expected:.0f} cost pairs...")
    
    print(f"Cost calculation complete. Total pairs processed: {calculation_count}")
    if optimal_s1_s2[0] is not None:
        print(f"Minimum cost found: {min_total_cost_found:.4f} at s1={optimal_s1_s2[0]}, s2={optimal_s1_s2[1]} (1-based on processed data N'={N_actual})")
    else:
        print("No valid (s1,s2) pair found with finite cost (or N_actual was too small).")

    print("\n--- Step 4: Visualizing 2D Cost Landscape (Multi-Dimensional Cost) ---")
    if calculation_count == 0 and N_actual >=3 :
        print("No costs calculated, skipping visualization.")
    elif N_actual < 3:
        print(f"N_actual ({N_actual}) is too small for 3 segments. Skipping visualization.")
    else:
        x_contour = np.arange(1, N_actual) 
        y_contour = np.arange(1, N_actual) 
        cost_grid_slice = cost_grid[1:N_actual, 1:N_actual] 

        fig, ax = plt.subplots(figsize=(12, 10))
        finite_costs_in_slice = cost_grid_slice[np.isfinite(cost_grid_slice)]

        if len(finite_costs_in_slice) > 0:
            # --- MODIFIED LEVEL DETERMINATION ---
            # Start with the actual minimum found if it's finite and sensible
            # Ensure min_total_cost_found is considered for the lower bound of levels
            actual_min_cost = min_total_cost_found if np.isfinite(min_total_cost_found) else np.min(finite_costs_in_slice)
            
            # Use percentiles for a robust upper bound, and potentially lower bound if actual_min_cost is an extreme outlier
            percentile_1 = np.percentile(finite_costs_in_slice, 1) if len(finite_costs_in_slice) > 100 else np.min(finite_costs_in_slice)
            percentile_99 = np.percentile(finite_costs_in_slice, 99) if len(finite_costs_in_slice) > 100 else np.max(finite_costs_in_slice)

            cost_min_for_levels = min(actual_min_cost, percentile_1)
            cost_max_for_levels = percentile_99
            
            if cost_min_for_levels >= cost_max_for_levels: 
                cost_max_for_levels = cost_min_for_levels + max(1.0, abs(cost_min_for_levels * 0.1)) 
            
            levels = np.linspace(cost_min_for_levels, cost_max_for_levels, 30) 
            # --- END OF MODIFICATION ---

            contour = ax.contourf(x_contour, y_contour, cost_grid_slice, levels=levels, cmap='viridis', extend='both') 
            fig.colorbar(contour, ax=ax, label=f'Total Weighted Cost(s1,s2) + {3*LAMBDA_FOR_LANDSCAPE:.4f}')

            if optimal_s1_s2[0] is not None:
                ax.plot(optimal_s1_s2[0], optimal_s1_s2[1], 'rX', markersize=12, markeredgecolor='white',
                        label=f'Min Cost ({min_total_cost_found:.2f}) at ({optimal_s1_s2[0]}, {optimal_s1_s2[1]})')
                ax.legend(loc='upper left')
        else:
            ax.text(0.5, 0.5, "No finite costs to plot in slice.", ha='center', va='center', transform=ax.transAxes)

        ax.set_xlabel('Changepoint s1 (1-based, processed data)')
        ax.set_ylabel('Changepoint s2 (1-based, processed data)')
        title = (f'MD Cost Landscape (3 Segments, N\'={N_actual}, D={D_actual}, ResFactor={DOWNSAMPLING_FACTOR_LANDSCAPE})\n'
                 f'$\lambda$={LAMBDA_FOR_LANDSCAPE:.4f}, Dim(s): {config.DIMENSIONS_TO_USE}') # Show used dims
        ax.set_title(title)
        ax.set_xlim(0.5, N_actual - 1.5); ax.set_ylim(1.5, N_actual - 0.5)
        ax.grid(True, linestyle=':', alpha=0.5)

        plot_filename = f"cost_landscape_3seg_MD_Nprime{N_actual}_Res{DOWNSAMPLING_FACTOR_LANDSCAPE}_lambda{LAMBDA_FOR_LANDSCAPE:.4f}.png"
        full_output_path = os.path.join(SCRIPT_RESULTS_DIR, plot_filename)
        try:
            plt.savefig(full_output_path, dpi=150)
            print(f"Cost landscape plot saved to: {full_output_path}")
        except Exception as e:
            print(f"Error saving landscape plot: {e}")
        plt.show()
        plt.close(fig)

    overall_duration = time.time() - overall_start_time
    print(f"\n--- Multi-Dim Cost Landscape Script Finished in {overall_duration:.2f} seconds ---")

if __name__ == "__main__":
    # Ensure config_proof.py has LANDSCAPE_RESOLUTION_FACTOR if you want to control it from there.
    if not hasattr(config, 'LANDSCAPE_RESOLUTION_FACTOR'):
        print("Note: LANDSCAPE_RESOLUTION_FACTOR not found in config_proof.py, script will use its internal default or 1.")
        # The script's DOWNSAMPLING_FACTOR_LANDSCAPE will default to 1 if config.LANDSCAPE_RESOLUTION_FACTOR is missing.
        # You can also directly set DOWNSAMPLING_FACTOR_LANDSCAPE in the script's config section.

    generate_and_plot_cost_landscape_multidim()
