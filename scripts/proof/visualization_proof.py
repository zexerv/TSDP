# proof/visualization_proof.py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
import config_proof as config 

# Import necessary function to get OLS fits for individual dimensions within segments
from prefix_sum_proof import calculate_ls_fit_for_segment_1D 

def plot_1d_tube(min_values, max_values, target_dimension_name, 
                 aligned_trajectories=None, 
                 output_filename="1d_tube_plot.png"):
    # This function is for a single 1D tube. 
    # It might be less relevant for multi-D summary but can be used for debugging a single dim.
    if not config.ENABLE_FULL_TUBE_PLOT: return 
    if min_values is None or max_values is None or len(min_values) == 0: return
    if len(min_values) != len(max_values): return
    num_timesteps = len(min_values); time_axis = np.arange(num_timesteps) 
    plt.style.use('seaborn-v0_8-whitegrid'); plt.rcParams['font.family'] = config.PLOT_FONT
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_axis, max_values, color='lightblue', linestyle='-', linewidth=2, label='Max Bound (Tube)', zorder=2)
    ax.plot(time_axis, min_values, color='lightcoral', linestyle='-', linewidth=2, label='Min Bound (Tube)', zorder=2)
    ax.fill_between(time_axis, min_values, max_values, color='lightgrey', alpha=0.5, label='Tube Range', zorder=1)
    if aligned_trajectories: # Assuming aligned_trajectories are also 1D for this specific plot
        for i, traj in enumerate(aligned_trajectories):
            if traj is not None and len(traj) == num_timesteps:
                ax.plot(time_axis, traj, color='gray', linestyle='--', linewidth=0.7, alpha=0.6, 
                        label='Aligned Trajectory' if i == 0 else "_nolegend_", zorder=3) 
    ax.set_xlabel('Time Step (0-indexed)'); ax.set_ylabel(f'Value of {target_dimension_name}')
    ax.set_title(f'1D Trajectory Tube for Dimension: {target_dimension_name}')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)); ax.legend(by_label.values(), by_label.keys())
    ax.grid(True, linestyle=':', which='major', axis='both'); fig.tight_layout()
    results_path = config.RESULTS_DIR_PROOF
    if not os.path.exists(results_path):
        try: os.makedirs(results_path)
        except OSError: plt.show(); plt.close(fig); return 
    full_output_path = os.path.join(results_path, output_filename)
    try: plt.savefig(full_output_path); print(f"1D Tube plot saved to: {full_output_path}")
    except Exception: pass 
    plt.close(fig) 


def plot_all_prefix_sums(prefix_sums_dict, target_dimension_name_or_index, num_dimensions_total,
                         output_filename="all_prefix_sums.png"):
    # For multi-D, this would plot prefix sums for ONE specified dimension.
    if not config.ENABLE_PREFIX_SUM_PLOTS: return 
    if not prefix_sums_dict: return

    d_idx_to_plot = -1
    plot_title_dim_suffix = ""

    if isinstance(target_dimension_name_or_index, int):
        d_idx_to_plot = target_dimension_name_or_index
        if 0 <= d_idx_to_plot < len(config.DIMENSIONS_TO_USE):
             plot_title_dim_suffix = f" for Dim: {config.DIMENSIONS_TO_USE[d_idx_to_plot]} (d{d_idx_to_plot})"
        else:
             plot_title_dim_suffix = f" for Dim Index: {d_idx_to_plot}"
    elif isinstance(target_dimension_name_or_index, str):
        try:
            d_idx_to_plot = config.DIMENSIONS_TO_USE.index(target_dimension_name_or_index)
            plot_title_dim_suffix = f" for Dim: {target_dimension_name_or_index} (d{d_idx_to_plot})"
        except ValueError:
            print(f"Warning (plot_all_prefix_sums): Dimension name '{target_dimension_name_or_index}' not in config.DIMENSIONS_TO_USE. Cannot plot specific prefix sums.")
            return
    else: # Default to plotting for dimension 0 if specific one not given or invalid
        d_idx_to_plot = 0
        plot_title_dim_suffix = f" for Dim: {config.DIMENSIONS_TO_USE[0]} (d0) (Defaulted)"
        
    if not (0 <= d_idx_to_plot < num_dimensions_total):
        print(f"Warning (plot_all_prefix_sums): d_idx_to_plot {d_idx_to_plot} is out of range for D={num_dimensions_total}. Cannot plot.")
        return

    keys_to_check_templates = ['ps_Y_min_d{}', 'ps_tY_min_d{}', 'ps_YY_min_d{}', 
                               'ps_Y_max_d{}', 'ps_tY_max_d{}', 'ps_YY_max_d{}']
    if any(template.format(d_idx_to_plot) not in prefix_sums_dict for template in keys_to_check_templates):
        print(f"Warning (plot_all_prefix_sums): Missing some prefix sum keys for dimension {d_idx_to_plot}. Cannot plot all.")
        return

    # Universal time prefix sums
    universal_keys = ['ps_n', 'ps_t', 'ps_tt']
    # Per-dimension prefix sums for the selected dimension
    dim_specific_keys = [template.format(d_idx_to_plot) for template in keys_to_check_templates]
    keys_ordered_for_plot = universal_keys + dim_specific_keys
    
    N_plus_1 = len(prefix_sums_dict['ps_n']); time_axis_ps = np.arange(N_plus_1) 
    plt.style.use('seaborn-v0_8-whitegrid'); plt.rcParams['font.family'] = config.PLOT_FONT
    fig, axs = plt.subplots(3, 3, figsize=(18, 15), sharex=False); axs_flat = axs.flatten()
    fig.suptitle(f'Prefix Sums{plot_title_dim_suffix}', fontsize=16, y=0.99)

    for i, key in enumerate(keys_ordered_for_plot):
        if i >= len(axs_flat): break 
        ax = axs_flat[i]; ps_array = prefix_sums_dict.get(key) # Use .get for safety
        if ps_array is None or not isinstance(ps_array, np.ndarray) or ps_array.size == 0 or len(ps_array) != N_plus_1:
             ax.set_title(key + " (Invalid/Missing Data)"); continue
        ax.plot(time_axis_ps, ps_array, marker='.', linestyle='-', markersize=3); ax.set_title(key)
        ax.set_xlabel('Index of Prefix Sum Array (0 to N)'); ax.set_ylabel('Cumulative Sum')
        ax.grid(True, linestyle=':', which='major', axis='both')
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    results_path = config.RESULTS_DIR_PROOF
    if not os.path.exists(results_path):
        try: os.makedirs(results_path)
        except OSError: plt.show(); plt.close(fig); return
    full_output_path = os.path.join(results_path, output_filename)
    try: plt.savefig(full_output_path); print(f"Prefix sums plot saved to: {full_output_path}")
    except Exception: pass
    plt.close(fig)


def plot_segment_ols_fit(*args, **kwargs): # This was for single segment, 1D.
    # For multi-D, plot_optimal_segmentation_with_fits is more relevant.
    # Can be kept for debugging specific 1D segment fits if adapted.
    if not config.ENABLE_DETAILED_SEGMENT_PLOTS: return 
    # ... (Implementation would need to be adapted or this function deprecated for multi-D focus)
    print("Note: plot_segment_ols_fit is designed for 1D; use plot_optimal_segmentation_with_fits for multi-D PELT results.")
    pass


def plot_optimal_segmentation_with_fits(
    T_min_values_NxD, T_max_values_NxD, # (Potentially downsampled) N x D tube data
    optimal_segment_endpoints,      # List of 0-indexed changepoints [0, cp1, ..., N_processed]
    all_prefix_sums,                # Prefix sums for the T_min/max_values_NxD
    lambda_penalty,
    total_cost_F_N,
    N_prime_effective,              # Actual N used for PELT (after downsampling)
    resolution_factor,
    output_filename="optimal_segmentation_pelt_multidim.png"):
    """
    Visualizes optimal PELT segmentation for selected dimensions from multi-dimensional data.
    Plots T_min/T_max tubes for these dimensions and their OLS fits within each segment.
    """
    if not config.ENABLE_OPTIMAL_PELT_PLOT: return 

    if T_min_values_NxD is None or T_max_values_NxD is None or not optimal_segment_endpoints:
        print("Warning (plot_optimal_segmentation_MD): Invalid input data. Cannot plot.")
        return
    
    N_processed, D_processed = T_min_values_NxD.shape
    if N_processed == 0 or D_processed == 0:
        print("Warning (plot_optimal_segmentation_MD): Processed tube data is empty. Cannot plot.")
        return

    dims_to_plot_names = config.DIMENSIONS_TO_PLOT_IN_PELT_VIS
    # Get indices of these dimensions based on config.DIMENSIONS_TO_USE
    dims_to_plot_indices = []
    valid_dims_to_plot_names = []
    for name in dims_to_plot_names:
        try:
            idx = config.DIMENSIONS_TO_USE.index(name)
            if idx < D_processed: # Ensure index is valid for the processed data
                dims_to_plot_indices.append(idx)
                valid_dims_to_plot_names.append(name)
        except ValueError:
            print(f"Warning: Dimension '{name}' from DIMENSIONS_TO_PLOT_IN_PELT_VIS not found in config.DIMENSIONS_TO_USE or not in processed data.")
    
    if not dims_to_plot_indices:
        print("Error (plot_optimal_segmentation_MD): No valid dimensions selected for plotting. Check config.DIMENSIONS_TO_PLOT_IN_PELT_VIS.")
        return

    num_dims_to_plot = len(dims_to_plot_indices)
    time_axis_processed = np.arange(N_processed) 

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = config.PLOT_FONT
    
    # Create a figure with subplots for each dimension to plot
    # Adjust layout if num_dims_to_plot is large
    n_cols_plot = 1 if num_dims_to_plot <= 2 else 2
    n_rows_plot = (num_dims_to_plot + n_cols_plot - 1) // n_cols_plot
    fig, axs = plt.subplots(n_rows_plot, n_cols_plot, figsize=(8 * n_cols_plot, 5 * n_rows_plot), sharex=True, squeeze=False)
    axs_flat = axs.flatten()

    fig_title = (f'PELT Segmentation (ResFactor={resolution_factor}, N\'={N_prime_effective}, D={D_processed})\n'
                 f'$\lambda$={lambda_penalty:.3f}, Total Cost $F[N\']$={total_cost_F_N:.2f}, Segments={len(optimal_segment_endpoints)-1}')
    fig.suptitle(fig_title, fontsize=14, y=0.99 if n_rows_plot > 1 else 1.02)


    num_segments = len(optimal_segment_endpoints) - 1
    fit_line_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown'] # Cycle through these

    for plot_idx, d_actual_idx in enumerate(dims_to_plot_indices):
        ax = axs_flat[plot_idx]
        dim_name = valid_dims_to_plot_names[plot_idx]

        # Plot the tube for this dimension
        ax.plot(time_axis_processed, T_max_values_NxD[:, d_actual_idx], color='silver', linestyle='-', linewidth=1.0, label=f'T_max ({dim_name})')
        ax.plot(time_axis_processed, T_min_values_NxD[:, d_actual_idx], color='silver', linestyle='-', linewidth=1.0, label=f'T_min ({dim_name})')
        ax.fill_between(time_axis_processed, T_min_values_NxD[:, d_actual_idx], T_max_values_NxD[:, d_actual_idx], 
                        color='whitesmoke', alpha=0.7, zorder=0)

        for k_seg in range(num_segments):
            seg_start_0idx = optimal_segment_endpoints[k_seg]
            seg_end_0idx = optimal_segment_endpoints[k_seg+1] 
            if seg_start_0idx >= seg_end_0idx : continue 

            seg_start_1based_for_Cls = seg_start_0idx + 1
            seg_end_1based_for_Cls = seg_end_0idx 
            
            # Get OLS fit for T_min of this dimension and segment
            a_m, b_m, _ = calculate_ls_fit_for_segment_1D(
                "ps_Y_min", d_actual_idx, 
                seg_start_1based_for_Cls, seg_end_1based_for_Cls,
                all_prefix_sums, 
                T_min_values_NxD[:, d_actual_idx] 
            )
            # Get OLS fit for T_max of this dimension and segment
            a_mx, b_mx, _ = calculate_ls_fit_for_segment_1D(
                "ps_Y_max", d_actual_idx,
                seg_start_1based_for_Cls, seg_end_1based_for_Cls,
                all_prefix_sums,
                T_max_values_NxD[:, d_actual_idx]
            )

            segment_plot_x_axis = np.arange(seg_start_0idx, seg_end_0idx)
            k_values_for_line_fit = np.arange(seg_start_1based_for_Cls, seg_end_1based_for_Cls + 1)

            if k_values_for_line_fit.size > 0 and len(segment_plot_x_axis) == len(k_values_for_line_fit):
                fitted_line_min_seg = a_m * k_values_for_line_fit + b_m
                fitted_line_max_seg = a_mx * k_values_for_line_fit + b_mx
                
                line_color = fit_line_colors[k_seg % len(fit_line_colors)]
                ax.plot(segment_plot_x_axis, fitted_line_min_seg, linestyle='--', linewidth=2.0, color=line_color)
                ax.plot(segment_plot_x_axis, fitted_line_max_seg, linestyle='--', linewidth=2.0, color=line_color)

            if seg_end_0idx < N_processed : 
                ax.axvline(x=seg_end_0idx - 0.5, color='black', linestyle=':', linewidth=1.2, alpha=0.7) 
        
        ax.set_ylabel(f'Value of {dim_name}')
        ax.grid(True, linestyle=':')
        if plot_idx >= num_dims_to_plot - n_cols_plot : # Only add x-label to bottom row plots
             ax.set_xlabel(f'Time Step (0-indexed, Processed N\'={N_processed})')
        
        # Simple legend for each subplot
        handles_subplot = [
            plt.Line2D([0], [0], color='silver', lw=2, label='Tube Boundary'),
            plt.Line2D([0], [0], color=fit_line_colors[0], linestyle='--', lw=2, label='Segment Fit'),
            plt.Line2D([0], [0], color='black', linestyle=':', lw=1.2, label='Changepoint')
        ]
        ax.legend(handles=handles_subplot, loc='best', fontsize='small')


    # Hide any unused subplots
    for i in range(num_dims_to_plot, n_rows_plot * n_cols_plot):
        fig.delaxes(axs_flat[i])

    plt.tight_layout(rect=[0, 0, 1, 0.95 if n_rows_plot > 1 and num_dims_to_plot > 1 else 0.92]) # Adjust for suptitle
    results_path = config.RESULTS_DIR_PROOF
    if not os.path.exists(results_path):
        try: os.makedirs(results_path)
        except OSError: plt.show(); plt.close(fig); return

    full_output_path = os.path.join(results_path, output_filename)
    try:
        plt.savefig(full_output_path, dpi=150)
        print(f"Multi-dim optimal segmentation plot saved to: {full_output_path}")
    except Exception as e:
        print(f"Error saving multi-dim optimal segmentation plot: {e}")
    plt.close(fig) 


def plot_experiment_summary_metrics(results_df: pd.DataFrame, target_dimension_name: str):
    # This function plots summary metrics from experiment runs.
    # It does not directly depend on multi-D data structure, only on the DataFrame columns.
    # So, it should remain largely functional.
    if results_df.empty:
        print("Info (plot_experiment_summary_metrics): Results DataFrame is empty. No summary plots generated.")
        return
    # ... (rest of plot_experiment_summary_metrics - unchanged from v7, ensure plt.close(fig)) ...
    print("\n--- Generating Experiment Summary Plots ---")
    plt.style.use('seaborn-v0_8-whitegrid'); plt.rcParams['font.family'] = config.PLOT_FONT
    results_path = config.RESULTS_DIR_PROOF 
    if not os.path.exists(results_path):
        try: os.makedirs(results_path)
        except OSError as e: print(f"Error creating results directory {results_path}: {e}. Plots will be shown but not saved.")
    def save_and_show_summary(fig, filename_suffix): # Renamed to avoid conflict
        filename = f"summary_MD_{target_dimension_name}_{filename_suffix}.png" # Added MD prefix
        full_output_path = os.path.join(results_path, filename)
        try: fig.savefig(full_output_path); print(f"Summary plot saved to: {full_output_path}")
        except Exception as e: print(f"Error saving summary plot {filename}: {e}")
        plt.close(fig) 
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for res_factor, group in results_df.groupby('Resolution_Factor'):
        group_sorted = group.sort_values(by='N_prime')
        ax1.plot(group_sorted['N_prime'], group_sorted['PELT_Runtime_s'], marker='o', linestyle='-', label=f'ResFactor={res_factor}')
    ax1.set_xlabel("N' (Processed Data Length)"); ax1.set_ylabel("PELT Runtime (seconds)")
    ax1.set_title(f"PELT Runtime vs. N' ({'Multi-Dim' if len(config.DIMENSIONS_TO_USE)>1 else target_dimension_name})")
    ax1.legend(); ax1.grid(True); save_and_show_summary(fig1, "runtime_vs_Nprime")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for res_factor, group in results_df.groupby('Resolution_Factor'):
        group_sorted = group.sort_values(by='N_prime')
        ax2.plot(group_sorted['N_prime'], group_sorted['Max_R_Candidates_Size'], marker='o', linestyle='-', label=f'ResFactor={res_factor}')
    ax2.set_xlabel("N' (Processed Data Length)"); ax2.set_ylabel("Max |R_candidates| Size")
    ax2.set_title(f"Max |R_candidates| vs. N' ({'Multi-Dim' if len(config.DIMENSIONS_TO_USE)>1 else target_dimension_name})")
    ax2.legend(); ax2.grid(True); save_and_show_summary(fig2, "max_R_vs_Nprime")
    resolution_factors_present = sorted(results_df['Resolution_Factor'].unique())
    for res_factor in resolution_factors_present:
        df_subset_res = results_df[results_df['Resolution_Factor'] == res_factor].sort_values(by='Lambda')
        if df_subset_res.empty: continue
        avg_N_for_title = df_subset_res['N_prime'].median() 
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(df_subset_res['Lambda'], df_subset_res['Num_Segments'], marker='o', linestyle='-')
        ax3.set_xlabel("Lambda (λ) Penalty"); ax3.set_ylabel("Number of Segments Found")
        ax3.set_title(f"Num Segments vs. Lambda (ResF={res_factor}, N'~{avg_N_for_title:.0f}) ({'MD' if len(config.DIMENSIONS_TO_USE)>1 else target_dimension_name})")
        ax3.grid(True); ax3.set_xscale('log'); save_and_show_summary(fig3, f"numsegments_vs_lambda_res{res_factor}")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(df_subset_res['Lambda'], df_subset_res['PELT_Runtime_s'], marker='o', linestyle='-')
        ax4.set_xlabel("Lambda (λ) Penalty"); ax4.set_ylabel("PELT Runtime (seconds)")
        ax4.set_title(f"Runtime vs. Lambda (ResF={res_factor}, N'~{avg_N_for_title:.0f}) ({'MD' if len(config.DIMENSIONS_TO_USE)>1 else target_dimension_name})")
        ax4.grid(True); ax4.set_xscale('log'); save_and_show_summary(fig4, f"runtime_vs_lambda_res{res_factor}")
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        ax5.plot(df_subset_res['Lambda'], df_subset_res['Max_R_Candidates_Size'], marker='o', linestyle='-')
        ax5.set_xlabel("Lambda (λ) Penalty"); ax5.set_ylabel("Max |R_candidates| Size")
        ax5.set_title(f"Max |R| vs. Lambda (ResF={res_factor}, N'~{avg_N_for_title:.0f}) ({'MD' if len(config.DIMENSIONS_TO_USE)>1 else target_dimension_name})")
        ax5.grid(True); ax5.set_xscale('log'); save_and_show_summary(fig5, f"max_R_vs_lambda_res{res_factor}")
    print("--- Experiment summary plots generation complete. ---")


if __name__ == '__main__':
    print("Testing visualization_proof.py (Multi-Dimensional Update)...")
    # Minimal test for plot_optimal_segmentation_with_fits (multi-D)
    N_test, D_test = 30, 2
    config.DIMENSIONS_TO_USE = ['tx_test', 'ty_test'] # Mock config for test
    config.DIMENSIONS_TO_PLOT_IN_PELT_VIS = ['tx_test', 'ty_test']
    
    min_tube_md = np.random.rand(N_test, D_test)
    max_tube_md = min_tube_md + 0.5 + np.random.rand(N_test, D_test)*0.2
    
    # Mock prefix sums (enough structure for the plotting function to call C_LS)
    mock_ps_md = {'N_timesteps': N_test, 'D_dimensions': D_test, 'ps_t': np.cumsum(np.arange(N_test+1))}
    for d in range(D_test):
        mock_ps_md[f'ps_Y_min_d{d}'] = np.cumsum(np.concatenate(([0],min_tube_md[:,d])))
        mock_ps_md[f'ps_tY_min_d{d}'] = np.cumsum(np.concatenate(([0],np.arange(1,N_test+1)*min_tube_md[:,d])))
        mock_ps_md[f'ps_YY_min_d{d}'] = np.cumsum(np.concatenate(([0],min_tube_md[:,d]**2)))
        mock_ps_md[f'ps_Y_max_d{d}'] = np.cumsum(np.concatenate(([0],max_tube_md[:,d])))
        mock_ps_md[f'ps_tY_max_d{d}'] = np.cumsum(np.concatenate(([0],np.arange(1,N_test+1)*max_tube_md[:,d])))
        mock_ps_md[f'ps_YY_max_d{d}'] = np.cumsum(np.concatenate(([0],max_tube_md[:,d]**2)))
        
    optimal_cps_md = [0, N_test // 2, N_test]
    lambda_p_md = 0.1
    total_cost_md = 5.0 # Dummy value

    plot_optimal_segmentation_with_fits(
        min_tube_md, max_tube_md,
        optimal_cps_md,
        mock_ps_md, # Pass the multi-dim prefix sums
        lambda_p_md, total_cost_md,
        N_prime_effective=N_test, resolution_factor=1,
        output_filename="test_optimal_seg_multidim.png"
    )
    print("Multi-dimensional optimal segmentation plot test completed (check output file).")
