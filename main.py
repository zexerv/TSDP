#!/usr/bin/env python3
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# Import functions from our modules
try:
    import config
    # Import the main analysis function
    from segment_analyzer import analyze_segments_cross_section
    # Import plotting functions (Corrected imports)
    from visualization import plot_cost_vs_segments, plot_segmentation_with_trapezoids
    from visualization import plot_gmm_mean_cov_simple, print_cross_section_stats # Import new summary functions

    # Removed incorrect/unused imports:
    # from segment_analyzer import get_state_at_time # Removed
    # from visualization import plot_gmm_cross_section_interactive # Removed

except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary project modules.")
    print(f"Ensure all .py files are in the same directory.")
    print(f"Error details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during project imports: {e}")
    sys.exit(1)


if __name__ == "__main__":
    overall_start_time = time.time()
    print("--- Starting Trajectory Analysis Pipeline ---")

    # --- Run the Core Analysis ---
    print("\n--- Running Core Analysis (Loading, Alignment, Segmentation, GMM K=1 Analysis) ---")
    analysis_results = None
    try:
        # analyze_segments_cross_section handles steps 1-11 internally now
        # It returns the dictionary needed for plotting
        analysis_results = analyze_segments_cross_section()

        # Check if analysis succeeded
        if analysis_results is None:
            print("Analysis function returned None. Exiting.")
            sys.exit(1)

        # Unpack results
        processed_aligned_trajs = analysis_results.get("processed_aligned_trajs")
        min_vals_plot_tube = analysis_results.get("min_vals_plot_tube")
        max_vals_plot_tube = analysis_results.get("max_vals_plot_tube")
        optimal_segment_indices_orig = analysis_results.get("optimal_segment_indices_orig")
        valid_mapped_events_list = analysis_results.get("valid_mapped_events_list")
        raw_costs_per_segment_count = analysis_results.get("raw_costs_per_segment_count")
        optimal_num_segments = analysis_results.get("optimal_num_segments")

        # Check if essential plotting data is present
        # Note: min/max_vals_plot_tube might be None if plotting fails, handle gracefully
        if not all([processed_aligned_trajs is not None,
                    optimal_segment_indices_orig is not None,
                    valid_mapped_events_list is not None,
                    raw_costs_per_segment_count is not None,
                    optimal_num_segments is not None and optimal_num_segments != -1]):
            print("Error: Analysis function did not return all necessary data for plotting. Exiting.")
            sys.exit(1)
        if min_vals_plot_tube is None or max_vals_plot_tube is None:
             print("Warning: Tube data for plotting was not generated successfully.")
             # Allow proceeding, but main plot might be incomplete

        print("--- Core Analysis Finished Successfully ---")

    except Exception as e:
        print(f"\n--- FATAL ERROR during analysis pipeline ---")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


    # --- Plot Results ---
    print("\n--- Generating Plots ---")
    try:
        # Plot 1: Cost vs. Number of Segments
        print("  Generating Cost vs. Segments plot...")
        plot_cost_vs_segments(raw_costs_per_segment_count, config.MAX_SEGMENTS,
                              optimal_num_segments, config.LAMBDA_PENALTY)

        # Plot 2: Main Segmentation Plot (Tube Plot)
        # Ensure tube data exists before plotting
        if min_vals_plot_tube is not None and max_vals_plot_tube is not None:
            print("  Generating Main Segmentation plot (Tube)...")
            plot_title = (f'Optimal Segmentation ({optimal_num_segments} Segs, λ={config.LAMBDA_PENALTY:.3f}, '
                          f'RotW={config.ROTATION_WEIGHT:.3f}, DP MaxLen={config.MAX_DP_LENGTH})\n'
                          f'Plot Features: {config.PLOT_FEATURE_COLS}') # Indicate plotted features

            plot_segmentation_with_trapezoids(
                min_vals_orig=min_vals_plot_tube, # Use tube data for plot features
                max_vals_orig=max_vals_plot_tube, # Use tube data for plot features
                segment_indices_orig=optimal_segment_indices_orig,
                processed_aligned_trajs=processed_aligned_trajs,
                all_mapped_events_list=valid_mapped_events_list,
                feature_names=config.PLOT_FEATURE_COLS, # Use features defined for plotting
                title=plot_title
            )
        else:
            print("  Skipping Main Segmentation plot (Tube data missing).")

        # Plot 3: Simple Mean/Covariance Summary Plot (Optional)
        print("\n--- Generating Cross-Section GMM (K=1) Summary Plot ---")
        # Define pairs you want to see ellipses for.
        # This should include features present in config.ANALYSIS_ORIENTATION_REPRESENTATION
        # Example: If using 'rotation_matrix'
        ellipse_pairs = [('tx','ty'), ('r11','r12'), ('r11','r21'), ('r22','r33')]
        # Example: If using 'log_map'
        # ellipse_pairs = [('tx','ty'), ('vx_log','vy_log')]
        # Example: If using 'quaternion'
        # ellipse_pairs = [('tx','ty'), ('qw','qx'), ('qy','qz')]

        plot_gmm_mean_cov_simple(
            results_path=config.CROSS_SECTION_STATS_OUTPUT_PATH,
            pairs_to_plot=ellipse_pairs # Pass the desired pairs
        )

        # Option: Print summary stats instead of/in addition to plot
        # print("\n--- Printing Cross-Section Stats Summary ---")
        # print_cross_section_stats(config.CROSS_SECTION_STATS_OUTPUT_PATH)


        print("--- Plotting Finished ---")

    except Exception as e:
        print(f"\n--- ERROR during plotting ---")
        print(f"Error: {e}")
        traceback.print_exc()
        # Continue to show script duration even if plotting fails

    overall_duration = time.time() - overall_start_time
    print(f"\n--- Script Finished in {overall_duration:.2f} seconds ---")
