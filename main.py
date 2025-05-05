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
    from segment_analyzer import get_state_at_time # Need this helper

    # Import plotting functions
    from visualization import plot_gmm_cross_section_interactive # Import the new function

    from visualization import plot_cost_vs_segments, plot_segmentation_with_trapezoids
    # Import necessary components if analyze_segments_cross_section doesn't handle everything internally
    # (Assuming analyze_segments_cross_section now handles steps 1-12 internally)
    # from data_loading import load_selected_data, find_events
    # from alignment import align_trajectories
    # from segmentation import (calculate_tube, downsample_tube_data,
    #                           find_optimal_segmentation, map_boundaries_to_original)

except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary project modules.")
    print(f"Ensure all .py files are in the same directory.")
    print(f"Error details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during project imports: {e}")
    sys.exit(1)


# Optional: Increase recursion depth limit if DP requires it (usually not needed with downsampling)
# sys.setrecursionlimit(config.MAX_DP_LENGTH * config.MAX_SEGMENTS + 100)


if __name__ == "__main__":
    overall_start_time = time.time()
    print("--- Starting Trajectory Analysis Pipeline ---")

    # --- Run the Core Analysis ---
    # The analyze_segments_cross_section function now handles:
    # 1. Loading Data
    # 2. Finding Events
    # 3. Aligning Trajectories & Mapping Events
    # 4. Calculating Tube (for plotting features)
    # 5. Downsampling (for DP features)
    # 6. Finding Optimal Segmentation (DP)
    # 7. Mapping Boundaries
    # 8. Preprocessing Data (adding quat/logmap)
    # 9. Identifying Cross-Sections
    # 10. Analyzing Cross-Sections (GMM, Stats)
    # 11. Saving Cross-Section Results
    # 12. Backup Global GMM (Optional)
    # It needs to return the necessary data for plotting.

    print("\n--- Running Core Analysis (Loading, Alignment, Segmentation, GMM Analysis) ---")
    analysis_results = None
    try:
        # We assume analyze_segments_cross_section is modified to return needed plot data
        # Modify analyze_segments_cross_section to return a dictionary or tuple containing:
        # - processed_aligned_trajs
        # - min_vals_plot_tube
        # - max_vals_plot_tube
        # - optimal_segment_indices_orig
        # - valid_mapped_events_list
        # - raw_costs_per_segment_count (for cost plot)
        # - optimal_num_segments (for cost plot)

        # Placeholder call - adjust based on actual return signature of analyze_segments_cross_section
        analysis_results = analyze_segments_cross_section()

        # --- Check if analysis succeeded ---
        # Add checks here based on what analyze_segments_cross_section returns
        # For example, if it returns None on failure:
        if analysis_results is None:
             print("Analysis function returned None. Exiting.")
             sys.exit(1)

        # --- Unpack results (assuming a dictionary return for clarity) ---
        processed_aligned_trajs = analysis_results.get("processed_aligned_trajs")
        min_vals_plot_tube = analysis_results.get("min_vals_plot_tube")
        max_vals_plot_tube = analysis_results.get("max_vals_plot_tube")
        optimal_segment_indices_orig = analysis_results.get("optimal_segment_indices_orig")
        valid_mapped_events_list = analysis_results.get("valid_mapped_events_list")
        raw_costs_per_segment_count = analysis_results.get("raw_costs_per_segment_count")
        optimal_num_segments = analysis_results.get("optimal_num_segments")

        # Check if essential plotting data is present
        if not all([processed_aligned_trajs is not None,
                   min_vals_plot_tube is not None,
                   max_vals_plot_tube is not None,
                   optimal_segment_indices_orig is not None,
                   valid_mapped_events_list is not None,
                   raw_costs_per_segment_count is not None,
                   optimal_num_segments is not None]):
            print("Error: Analysis function did not return all necessary data for plotting. Exiting.")
            sys.exit(1)

        print("--- Core Analysis Finished Successfully ---")

    except Exception as e:
        print(f"\n--- FATAL ERROR during analysis pipeline ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- Plot Results ---
    print("\n--- Generating Plots ---")
    try:
        # Plot 1: Cost vs. Number of Segments
        print("  Generating Cost vs. Segments plot...")
        plot_cost_vs_segments(raw_costs_per_segment_count, config.MAX_SEGMENTS,
                              optimal_num_segments, config.LAMBDA_PENALTY)

        # Plot 2: Segmentation Plot
        print("  Generating Main Segmentation plot...")
        plot_title = (f'Optimal Segmentation ({optimal_num_segments} Segs, λ={config.LAMBDA_PENALTY:.3f}, '
                      f'RotW={config.ROTATION_WEIGHT:.3f}, DP MaxLen={config.MAX_DP_LENGTH})')

        plot_segmentation_with_trapezoids(
            min_vals_orig=min_vals_plot_tube, # Use tube data for plot features
            max_vals_orig=max_vals_plot_tube, # Use tube data for plot features
            segment_indices_orig=optimal_segment_indices_orig,
            processed_aligned_trajs=processed_aligned_trajs, # Pass the preprocessed list
            all_mapped_events_list=valid_mapped_events_list,
            feature_names=config.PLOT_FEATURE_COLS, # Use features defined for plotting
            cross_section_results_path=config.CROSS_SECTION_STATS_OUTPUT_PATH, # Path to load results for deviation plot
            title=plot_title
        )
        print("--- Plotting Finished ---")



        # --- Plot Interactive GMM (Example: First Boundary Cross-Section) ---
        print("\n  Generating Interactive GMM plot (if data available)...")
        cross_section_data = None
        cs_path = Path(config.CROSS_SECTION_STATS_OUTPUT_PATH)
        if cs_path.is_file():
            try:
                with open(cs_path, 'r') as f:
                    cross_section_data = yaml.safe_load(f)
                if not isinstance(cross_section_data, list): cross_section_data = None
            except Exception as e:
                print(f"    Warning: Could not load cross-section data for GMM plot: {e}")
                cross_section_data = None

        if cross_section_data and processed_aligned_trajs:
            # Find the first boundary cross-section with a valid GMM
            cs_to_plot = None
            target_cs_time = None
            for cs in cross_section_data:
                if cs.get('type') == 'boundary' and cs.get('segment_gmm_params') is not None:
                    cs_to_plot = cs
                    target_cs_time = cs.get('time_index')
                    break # Found one

            if cs_to_plot and target_cs_time is not None:
                print(f"    Plotting GMM for cross-section at time: {target_cs_time:.2f}")
                # Recalculate snapshots for this cross-section
                snapshot_states_recalc = []
                gmm_state_cols = config.GMM_STATE_COLS
                for traj_df in processed_aligned_trajs:
                    # Ensure target_cs_time is float for get_state_at_time
                    state_vec, _ = get_state_at_time(traj_df, float(target_cs_time), gmm_state_cols)
                    if state_vec is not None:
                        snapshot_states_recalc.append(state_vec)

                if snapshot_states_recalc:
                    snapshot_data_np = np.array(snapshot_states_recalc)
                    plot_gmm_cross_section_interactive(
                        gmm_params=cs_to_plot['segment_gmm_params'],
                        snapshot_data=snapshot_data_np,
                        feature_names=config.GMM_STATE_COLS, # Use the GMM state feature names
                        title=f"GMM at Boundary t={target_cs_time:.2f}"
                    )
                else:
                    print(f"    Warning: Could not recalculate snapshots for time {target_cs_time:.2f}. Skipping interactive GMM plot.")
            else:
                print("    Info: No boundary cross-section with valid GMM found to plot.")
        elif not cross_section_data:
             print("    Info: Cross-section data file not found or invalid. Skipping interactive GMM plot.")
        else:
             print("    Info: No processed trajectories available. Skipping interactive GMM plot.")

    except Exception as e:
        print(f"\n--- ERROR during plotting ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        # Continue to show script duration even if plotting fails

    overall_duration = time.time() - overall_start_time
    print(f"\n--- Script Finished in {overall_duration:.2f} seconds ---")

