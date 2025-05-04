#!/usr/bin/env python3
import time
import sys
import numpy as np
import pandas as pd

# Import functions from our modules
import config
from data_loading import load_selected_data, find_events
from alignment import align_trajectories
from segmentation import (calculate_tube, downsample_tube_data,
                          find_optimal_segmentation, map_boundaries_to_original)
from visualization import plot_cost_vs_segments, plot_segmentation_with_trapezoids

# Optional: Increase recursion depth limit if DP requires it (usually not needed with downsampling)
# sys.setrecursionlimit(config.MAX_DP_LENGTH * config.MAX_SEGMENTS + 100)


if __name__ == "__main__":
    overall_start_time = time.time()

    # 1. Load data
    print("--- 1. Loading Data ---")
    # Use the comprehensive ALL_LOAD_COLS list from the updated config
    original_trajs_df, loaded_files = load_selected_data(
        config.PARENT_FOLDER_PATH,
        load_all=config.LOAD_ALL_FILES,
        file_list=config.FILE_LIST,
        columns_to_load=config.ALL_LOAD_COLS # Load all necessary columns
    )
    if not original_trajs_df:
        sys.exit("Exiting: No trajectories loaded.")
    print(f"Loaded {len(original_trajs_df)} trajectories: {loaded_files}")

    # 2. Find Events in Original Trajectories
    print("\n--- 2. Finding Events in Original Trajectories ---")
    original_events_list = []
    for i, df in enumerate(original_trajs_df):
         print(f"  Finding events for trajectory {i} ({loaded_files[i]})...") # Add print here
         if df.empty:
             print(f"    Warning: DataFrame for trajectory {i} ({loaded_files[i]}) is empty. Appending empty events dict.")
             events = {'start': [], 'end': [], 'state_change': [], 'wp_saved': [], 'gripper_change': []} # Default empty dict
         else:
             # Pass only the basic event config; find_events uses interface_id for state_change
             events = find_events(df, config.EVENT_COLUMNS) # Call find_events
         original_events_list.append(events) # Appends the result (should be a dict)

    # --- Verification Check (Optional but Recommended) ---
    print("\n--- Verifying Events List Contents ---")
    list_ok = True
    if len(original_events_list) != len(loaded_files):
        print(f"FATAL ERROR: Mismatch in length between loaded files ({len(loaded_files)}) and events list ({len(original_events_list)})")
        list_ok = False
    else:
        for i, evt_dict in enumerate(original_events_list):
            print(f"  Checking events_list index {i} (File: {loaded_files[i]}):") # Added file context
            if evt_dict is None:
                print(f"    FATAL ERROR: Event dictionary is None!")
                list_ok = False
                break # Exit loop early if None found
            elif isinstance(evt_dict, dict):
                 print(f"    Type: {type(evt_dict)}, Keys: {list(evt_dict.keys())}")
                 # Specifically check the content for state_change
                 print(f"    State Change Indices Found: {evt_dict.get('state_change', 'N/A')}")
            else:
                print(f"    FATAL ERROR: Content is not None or dict, it is: {type(evt_dict)}")
                list_ok = False
                break # Exit loop early

    if not list_ok:
        sys.exit("Exiting due to problems found in events list (None or wrong type found).")
    print("--- Events List Verification Passed ---")
    # --- END Verification Check ---

    # 3. Perform Alignment and Map Events
    print("\n--- 3. Aligning Trajectories & Mapping Events ---")
    align_start_time = time.time()
    # align_trajectories now returns aligned DFs AND the list of mapped events
    aligned_trajs_df, mapped_events_list, ref_idx = align_trajectories(
        original_trajs_df,
        original_events_list, # Pass the verified list
        config.ALIGNMENT_TYPE,
        config.ALIGNMENT_BASE_FEATURE_COLS,
        config.ALL_LOAD_COLS, # Ensure aligned DFs include all needed cols for tube calc etc.
        config.NORMALIZE_FOR_COMBINED
    )
    align_duration = time.time() - align_start_time
    print(f"Alignment and event mapping complete in {align_duration:.2f} seconds.")
    if ref_idx == -1 or not aligned_trajs_df:
        sys.exit("Exiting: Alignment failed.")

    # Filter out potentially empty DFs post-alignment (shouldn't happen ideally)
    valid_aligned_trajs_df = [df for df in aligned_trajs_df if not df.empty]
    if not valid_aligned_trajs_df:
        sys.exit("Exiting: Alignment produced no valid (non-empty) trajectories.")
    # Also filter mapped_events_list to correspond to valid_aligned_trajs_df
    valid_indices = [i for i, df in enumerate(aligned_trajs_df) if not df.empty]
    valid_mapped_events_list = [mapped_events_list[i] for i in valid_indices]

    if len(valid_aligned_trajs_df) != len(valid_mapped_events_list):
         # This case should be extremely unlikely if alignment worked correctly
         print("Warning: Mismatch between valid aligned trajectories and mapped events after filtering.")
         # Decide how to handle: exit or proceed with potentially mismatched data? Exit is safer.
         sys.exit("Exiting due to event/trajectory list mismatch after filtering.")

    print(f"Produced {len(valid_aligned_trajs_df)} valid aligned trajectories.")


    # 4. Calculate Tube using Segmentation Features
    print("\n--- 4. Calculating Tube ---")
    # Use only the valid aligned trajectories
    min_vals_orig, max_vals_orig, n_timesteps_orig = calculate_tube(
        valid_aligned_trajs_df, config.SEGMENTATION_FEATURE_COLS
    )
    if min_vals_orig is None or n_timesteps_orig <= 1:
        sys.exit("Exiting: Failed to calculate tube or not enough time steps.")

    # 5. Downsample Tube Data for DP if necessary
    print("\n--- 5. Downsampling Data for DP (if needed) ---")
    min_vals_dp, max_vals_dp, n_timesteps_dp, stride = downsample_tube_data(
        min_vals_orig, max_vals_orig, config.MAX_DP_LENGTH
    )

    # 6. Find Optimal Segmentation using DP
    print("\n--- 6. Finding Optimal Segmentation (DP) ---")
    segmentation_start_time = time.time()
    segment_indices_dp, optimal_num_segments, min_total_cost, raw_costs_per_segment_count = find_optimal_segmentation(
        min_vals_dp, max_vals_dp, n_timesteps_dp,
        config.MAX_SEGMENTS,
        config.LAMBDA_PENALTY,
        config.SEGMENTATION_FEATURE_COLS, # Feature names list
        config.POS_COLS, # Pass the set of position columns from config
        config.ROTATION_WEIGHT,
        valid_mapped_events_list, # Pass the mapped events
        stride # Pass the downsampling stride
    )
    segmentation_duration = time.time() - segmentation_start_time
    print(f"Segmentation DP finished in {segmentation_duration:.2f} seconds.")
    if optimal_num_segments == -1 or segment_indices_dp is None:
        sys.exit("Exiting: Segmentation optimization failed.")

    # 7. Map DP boundaries back to original time scale
    print("\n--- 7. Mapping Segment Boundaries ---")
    optimal_segment_indices_orig = map_boundaries_to_original(
        segment_indices_dp, stride, n_timesteps_orig
    )

    print(f"\n--- Optimal Segmentation Found ---")
    print(f"  Number of Segments: {optimal_num_segments}")
    print(f"  Segment Boundaries (Original Time Scale): {optimal_segment_indices_orig}")
    print(f"  Min Total Cost (Raw + Penalty, based on DP data): {min_total_cost:.4f}")

    # 8. Plot Results
    print("\n--- 8. Plotting Results ---")
    # Plot 1: Cost vs. Number of Segments
    plot_cost_vs_segments(raw_costs_per_segment_count, config.MAX_SEGMENTS,
                           optimal_num_segments, config.LAMBDA_PENALTY)

    # Plot 2: Optimal Segmentation with Trapezoids, Individual Trajectories and All Events
    # Use LaTeX for lambda and include other relevant params
    plot_title = (f'Optimal Segmentation ({optimal_num_segments} Segs, $\lambda$={config.LAMBDA_PENALTY:.3f}, RotW={config.ROTATION_WEIGHT:.3f}, DP MaxLen={config.MAX_DP_LENGTH})')

    # Pass the filtered lists to the plotting function
    plot_segmentation_with_trapezoids(
        min_vals_orig,
        max_vals_orig,
        optimal_segment_indices_orig,
        valid_aligned_trajs_df,      # Pass the list of valid aligned trajectory DFs
        valid_mapped_events_list,    # Pass the corresponding list of mapped event dictionaries
        config.SEGMENTATION_FEATURE_COLS,
        title=plot_title
    )

    overall_duration = time.time() - overall_start_time
    print(f"\n--- Script Finished in {overall_duration:.2f} seconds ---")
