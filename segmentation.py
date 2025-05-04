import numpy as np
import math
import time
import config # Import configuration
# Use the renamed function from alignment.py v4
from alignment import find_pre_post_waypoints, find_first_gripper_changes

def calculate_tube(aligned_trajectories_df_list, feature_cols):
    """Calculates the min/max envelope (tube) for specified features. (Unchanged)"""
    if not aligned_trajectories_df_list:
        print("Warning: No aligned trajectories to calculate tube from.")
        return None, None, 0

    target_len = 0
    first_valid_df = None
    for df in aligned_trajectories_df_list:
        if not df.empty:
            target_len = len(df)
            first_valid_df = df
            break
    if target_len == 0 or first_valid_df is None:
        print("Warning: Could not determine target length from aligned trajectories.")
        return None, None, 0

    cols_present_in_first = [col for col in feature_cols if col in first_valid_df.columns]
    if len(cols_present_in_first) != len(feature_cols):
         print(f"Warning: Not all requested tube features ({feature_cols}) exist in the aligned data. Using: {cols_present_in_first}")
         feature_cols = cols_present_in_first
         if not feature_cols:
             print("Error: No specified tube features exist in the aligned data.")
             return None, None, 0

    stacked_data = []
    for i, df in enumerate(aligned_trajectories_df_list):
        if not df.empty:
            if not all(c in df.columns for c in feature_cols):
                print(f"Warning: Skipping trajectory {i} as it misses some tube feature columns.")
                continue
            if len(df) != target_len:
                print(f"Warning: Skipping trajectory {i} with inconsistent length {len(df)} (expected {target_len}).")
                continue
            stacked_data.append(df[feature_cols].values)

    if not stacked_data:
        print(f"Warning: No valid data found for features {feature_cols} in aligned trajectories.")
        return None, None, 0

    try:
        trajectory_array = np.stack(stacked_data, axis=0)
    except ValueError as e:
        print(f"Error stacking trajectories: {e}. Check alignment output consistency.")
        return None, None, 0

    min_vals = np.min(trajectory_array, axis=0)
    max_vals = np.max(trajectory_array, axis=0)
    n_timesteps = min_vals.shape[0]

    print(f"Calculated tube: {n_timesteps} time steps, {min_vals.shape[1]} dimensions ({feature_cols}).")
    return min_vals, max_vals, n_timesteps


def downsample_tube_data(min_vals, max_vals, max_length):
    """Downsamples tube data if its length exceeds max_length using stride. (Unchanged)"""
    n_timesteps_orig = min_vals.shape[0]
    if n_timesteps_orig <= max_length:
        print("Data length within limit, no downsampling needed for DP.")
        return min_vals, max_vals, n_timesteps_orig, 1 # Stride is 1

    stride = math.ceil(n_timesteps_orig / max_length)
    print(f"Downsampling data for DP: Original length={n_timesteps_orig}, Target max={max_length}, Stride={stride}")

    min_vals_ds = min_vals[::stride]
    max_vals_ds = max_vals[::stride]
    n_timesteps_ds = min_vals_ds.shape[0]
    print(f"Downsampled DP length: {n_timesteps_ds}")

    return min_vals_ds, max_vals_ds, n_timesteps_ds, stride

def calculate_segment_cost(start_idx, end_idx, min_vals_dim, max_vals_dim, weight=1.0):
    """
    Calculates the geometric cost of approximating a tube segment (for a single dimension)
    with linear boundaries (trapezoid). (Unchanged)
    """
    if start_idx >= end_idx: return 0.0
    segment_len_points = end_idx - start_idx + 1
    if segment_len_points <= 1: return 0.0

    actual_min = min_vals_dim[start_idx : end_idx + 1]
    actual_max = max_vals_dim[start_idx : end_idx + 1]
    t = np.arange(segment_len_points)

    min_start_val = actual_min[0]; min_end_val = actual_min[-1]
    min_slope = (min_end_val - min_start_val) / (segment_len_points - 1) if segment_len_points > 1 else 0
    approx_min = min_start_val + min_slope * t

    max_start_val = actual_max[0]; max_end_val = actual_max[-1]
    max_slope = (max_end_val - max_start_val) / (segment_len_points - 1) if segment_len_points > 1 else 0
    approx_max = max_start_val + max_slope * t

    error_min = np.sum((actual_min - approx_min)**2)
    error_max = np.sum((actual_max - approx_max)**2)

    return weight * (error_min + error_max)


def calculate_representative_event_indices(mapped_events_list, stride):
    """
    Calculates representative indices (median) for key events across all trajectories,
    mapped to the downsampled time scale used by the DP. (Unchanged)
    """
    print("Calculating representative event indices for segmentation penalty...")
    key_event_indices = {
        'sc': [], 'pre_wp1': [], 'post_wp1': [], 'pre_gc1': [], 'post_gc1': []
    }
    num_trajectories = len(mapped_events_list)
    if num_trajectories == 0:
        print("  No mapped events provided.")
        return {key: None for key in key_event_indices}

    for events in mapped_events_list:
        sc_indices = events.get('state_change', [])
        wp_indices = sorted(events.get('wp_saved', []))
        gc_indices = sorted(events.get('gripper_change', []))

        first_sc_idx = min(sc_indices) if sc_indices else None
        if first_sc_idx is not None:
            key_event_indices['sc'].append(first_sc_idx)
            pre_wps, post_wps = find_pre_post_waypoints(first_sc_idx, wp_indices, n=1)
            pre_wp1 = pre_wps[0]; post_wp1 = post_wps[0]
            if pre_wp1 is not None: key_event_indices['pre_wp1'].append(pre_wp1)
            if post_wp1 is not None: key_event_indices['post_wp1'].append(post_wp1)
            pre_gc1, post_gc1 = find_first_gripper_changes(first_sc_idx, gc_indices)
            if pre_gc1 is not None: key_event_indices['pre_gc1'].append(pre_gc1)
            if post_gc1 is not None: key_event_indices['post_gc1'].append(post_gc1)

    representative_indices = {}
    for key, indices in key_event_indices.items():
        if indices:
            median_idx_orig = int(np.median(indices))
            median_idx_dp = int(round(median_idx_orig / stride))
            representative_indices[key] = median_idx_dp
            print(f"  Representative index for '{key}': {median_idx_orig} (Original) -> {median_idx_dp} (DP Scale)")
        else:
            representative_indices[key] = None
            print(f"  Representative index for '{key}': Not found in any trajectory.")

    return representative_indices


# --- MODIFIED: find_optimal_segmentation Function ---
def find_optimal_segmentation(min_vals_dp, max_vals_dp, n_timesteps_dp, max_segments,
                              lambda_penalty, segmentation_features, pos_cols, rotation_weight,
                              mapped_events_list, # NEW: Pass mapped events
                              stride # NEW: Pass stride for mapping event indices
                              ):
    """
    Finds the optimal segmentation using dynamic programming, including an
    optional penalty for segments containing pairs of key events that should be separated.
    """
    n_dims = min_vals_dp.shape[1]
    if n_dims == 0 or n_timesteps_dp <= 1:
        print("Error: Cannot perform segmentation with no dimensions or <= 1 time step.")
        return None, -1, np.inf, np.full(max_segments, np.inf)

    dp_start_time = time.time()

    # --- Calculate Representative Event Indices (mapped to DP scale) ---
    # Penalties are only active if > 0 in config
    sc_neighbor_penalty = config.SEGMENTATION_EVENT_COOCCURRENCE_PENALTY
    neighbor_neighbor_penalty = config.SEGMENTATION_NEIGHBOR_COOCCURRENCE_PENALTY if config.PENALIZE_NEIGHBOR_COOCCURRENCE else 0.0

    rep_indices_dp = {}
    if sc_neighbor_penalty > 0 or neighbor_neighbor_penalty > 0:
        rep_indices_dp = calculate_representative_event_indices(mapped_events_list, stride)
    else:
        print("Skipping event co-occurrence penalty calculation (penalties are 0).")

    # --- Precompute Geometric Segment Costs (on DP data) ---
    print("Precomputing geometric segment costs (on DP data)...")
    cost_cache = np.full((n_timesteps_dp, n_timesteps_dp), np.inf)
    precompute_start_time = time.time()

    for i in range(n_timesteps_dp):
        cost_cache[i, i] = 0.0
        for j in range(i + 1, n_timesteps_dp):
            segment_cost_total = 0.0
            for d in range(n_dims):
                feature_name = segmentation_features[d]
                current_weight = 1.0 if feature_name in pos_cols else rotation_weight
                cost_d = calculate_segment_cost(i, j, min_vals_dp[:, d], max_vals_dp[:, d], weight=current_weight)
                if not np.isfinite(cost_d):
                    segment_cost_total = np.inf; break
                segment_cost_total += cost_d
            cost_cache[i, j] = segment_cost_total

    print(f"Geometric segment costs precomputed in {time.time() - precompute_start_time:.2f} seconds.")

    # --- DP Calculation ---
    dp = np.full((n_timesteps_dp, max_segments + 1), np.inf)
    bp = np.full((n_timesteps_dp, max_segments + 1), -1, dtype=int)

    # --- REFINED PENALTY LOGIC ---
    # Helper function to check co-occurrence within the DP loop
    def check_cooccurrence(seg_start, seg_end, rep_indices):
        total_penalty = 0.0

        sc_idx = rep_indices.get('sc')
        pre_neighbor_idx = rep_indices.get('pre_wp1') or rep_indices.get('pre_gc1')
        post_neighbor_idx = rep_indices.get('post_wp1') or rep_indices.get('post_gc1')

        # Check SC vs Neighbors (if penalty > 0)
        if sc_neighbor_penalty > 0 and sc_idx is not None:
            sc_in_segment = (seg_start <= sc_idx <= seg_end)
            pre_in_segment = (pre_neighbor_idx is not None and seg_start <= pre_neighbor_idx <= seg_end)
            post_in_segment = (post_neighbor_idx is not None and seg_start <= post_neighbor_idx <= seg_end)

            if sc_in_segment and (pre_in_segment or post_in_segment):
                total_penalty += sc_neighbor_penalty
                # print(f"  Debug: SC-Neighbor penalty added for segment [{seg_start}, {seg_end}]")

        # Check Pre-Neighbor vs Post-Neighbor (if penalty > 0 and flag is True)
        if neighbor_neighbor_penalty > 0: # Implicitly checks config.PENALIZE_NEIGHBOR_COOCCURRENCE
             if pre_neighbor_idx is not None and post_neighbor_idx is not None:
                 pre_in_segment = (seg_start <= pre_neighbor_idx <= seg_end)
                 post_in_segment = (seg_start <= post_neighbor_idx <= seg_end)

                 if pre_in_segment and post_in_segment:
                     # Add penalty only if SC penalty wasn't already added for this segment
                     # to avoid double-penalizing in some cases (optional rule).
                     # Or simply add it regardless:
                     total_penalty += neighbor_neighbor_penalty
                     # print(f"  Debug: Neighbor-Neighbor penalty added for segment [{seg_start}, {seg_end}]")

        return total_penalty
    # --- END REFINED PENALTY LOGIC ---


    # Initialization: Cost for 1 segment ending at time t (segment is [0, t])
    for t in range(n_timesteps_dp):
        geometric_cost = cost_cache[0, t]
        if np.isfinite(geometric_cost):
            event_penalty = check_cooccurrence(0, t, rep_indices_dp)
            dp[t, 1] = geometric_cost + event_penalty
            bp[t, 1] = 0
        else:
             dp[t, 1] = np.inf # Ensure infinite cost propagates


    print("Running dynamic programming for segmentation...")
    dp_fill_start_time = time.time()
    for m in range(2, max_segments + 1):
        for t in range(m - 1, n_timesteps_dp):
            min_cost_for_t_m = np.inf
            best_prev_t_end = -1

            for j in range(m - 2, t):
                cost_prev_segments = dp[j, m - 1]
                cost_last_segment_geom = cost_cache[j + 1, t]

                if np.isfinite(cost_prev_segments) and np.isfinite(cost_last_segment_geom):
                    # Calculate event penalty for the last segment [j+1, t]
                    event_penalty = check_cooccurrence(j + 1, t, rep_indices_dp)
                    current_total_cost = cost_prev_segments + cost_last_segment_geom + event_penalty

                    if current_total_cost < min_cost_for_t_m:
                        min_cost_for_t_m = current_total_cost
                        best_prev_t_end = j

            if best_prev_t_end != -1:
                dp[t, m] = min_cost_for_t_m
                bp[t, m] = best_prev_t_end + 1

    print(f"DP table filled in {time.time() - dp_fill_start_time:.2f} seconds.")

    # --- Find Optimal Number of Segments ---
    final_raw_costs = dp[n_timesteps_dp - 1, 1:] # These costs include the co-occurrence penalties
    segment_counts = np.arange(1, max_segments + 1)
    final_costs_with_lambda = final_raw_costs + lambda_penalty * segment_counts # Add lambda penalty

    valid_indices = np.where(np.isfinite(final_costs_with_lambda))[0]
    if len(valid_indices) == 0:
        print(f"Warning: Could not find any valid segmentation up to {max_segments} segments with finite cost.")
        if np.isfinite(dp[n_timesteps_dp - 1, 1]):
            optimal_num_segments = 1
            min_total_cost = dp[n_timesteps_dp - 1, 1] + lambda_penalty
            print("Falling back to 1 segment.")
        else:
            print("Error: Cannot find any valid segmentation, even 1 segment has infinite cost.")
            return None, -1, np.inf, final_raw_costs
    else:
        optimal_valid_idx = np.argmin(final_costs_with_lambda[valid_indices])
        optimal_num_segments = valid_indices[optimal_valid_idx] + 1
        min_total_cost = final_costs_with_lambda[valid_indices[optimal_valid_idx]]

    print(f"Optimal number of segments found: {optimal_num_segments} with total cost (incl. lambda & event penalties): {min_total_cost:.4f}")

    # --- Backtrack to find segment boundaries (in DP scale) ---
    segment_boundaries_dp = [n_timesteps_dp]
    current_t = n_timesteps_dp - 1
    current_m = optimal_num_segments
    while current_m > 0:
        start_of_current_segment = bp[current_t, current_m]
        if start_of_current_segment == -1:
            print(f"Error during backtracking: Invalid backpointer found at t={current_t}, m={current_m}.")
            if current_m == 1: start_of_current_segment = 0; print("Assuming start at 0.")
            else: break
        segment_boundaries_dp.append(start_of_current_segment)
        current_t = start_of_current_segment - 1
        current_m -= 1
        if start_of_current_segment == 0: break

    final_segment_indices_dp = sorted(list(set(segment_boundaries_dp)))
    if 0 not in final_segment_indices_dp: final_segment_indices_dp.insert(0, 0)
    if n_timesteps_dp not in final_segment_indices_dp: final_segment_indices_dp.append(n_timesteps_dp)
    final_segment_indices_dp = sorted(list(set(final_segment_indices_dp)))

    return final_segment_indices_dp, optimal_num_segments, min_total_cost, final_raw_costs


def map_boundaries_to_original(segment_indices_dp, stride, n_timesteps_original):
    """Maps segment boundaries from downsampled DP scale back to original time scale. (Unchanged)"""
    if stride == 1:
        print("No downsampling was performed; using original boundaries.")
        if segment_indices_dp[-1] != n_timesteps_original:
             segment_indices_dp[-1] = n_timesteps_original
        return sorted(list(set(segment_indices_dp)))

    segment_indices_orig = [int(round(idx * stride)) for idx in segment_indices_dp]

    if segment_indices_orig[-1] > n_timesteps_original:
        print(f"Warning: Mapped final boundary {segment_indices_orig[-1]} exceeds original length {n_timesteps_original}. Clamping.")
        segment_indices_orig[-1] = n_timesteps_original
    elif segment_indices_orig[-1] < n_timesteps_original:
         last_dp_index = (n_timesteps_dp -1)
         if segment_indices_dp[-2] == last_dp_index or segment_indices_dp[-1] == n_timesteps_dp:
             print(f"Adjusting final boundary from {segment_indices_orig[-1]} to {n_timesteps_original} (original end length)")
             segment_indices_orig[-1] = n_timesteps_original

    segment_indices_orig = sorted(list(set(segment_indices_orig)))
    if 0 not in segment_indices_orig:
         segment_indices_orig.insert(0,0)
         segment_indices_orig = sorted(list(set(segment_indices_orig)))

    print(f"Mapped DP boundaries back to original scale indices: {segment_indices_orig}")
    return segment_indices_orig
