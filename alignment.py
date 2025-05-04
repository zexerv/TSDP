
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from functools import partial
import config # Import configuration
import time # For timing DTW
import sys # For checking module import

# Conditionally import fastdtw based on config
if config.USE_FASTDTW_APPROXIMATION:
    try:
        from fastdtw import fastdtw
    except ImportError:
        print("\n*** WARNING: fastdtw library not found, but USE_FASTDTW_APPROXIMATION is True. ***")
        print("*** Please install fastdtw (`pip install fastdtw`) or set USE_FASTDTW_APPROXIMATION to False. ***\n")
        # Set flag to False to prevent errors later if import failed
        config.USE_FASTDTW_APPROXIMATION = False


def calculate_derivative(trajectory):
    """Estimates derivative using simple finite difference."""
    if len(trajectory) < 2:
        return np.zeros_like(trajectory)
    return np.diff(trajectory, axis=0, prepend=trajectory[0:1,:])

def map_events_using_dtw_path(original_event_indices, dtw_path, traj_len):
    """Maps event indices using the DTW path. (Unchanged)"""
    if not dtw_path: return {key: [] for key in original_event_indices}
    mapped_events = {key: [] for key in original_event_indices}
    path_array = np.array(dtw_path)
    traj_to_ref_map = {}
    last_traj_idx = -1
    for ref_i, traj_i in dtw_path:
        if traj_i > last_traj_idx:
            for k in range(last_traj_idx + 1, traj_i + 1):
                 if k not in traj_to_ref_map: traj_to_ref_map[k] = ref_i
            last_traj_idx = traj_i
        elif traj_i not in traj_to_ref_map: traj_to_ref_map[traj_i] = ref_i
    mapped_ref = -1
    full_traj_to_ref_map = {}
    for i in range(traj_len):
        if i in traj_to_ref_map: mapped_ref = traj_to_ref_map[i]
        full_traj_to_ref_map[i] = mapped_ref
    for event_type, indices in original_event_indices.items():
        mapped_indices = set()
        for orig_idx in indices:
            if 0 <= orig_idx < traj_len:
                 mapped_idx = full_traj_to_ref_map.get(orig_idx, -1)
                 if mapped_idx != -1: mapped_indices.add(mapped_idx)
        mapped_events[event_type] = sorted(list(mapped_indices))
    return mapped_events

def weighted_euclidean(p, q, weights):
    """Calculates the weighted Euclidean distance. (Unchanged)"""
    diff = p - q
    squared_diff = diff * diff
    weighted_squared_diff = weights * squared_diff
    return np.sqrt(np.sum(weighted_squared_diff))

# --- MODIFIED Helper to find first AND second pre/post state change waypoints ---
def find_pre_post_waypoints(state_change_idx, wp_saved_indices, n=2):
    """
    Finds the indices of the first 'n' waypoints before and the first 'n' waypoints
    at or after the state change event.

    Args:
        state_change_idx (int or None): Index of the first state change event.
        wp_saved_indices (list): Sorted list of waypoint indices.
        n (int): Number of pre/post waypoints to find.

    Returns:
        tuple: (list_of_pre_wp_indices, list_of_post_wp_indices)
               Lists contain indices or None if fewer than 'n' are found.
               Example for n=2: ([pre_wp2, pre_wp1], [post_wp1, post_wp2])
               Indices are ordered closest to state_change first (pre_wp1, post_wp1).
    """
    pre_wp_indices = [None] * n
    post_wp_indices = [None] * n

    if not wp_saved_indices or state_change_idx is None:
        return pre_wp_indices, post_wp_indices

    # Find waypoints *before* state change
    pre_wps = [wp for wp in wp_saved_indices if wp < state_change_idx]
    pre_wps.reverse() # Closest to state_change is now first
    for i in range(min(n, len(pre_wps))):
        pre_wp_indices[i] = pre_wps[i]

    # Find waypoints *at or after* state change
    post_wps = [wp for wp in wp_saved_indices if wp >= state_change_idx]
    for i in range(min(n, len(post_wps))):
        post_wp_indices[i] = post_wps[i]

    # Return lists ordered closest to state_change first
    # Note: pre_wp_indices list order is [closest, 2nd_closest,...] from the reverse() above
    return pre_wp_indices, post_wp_indices

# --- NEW Helper to find first pre/post gripper change ---
def find_first_gripper_changes(state_change_idx, gripper_change_indices):
    """
    Finds the index of the first gripper change before and the first gripper change
    at or after the state change event.

    Args:
        state_change_idx (int or None): Index of the first state change event.
        gripper_change_indices (list): Sorted list of gripper change indices.

    Returns:
        tuple: (first_pre_gripper_idx, first_post_gripper_idx)
               Indices can be None if no corresponding change is found.
    """
    first_pre_gripper_idx = None
    first_post_gripper_idx = None

    if not gripper_change_indices or state_change_idx is None:
        return None, None

    # Find first gripper change *before* state change
    pre_grippers = [gc for gc in gripper_change_indices if gc < state_change_idx]
    if pre_grippers:
        first_pre_gripper_idx = pre_grippers[-1] # Last one before

    # Find first gripper change *at or after* state change
    post_grippers = [gc for gc in gripper_change_indices if gc >= state_change_idx]
    if post_grippers:
        first_post_gripper_idx = post_grippers[0] # First one at or after

    return first_pre_gripper_idx, first_post_gripper_idx


# --- REVISED: Manual DTW Implementation with Conditional & Enhanced Penalties ---
def manual_dtw_with_conditional_penalties(
    ref_seq, traj_seq, dist_func,
    # Event indices for reference sequence
    ref_sc_indices, ref_wp_indices_pre, ref_wp_indices_post, ref_gripper_indices_pre, ref_gripper_indices_post,
    # Event indices for trajectory sequence
    traj_sc_indices, traj_wp_indices_pre, traj_wp_indices_post, traj_gripper_indices_pre, traj_gripper_indices_post,
    # Penalty values
    sc_penalty, wp_penalties_pre, wp_penalties_post, gripper_penalties_pre, gripper_penalties_post,
    # Activation flags (for N>1 waypoints)
    activate_wp_penalties_pre, activate_wp_penalties_post
    ):
    """
    Performs DTW with penalties for state_change, gripper changes, and
    conditionally applied penalties for N>1 waypoints.

    Args:
        ref_seq, traj_seq: Sequences to align.
        dist_func: Base distance function.
        ref_..._indices / traj_..._indices: Sets/Lists of indices for each event type.
            wp_indices_pre/post: Lists like [[wp1_idx, wp2_idx,...], [wp1_idx, wp2_idx,...]]
            gripper_indices_pre/post: Lists like [[gc1_idx], [gc1_idx]] (only first supported now)
        sc_penalty, ..._penalties_...: Penalty values for each event type.
            wp_penalties_pre/post: Lists of penalties [wp1_pen, wp2_pen, ...]
            gripper_penalties_pre/post: Lists of penalties [gc1_pen]
        activate_wp_penalties_pre/post: Boolean lists [activate_wp1, activate_wp2, ...]
                                        Indicates if penalty for N-th WP should be applied globally.

    Returns:
        tuple: (total_distance, path)
    """
    n_ref = len(ref_seq)
    n_traj = len(traj_seq)
    if n_ref == 0 or n_traj == 0: return np.inf, []

    cost_matrix = np.full((n_ref + 1, n_traj + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    # Helper to check if an index matches a specific event index in a list
    def is_nth_event(idx, event_list, n):
        return len(event_list) > n and event_list[n] is not None and idx == event_list[n]

    for i in range(1, n_ref + 1):
        for j in range(1, n_traj + 1):
            ref_idx = i - 1
            traj_idx = j - 1
            cost = dist_func(ref_seq[ref_idx], traj_seq[traj_idx])

            # --- Add event penalties ---
            # 1. State Change
            ref_is_sc = ref_idx in ref_sc_indices
            traj_is_sc = traj_idx in traj_sc_indices
            if (ref_is_sc != traj_is_sc): cost += sc_penalty

            # 2. Waypoints (Pre & Post)
            for n in range(len(wp_penalties_pre)): # Iterate up to max supported waypoints (e.g., N=2)
                # Pre-Waypoints (only apply if activation flag is True for n>0)
                if n == 0 or activate_wp_penalties_pre[n]:
                    ref_is_wp_pre = is_nth_event(ref_idx, ref_wp_indices_pre, n)
                    traj_is_wp_pre = is_nth_event(traj_idx, traj_wp_indices_pre, n)
                    if (ref_is_wp_pre != traj_is_wp_pre): cost += wp_penalties_pre[n]
                # Post-Waypoints (only apply if activation flag is True for n>0)
                if n == 0 or activate_wp_penalties_post[n]:
                    ref_is_wp_post = is_nth_event(ref_idx, ref_wp_indices_post, n)
                    traj_is_wp_post = is_nth_event(traj_idx, traj_wp_indices_post, n)
                    if (ref_is_wp_post != traj_is_wp_post): cost += wp_penalties_post[n]

            # 3. Gripper Changes (Pre & Post - currently only first)
            for n in range(len(gripper_penalties_pre)): # Currently n=0 only
                 ref_is_gc_pre = is_nth_event(ref_idx, ref_gripper_indices_pre, n)
                 traj_is_gc_pre = is_nth_event(traj_idx, traj_gripper_indices_pre, n)
                 if (ref_is_gc_pre != traj_is_gc_pre): cost += gripper_penalties_pre[n]

                 ref_is_gc_post = is_nth_event(ref_idx, ref_gripper_indices_post, n)
                 traj_is_gc_post = is_nth_event(traj_idx, traj_gripper_indices_post, n)
                 if (ref_is_gc_post != traj_is_gc_post): cost += gripper_penalties_post[n]
            # --- End penalty addition ---

            cost_matrix[i, j] = cost + min(cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1])

    total_distance = cost_matrix[n_ref, n_traj]

    # Backtrack - Corrected index storage
    path = []
    i, j = n_ref, n_traj
    while i > 0 or j > 0:
        # Store indices corresponding to original sequences (0 to N-1)
        path.append((i-1, j-1))

        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            min_prev_cost = min(cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1])
            # Prioritize diagonal move in case of tie
            if cost_matrix[i-1, j-1] == min_prev_cost: i -= 1; j -= 1
            elif cost_matrix[i-1, j] == min_prev_cost: i -= 1
            else: j -= 1

    path.reverse()
    return total_distance, path


# --- REVISED: align_trajectories Function ---
def align_trajectories(original_trajs_df_list, original_events_list, alignment_type,
                       alignment_base_feature_cols, all_feature_cols, normalize_combined=True):
    """
    Aligns trajectories using either fastdtw or manual DTW with conditional penalties.
    Performs pre-processing to determine which N>1 waypoint penalties to activate.
    """
    n_trajectories = len(original_trajs_df_list)
    if n_trajectories == 0 or not original_events_list or n_trajectories != len(original_events_list):
        print("Error: Input trajectory list or events list is empty or mismatched.")
        return [], [], -1
    if alignment_type not in ['DTW', 'DDTW', 'Combined']:
        raise ValueError("alignment_type must be 'DTW', 'DDTW', or 'Combined'")

    # --- Basic Checks and Data Extraction ---
    first_df_cols = original_trajs_df_list[0].columns.tolist()
    missing_align_cols = [col for col in alignment_base_feature_cols if col not in first_df_cols]
    if missing_align_cols: raise ValueError(f"Alignment features missing: {missing_align_cols}")

    trajectories_align_np = [df[alignment_base_feature_cols].values for df in original_trajs_df_list]
    trajectories_full_np = [df[all_feature_cols].values for df in original_trajs_df_list]
    lengths = [len(t) for t in trajectories_align_np]
    if not lengths or max(lengths) == 0: return [], [], -1

    reference_index = np.argmax(lengths)
    ref_len = lengths[reference_index]
    print(f"\nUsing trajectory {reference_index} (length {ref_len}) as reference for {alignment_type}.")

    # --- Determine DTW Method ---
    use_manual_dtw = not config.USE_FASTDTW_APPROXIMATION
    impl_type = "Manual" if use_manual_dtw else "FastDTW"
    if use_manual_dtw:
        print("Using Manual DTW implementation with event penalties.")
    else:
        if 'fastdtw' not in sys.modules:
             print("\n*** ERROR: USE_FASTDTW_APPROXIMATION is True, but fastdtw could not be imported. ***")
             return [], [], -1
        print("Using fastdtw library (approximation, no event penalties).")

    # --- Pre-processing for Manual DTW Event Penalties ---
    all_traj_event_details = []
    activate_pre_wp_penalties = [True] # 1st waypoint penalty always active if found
    activate_post_wp_penalties = [True]
    max_waypoints_to_consider = 2 # Check up to the 2nd pre/post waypoint

    if use_manual_dtw:
        print("Pre-processing event details for manual DTW...")
        counts_pre_wp = [0] * max_waypoints_to_consider
        counts_post_wp = [0] * max_waypoints_to_consider

        for i in range(n_trajectories):
            events = original_events_list[i]
            sc_indices = set(events.get('state_change', []))
            wp_indices = sorted(events.get('wp_saved', []))
            gc_indices = sorted(events.get('gripper_change', [])) # Gripper changes

            first_sc_idx = min(sc_indices) if sc_indices else None

            # Find up to N waypoints pre/post
            wp_indices_pre, wp_indices_post = find_pre_post_waypoints(first_sc_idx, wp_indices, n=max_waypoints_to_consider)

            # Find first gripper change pre/post
            gripper_idx_pre, gripper_idx_post = find_first_gripper_changes(first_sc_idx, gc_indices)

            details = {
                "sc_indices": sc_indices,
                "wp_indices_pre": wp_indices_pre,   # List [wp1, wp2]
                "wp_indices_post": wp_indices_post, # List [wp1, wp2]
                "gripper_indices_pre": [gripper_idx_pre], # List [gc1]
                "gripper_indices_post": [gripper_idx_post] # List [gc1]
            }
            all_traj_event_details.append(details)

            # Count presence of N-th waypoints for activation check
            for n in range(max_waypoints_to_consider):
                if wp_indices_pre[n] is not None: counts_pre_wp[n] += 1
                if wp_indices_post[n] is not None: counts_post_wp[n] += 1

        # Determine activation flags for N>1 waypoints based on threshold
        threshold = config.WAYPOINT_PRESENCE_THRESHOLD
        for n in range(1, max_waypoints_to_consider): # Start from N=2 (index 1)
            activate_pre = (counts_pre_wp[n] / n_trajectories) >= threshold if n_trajectories > 0 else False
            activate_post = (counts_post_wp[n] / n_trajectories) >= threshold if n_trajectories > 0 else False
            activate_pre_wp_penalties.append(activate_pre)
            activate_post_wp_penalties.append(activate_post)
            print(f"  N={n+1} Pre-WP found in {counts_pre_wp[n]}/{n_trajectories}. Penalty Active: {activate_pre}")
            print(f"  N={n+1} Post-WP found in {counts_post_wp[n]}/{n_trajectories}. Penalty Active: {activate_post}")

        # Prepare penalty lists (ensure length matches max_waypoints_to_consider)
        wp_penalties_pre = [config.ALIGNMENT_PRE_WP1_PENALTY, config.ALIGNMENT_PRE_WP2_PENALTY][:max_waypoints_to_consider]
        wp_penalties_post = [config.ALIGNMENT_POST_WP1_PENALTY, config.ALIGNMENT_POST_WP2_PENALTY][:max_waypoints_to_consider]
        gripper_penalties_pre = [config.ALIGNMENT_PRE_GRIPPER1_PENALTY]
        gripper_penalties_post = [config.ALIGNMENT_POST_GRIPPER1_PENALTY]

    # --- Alignment Loop ---
    aligned_trajectories_np = []
    mapped_events_list = []
    dtw_times = []
    ref_event_details = all_traj_event_details[reference_index] if use_manual_dtw else None

    for i, traj_align in enumerate(trajectories_align_np):
        traj_full = trajectories_full_np[i]
        original_events = original_events_list[i]
        traj_len_orig = len(traj_align)
        curr_event_details = all_traj_event_details[i] if use_manual_dtw else None

        if i == reference_index:
            aligned_trajectories_np.append(trajectories_full_np[i].copy())
            mapped_events_list.append(original_events.copy())
            continue
        if traj_len_orig == 0:
            aligned_trajectories_np.append(np.zeros_like(trajectories_full_np[reference_index]))
            mapped_events_list.append({key: [] for key in original_events})
            continue

        print(f"  Aligning trajectory {i} (length {traj_len_orig}) to reference...")

        # Prepare data for DTW (DTW, DDTW, Combined) - same logic as before
        ref_data_for_dtw, traj_data_for_dtw = None, None
        # ... (data prep logic omitted for brevity - it's unchanged) ...
        if alignment_type == 'DTW':
            ref_data_for_dtw = trajectories_align_np[reference_index]
            traj_data_for_dtw = traj_align
        elif alignment_type == 'DDTW':
            ref_data_for_dtw = calculate_derivative(trajectories_align_np[reference_index])
            traj_data_for_dtw = calculate_derivative(traj_align)
            if not np.any(ref_data_for_dtw) and not np.any(traj_data_for_dtw):
                print("    Warning: Both derivative sequences zero. Using DTW fallback.")
                ref_data_for_dtw = trajectories_align_np[reference_index]
                traj_data_for_dtw = traj_align
        elif alignment_type == 'Combined':
            # ... (combined logic with normalization) ...
             ref_deriv = calculate_derivative(trajectories_align_np[reference_index])
             traj_deriv = calculate_derivative(traj_align)
             ref_pos = trajectories_align_np[reference_index]
             traj_pos = traj_align
             if normalize_combined:
                 # ... (normalization logic) ...
                 scaler_pos = MinMaxScaler(); scaler_deriv = MinMaxScaler()
                 combined_pos = np.vstack((ref_pos, traj_pos)); combined_deriv = np.vstack((ref_deriv, traj_deriv))
                 valid_pos_cols = np.where(np.ptp(combined_pos, axis=0) > 1e-6)[0]; valid_deriv_cols = np.where(np.ptp(combined_deriv, axis=0) > 1e-6)[0]
                 ref_pos_norm = ref_pos.copy(); traj_pos_norm = traj_pos.copy(); ref_deriv_norm = ref_deriv.copy(); traj_deriv_norm = traj_deriv.copy()
                 if len(valid_pos_cols) > 0:
                     scaler_pos.fit(np.vstack((ref_pos[:, valid_pos_cols], traj_pos[:, valid_pos_cols])))
                     ref_pos_norm[:, valid_pos_cols] = scaler_pos.transform(ref_pos[:, valid_pos_cols]); traj_pos_norm[:, valid_pos_cols] = scaler_pos.transform(traj_pos[:, valid_pos_cols])
                 if len(valid_deriv_cols) > 0:
                     scaler_deriv.fit(np.vstack((ref_deriv[:, valid_deriv_cols], traj_deriv[:, valid_deriv_cols])))
                     ref_deriv_norm[:, valid_deriv_cols] = scaler_deriv.transform(ref_deriv[:, valid_deriv_cols]); traj_deriv_norm[:, valid_deriv_cols] = scaler_deriv.transform(traj_deriv[:, valid_deriv_cols])
                 ref_data_for_dtw = np.hstack((ref_pos_norm, ref_deriv_norm)); traj_data_for_dtw = np.hstack((traj_pos_norm, traj_deriv_norm))
             else:
                 ref_data_for_dtw = np.hstack((ref_pos, ref_deriv)); traj_data_for_dtw = np.hstack((traj_pos, traj_deriv))


        # Calculate Weights for DTW distance - same logic as before
        n_dtw_dims = ref_data_for_dtw.shape[1]
        dtw_weights = np.ones(n_dtw_dims)
        # ... (weight application logic omitted for brevity - it's unchanged) ...
        for k in range(n_dtw_dims):
             base_feature_name = None; is_derivative = False
             if alignment_type == 'Combined':
                 if k < len(alignment_base_feature_cols): base_feature_name = alignment_base_feature_cols[k]; is_derivative = False
                 else: base_feature_name = alignment_base_feature_cols[k - len(alignment_base_feature_cols)]; is_derivative = True
             else: base_feature_name = alignment_base_feature_cols[k]; is_derivative = (alignment_type == 'DDTW')
             if base_feature_name not in config.POS_COLS: dtw_weights[k] *= config.ALIGNMENT_ROTATION_WEIGHT
             if is_derivative and alignment_type == 'Combined': dtw_weights[k] *= config.ALIGNMENT_DERIVATIVE_WEIGHT


        # Perform DTW using the selected method
        base_dist_func = partial(weighted_euclidean, weights=dtw_weights)
        distance = np.inf
        path = []
        dtw_start = time.time()

        if use_manual_dtw:
            distance, path = manual_dtw_with_conditional_penalties(
                ref_data_for_dtw, traj_data_for_dtw, base_dist_func,
                ref_event_details["sc_indices"], ref_event_details["wp_indices_pre"], ref_event_details["wp_indices_post"],
                ref_event_details["gripper_indices_pre"], ref_event_details["gripper_indices_post"],
                curr_event_details["sc_indices"], curr_event_details["wp_indices_pre"], curr_event_details["wp_indices_post"],
                curr_event_details["gripper_indices_pre"], curr_event_details["gripper_indices_post"],
                config.ALIGNMENT_STATE_CHANGE_PENALTY, wp_penalties_pre, wp_penalties_post,
                gripper_penalties_pre, gripper_penalties_post,
                activate_pre_wp_penalties, activate_post_wp_penalties
            )
        else:
            distance, path = fastdtw(ref_data_for_dtw, traj_data_for_dtw, dist=base_dist_func)

        dtw_end = time.time()
        dtw_times.append(dtw_end - dtw_start)
        print(f"    {impl_type} DTW ({alignment_type}, weighted) distance: {distance:.4f} (took {dtw_end - dtw_start:.3f}s)")

        # Warp the FULL trajectory - same logic as before
        if not path:
             print(f"    Warning: DTW returned empty path for trajectory {i}. Filling with zeros.")
             aligned_trajectories_np.append(np.zeros_like(trajectories_full_np[reference_index]))
             mapped_events_list.append({key: [] for key in original_events})
             continue
        # ... (warping logic omitted for brevity - it's unchanged) ...
        path_array = np.array(path)
        warped_traj_full = np.zeros_like(trajectories_full_np[reference_index])
        ref_to_traj_map = {}
        for ref_idx_p, traj_idx_p in path: ref_to_traj_map[ref_idx_p] = traj_idx_p
        last_valid_traj_idx = 0
        if 0 in ref_to_traj_map: last_valid_traj_idx = min(ref_to_traj_map[0], traj_len_orig -1)
        for ref_idx in range(ref_len):
             if ref_idx in ref_to_traj_map:
                  traj_idx = min(ref_to_traj_map[ref_idx], traj_len_orig - 1)
                  warped_traj_full[ref_idx] = traj_full[traj_idx]
                  last_valid_traj_idx = traj_idx
             else:
                 valid_idx = min(last_valid_traj_idx, traj_len_orig - 1)
                 if valid_idx >= 0 : warped_traj_full[ref_idx] = traj_full[valid_idx]
        aligned_trajectories_np.append(warped_traj_full)


        # Map events using the path - same logic as before
        mapped_events = map_events_using_dtw_path(original_events, path, traj_len_orig)
        mapped_events_list.append(mapped_events)

    # --- Final Steps ---
    aligned_trajectories_df = [pd.DataFrame(data=arr, columns=all_feature_cols) for arr in aligned_trajectories_np]
    if dtw_times: print(f"  Average DTW time per trajectory ({impl_type}): {np.mean(dtw_times):.3f}s")
    return aligned_trajectories_df, mapped_events_list, reference_index