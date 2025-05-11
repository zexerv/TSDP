# proof/prefix_sum_proof.py
import numpy as np
import config_proof as config # To access dimension types and weights

def calculate_all_prefix_sums_multidim(T_min_values_NxD, T_max_values_NxD):
    """
    Calculates prefix sums for each dimension of T_min_NxD and T_max_NxD,
    and universal time-related prefix sums.

    Args:
        T_min_values_NxD (np.ndarray): (N x D) array of T_min values.
        T_max_values_NxD (np.ndarray): (N x D) array of T_max values.

    Returns:
        dict: A dictionary containing all prefix sum arrays.
              Keys for per-dimension sums will be like 'ps_Y_min_d0', 'ps_tY_min_d1', etc.
              Universal time sums: 'ps_n', 'ps_t', 'ps_tt'.
              Also includes 'N_timesteps' and 'D_dimensions'.
              Returns None if input arrays are invalid.
    """
    if not isinstance(T_min_values_NxD, np.ndarray) or not isinstance(T_max_values_NxD, np.ndarray):
        print("Error (calculate_all_prefix_sums_multidim): Inputs must be NumPy arrays.")
        return None
    if T_min_values_NxD.ndim != 2 or T_max_values_NxD.ndim != 2:
        print("Error (calculate_all_prefix_sums_multidim): Input arrays must be 2-dimensional (N x D).")
        return None
    if T_min_values_NxD.shape != T_max_values_NxD.shape:
        print("Error (calculate_all_prefix_sums_multidim): T_min and T_max arrays must have the same shape.")
        return None

    N, D = T_min_values_NxD.shape
    if N == 0 or D == 0:
        print("Warning (calculate_all_prefix_sums_multidim): Input arrays are empty (N=0 or D=0).")
        return {
            'ps_n': np.array([0.0]), 'ps_t': np.array([0.0]), 'ps_tt': np.array([0.0]),
            'N_timesteps': 0, 'D_dimensions': 0
        }

    all_prefix_sums = {'N_timesteps': N, 'D_dimensions': D}

    ps_n = np.zeros(N + 1)
    ps_t = np.zeros(N + 1)
    ps_tt = np.zeros(N + 1)

    for j_time_idx in range(N): 
        k_one_based_time = float(j_time_idx + 1)
        ps_n[j_time_idx + 1] = ps_n[j_time_idx] + 1.0
        ps_t[j_time_idx + 1] = ps_t[j_time_idx] + k_one_based_time
        ps_tt[j_time_idx + 1] = ps_tt[j_time_idx] + k_one_based_time**2
    
    all_prefix_sums['ps_n'] = ps_n
    all_prefix_sums['ps_t'] = ps_t
    all_prefix_sums['ps_tt'] = ps_tt

    for d_idx in range(D):
        ps_Y_min_d = np.zeros(N + 1); ps_tY_min_d = np.zeros(N + 1); ps_YY_min_d = np.zeros(N + 1)
        ps_Y_max_d = np.zeros(N + 1); ps_tY_max_d = np.zeros(N + 1); ps_YY_max_d = np.zeros(N + 1)

        for j_time_idx in range(N): 
            k_one_based_time = float(j_time_idx + 1)
            
            y_min_current_d = float(T_min_values_NxD[j_time_idx, d_idx])
            ps_Y_min_d[j_time_idx + 1] = ps_Y_min_d[j_time_idx] + y_min_current_d
            ps_tY_min_d[j_time_idx + 1] = ps_tY_min_d[j_time_idx] + k_one_based_time * y_min_current_d
            ps_YY_min_d[j_time_idx + 1] = ps_YY_min_d[j_time_idx] + y_min_current_d**2

            y_max_current_d = float(T_max_values_NxD[j_time_idx, d_idx])
            ps_Y_max_d[j_time_idx + 1] = ps_Y_max_d[j_time_idx] + y_max_current_d
            ps_tY_max_d[j_time_idx + 1] = ps_tY_max_d[j_time_idx] + k_one_based_time * y_max_current_d
            ps_YY_max_d[j_time_idx + 1] = ps_YY_max_d[j_time_idx] + y_max_current_d**2

        all_prefix_sums[f'ps_Y_min_d{d_idx}'] = ps_Y_min_d
        all_prefix_sums[f'ps_tY_min_d{d_idx}'] = ps_tY_min_d
        all_prefix_sums[f'ps_YY_min_d{d_idx}'] = ps_YY_min_d
        all_prefix_sums[f'ps_Y_max_d{d_idx}'] = ps_Y_max_d
        all_prefix_sums[f'ps_tY_max_d{d_idx}'] = ps_tY_max_d
        all_prefix_sums[f'ps_YY_max_d{d_idx}'] = ps_YY_max_d
        
    return all_prefix_sums

def get_range_sum(prefix_sum_array, start_idx_1based, end_idx_1based):
    if not isinstance(prefix_sum_array, np.ndarray):
        raise TypeError("prefix_sum_array must be a NumPy array.")
    N_data_points = len(prefix_sum_array) - 1
    if start_idx_1based > end_idx_1based: return 0.0 
    if not (1 <= start_idx_1based <= N_data_points and 1 <= end_idx_1based <= N_data_points):
        raise IndexError(
            f"Invalid range for get_range_sum: start_1based={start_idx_1based}, "
            f"end_1based={end_idx_1based} for PS array representing {N_data_points} data points."
        )
    return prefix_sum_array[end_idx_1based] - prefix_sum_array[start_idx_1based - 1]


def calculate_ls_fit_for_segment_1D(
    series_label_prefix, 
    dimension_index_d,
    segment_start_idx_1based, 
    segment_end_idx_1based,
    all_prefix_sums, 
    original_series_data_1D):
    i = segment_start_idx_1based
    j = segment_end_idx_1based
    n_s = j - i + 1

    if n_s < 1: return 0.0, 0.0, np.inf 

    max_0_idx_data = len(original_series_data_1D) - 1
    if not (0 <= i - 1 <= max_0_idx_data and 0 <= j - 1 <= max_0_idx_data):
        return 0.0, 0.0, np.inf

    if n_s == 1: 
        val_at_i = original_series_data_1D[i-1] 
        return 0.0, float(val_at_i), 0.0

    if n_s == 2: 
        val_at_i = original_series_data_1D[i-1]
        val_at_j = original_series_data_1D[j-1]
        slope_a = (val_at_j - val_at_i) / (j - i) if (j - i) != 0 else 0.0
        intercept_b = val_at_i - slope_a * i
        return float(slope_a), float(intercept_b), 0.0

    try:
        S_n = float(n_s) 
        S_t = get_range_sum(all_prefix_sums['ps_t'], i, j)
        S_tt = get_range_sum(all_prefix_sums['ps_tt'], i, j)
        
        # --- CORRECTED KEY CONSTRUCTION ---
        if "_min" in series_label_prefix: # e.g. series_label_prefix is "ps_Y_min"
            type_suffix = "min"
        elif "_max" in series_label_prefix: # e.g. series_label_prefix is "ps_Y_max"
            type_suffix = "max"
        else:
            raise ValueError(f"Invalid series_label_prefix for key construction: {series_label_prefix}")

        ps_Y_key = f'ps_Y_{type_suffix}_d{dimension_index_d}'
        ps_tY_key = f'ps_tY_{type_suffix}_d{dimension_index_d}'
        ps_YY_key = f'ps_YY_{type_suffix}_d{dimension_index_d}'
        # --- END OF CORRECTION ---

        S_Y = get_range_sum(all_prefix_sums[ps_Y_key], i, j)
        S_tY = get_range_sum(all_prefix_sums[ps_tY_key], i, j)
        S_YY = get_range_sum(all_prefix_sums[ps_YY_key], i, j)
    except (IndexError, KeyError) as e:
        # print(f"Error ({series_label_prefix}{dimension_index_d}): Error accessing prefix sums for seg [{i},{j}]. {e}") # Can be verbose
        return 0.0, 0.0, np.inf

    denominator = (S_n * S_tt) - (S_t * S_t)

    if np.isclose(denominator, 0):
        if S_n > 0:
            mean_y = S_Y / S_n
            sse_horizontal = S_YY - (S_Y * S_Y) / S_n
            return 0.0, mean_y, max(0, sse_horizontal)
        return 0.0, 0.0, np.inf

    a = ((S_n * S_tY) - (S_t * S_Y)) / denominator
    b = (S_Y - (a * S_t)) / S_n
    sse = S_YY - (a * S_tY) - (b * S_Y)
    
    return float(a), float(b), max(0.0, float(sse))


def calculate_C_LS_multidim_weighted(
    segment_start_idx_1based, 
    segment_end_idx_1based, 
    all_prefix_sums, 
    original_T_min_NxD, 
    original_T_max_NxD
    ):
    N_from_ps = all_prefix_sums['N_timesteps']
    D_from_ps = all_prefix_sums['D_dimensions']

    if not (1 <= segment_start_idx_1based <= segment_end_idx_1based <= N_from_ps):
        return np.inf, [None]*D_from_ps, [None]*D_from_ps

    total_weighted_CLS_cost = 0.0
    fits_min_all_dims = []
    fits_max_all_dims = []

    if len(config.DIMENSIONS_TO_USE) != D_from_ps:
        print(f"CRITICAL Error (C_LS_multidim): Mismatch D_from_ps ({D_from_ps}) vs len(config.DIMENSIONS_TO_USE) ({len(config.DIMENSIONS_TO_USE)}). Using default weight 1.0.")
        # This indicates a setup issue if D_from_ps (from data) doesn't match config.

    for d_idx in range(D_from_ps):
        dim_name = "unknown_dim"
        weight_d = 1.0 
        
        if d_idx < len(config.DIMENSIONS_TO_USE): # Safety check
            dim_name = config.DIMENSIONS_TO_USE[d_idx]
            if dim_name in config.POSITION_DIMENSIONS:
                weight_d = config.WEIGHT_POSITION
            elif dim_name in config.ROTATION_DIMENSIONS:
                weight_d = config.WEIGHT_ROTATION
        else: # Should not happen if D_from_ps matches len(config.DIMENSIONS_TO_USE)
            print(f"Warning (C_LS_multidim): d_idx {d_idx} out of range for config.DIMENSIONS_TO_USE. Using default weight 1.0 for this dim.")


        a_min_d, b_min_d, SSE_min_d = calculate_ls_fit_for_segment_1D(
            "ps_Y_min", d_idx, 
            segment_start_idx_1based, segment_end_idx_1based,
            all_prefix_sums, 
            original_T_min_NxD[:, d_idx] 
        )
        fits_min_all_dims.append({'a': a_min_d, 'b': b_min_d, 'SSE': SSE_min_d, 'dim_idx': d_idx, 'dim_name': dim_name})

        a_max_d, b_max_d, SSE_max_d = calculate_ls_fit_for_segment_1D(
            "ps_Y_max", d_idx,
            segment_start_idx_1based, segment_end_idx_1based,
            all_prefix_sums,
            original_T_max_NxD[:, d_idx] 
        )
        fits_max_all_dims.append({'a': a_max_d, 'b': b_max_d, 'SSE': SSE_max_d, 'dim_idx': d_idx, 'dim_name': dim_name})

        if np.isinf(SSE_min_d) or np.isinf(SSE_max_d):
            total_weighted_CLS_cost = np.inf 
            break 
        
        total_weighted_CLS_cost += weight_d * (SSE_min_d + SSE_max_d)
        
    return total_weighted_CLS_cost, fits_min_all_dims, fits_max_all_dims


if __name__ == '__main__':
    print("Testing prefix_sum_proof.py (Multi-Dimensional & Weighted Cost - Corrected Keys)...")
    
    N_test, D_test = 3, 2
    T_min_test_NxD = np.array([[0., 9.], [1., 9.], [2., 11.]])
    T_max_test_NxD = np.array([[2., 12.], [3., 11.], [4., 13.]])

    all_ps = calculate_all_prefix_sums_multidim(T_min_test_NxD, T_max_test_NxD)

    if all_ps:
        print(f"\nCalculated Multi-Dim Prefix Sums (N={all_ps['N_timesteps']}, D={all_ps['D_dimensions']}):")
        # ... (assertions for ps_n, ps_t, ps_tt, ps_Y_min_d0, ps_tY_min_d0, ps_Y_max_d1 as before) ...
        assert np.array_equal(all_ps['ps_n'], np.array([0.,1.,2.,3.]))
        assert np.array_equal(all_ps['ps_t'], np.array([0.,1.,3.,6.]))
        assert np.array_equal(all_ps['ps_tt'], np.array([0.,1.,5.,14.]))
        assert np.array_equal(all_ps['ps_Y_min_d0'], np.array([0.,0.,1.,3.]))
        assert np.array_equal(all_ps['ps_tY_min_d0'], np.array([0.,0.,2.,8.]))
        assert np.array_equal(all_ps['ps_YY_min_d0'], np.array([0.,0.,1.,5.])) # 0^2, 1^2, 2^2 -> 0,1,4 -> ps: 0,0,1,5
        assert np.array_equal(all_ps['ps_Y_max_d1'], np.array([0.,12.,23.,36.]))
        assert np.array_equal(all_ps['ps_tY_max_d1'], np.array([0.,12.,34.,73.])) # 1*12, 2*11, 3*13 = 12,22,39 -> ps: 0,12,34,73
        assert np.array_equal(all_ps['ps_YY_max_d1'], np.array([0.,144.,265.,434.]))#12^2,11^2,13^2=144,121,169->ps:0,144,265,434
        print("Basic prefix sum assertions passed.")

        original_dims_to_use = config.DIMENSIONS_TO_USE
        original_pos_dims = config.POSITION_DIMENSIONS
        original_rot_dims = config.ROTATION_DIMENSIONS
        config.DIMENSIONS_TO_USE = ['tx_test', 'ty_test'] 
        config.POSITION_DIMENSIONS = ['tx_test', 'ty_test']
        config.ROTATION_DIMENSIONS = []
        config.WEIGHT_POSITION = 1.0
        config.WEIGHT_ROTATION = 0.0

        print("\nTesting C_LS_multidim_weighted with corrected keys...")
        seg_start, seg_end = 1, 2
        C_LS_md, fits_m, fits_mx = calculate_C_LS_multidim_weighted(
            seg_start, seg_end, all_ps, T_min_test_NxD, T_max_test_NxD
        )
        print(f"C_LS_multidim for segment [{seg_start}-{seg_end}]: {C_LS_md:.4f} (Expected 0.0)")
        assert np.isclose(C_LS_md, 0.0), "C_LS for perfect 2-point lines should be 0"
        
        seg_start2, seg_end2 = 1, 3
        C_LS_md2, fits_m2, fits_mx2 = calculate_C_LS_multidim_weighted(
            seg_start2, seg_end2, all_ps, T_min_test_NxD, T_max_test_NxD
        )
        print(f"C_LS_multidim for segment [{seg_start2}-{seg_end2}]: {C_LS_md2:.4f} (Expected ~2.6667)")
        # SSE_min_d0=0, SSE_min_d1=1.16666, SSE_max_d0=0, SSE_max_d1=1.5. Sum = 2.66666
        assert np.isclose(C_LS_md2, 2.666666666666664), f"C_LS mismatch, got {C_LS_md2}"


        config.DIMENSIONS_TO_USE = original_dims_to_use
        config.POSITION_DIMENSIONS = original_pos_dims
        config.ROTATION_DIMENSIONS = original_rot_dims
        print("Multi-dim C_LS tests with corrected keys passed.")
    else:
        print("Multi-dim prefix sum calculation failed, cannot run C_LS tests.")

