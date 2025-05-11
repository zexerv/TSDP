# proof/segmentation_pelt_proof.py
import numpy as np
import time

def pelt_segmentation(N, lambda_penalty, calculate_C_LS_func, 
                      all_prefix_sums, original_Y_min_NxD, original_Y_max_NxD,
                      log_R_history=False): 
    """
    Implements the PELT algorithm with corrected pruning logic.
    Records active R_candidates and returns P_changepoints.

    Args:
        N (int): Length of the time series.
        lambda_penalty (float): Penalty for adding a new segment.
        calculate_C_LS_func (function): Function to calculate segment cost.
        all_prefix_sums (dict): Dictionary of all prefix sum arrays.
        original_Y_min_NxD (np.ndarray): Full (N x D) T_min data.
        original_Y_max_NxD (np.ndarray): Full (N x D) T_max data.
        log_R_history (bool): If True, logs |R| at intervals.

    Returns:
        tuple: (optimal_segment_endpoints, F_N, max_R_candidates_size, 
                active_R_sets_for_t, R_candidates_length_history, P_changepoints_array)
               P_changepoints_array: Array where P[t] is the optimal tau for F[t].
    """
    if N <= 0:
        return [0], -lambda_penalty, 0, [[] for _ in range(N + 1)], [], np.zeros(N+1, dtype=int)

    pelt_start_time = time.time()
    F_costs = np.full(N + 1, np.inf)
    P_changepoints_array = np.zeros(N + 1, dtype=int) # Store P[t] = best_tau
    F_costs[0] = -lambda_penalty 
    R_candidates = [0]      
    max_R_candidates_size = 1 
    active_R_sets_for_t = [[] for _ in range(N + 1)] 
    active_R_sets_for_t[0] = [0] 
    R_candidates_length_history = [] 

    for t_current_end in range(1, N + 1): 
        min_cost_for_t = np.inf
        best_tau_for_t = 0 
        active_R_sets_for_t[t_current_end] = list(R_candidates) 
        candidate_costs_info = [] 

        for tau_candidate in R_candidates: 
            if tau_candidate >= t_current_end : continue
            C_LS_val, _, _ = calculate_C_LS_func(
                tau_candidate + 1, t_current_end, all_prefix_sums, 
                original_Y_min_NxD, original_Y_max_NxD)
            if np.isinf(C_LS_val): continue
            current_total_cost = F_costs[tau_candidate] + C_LS_val + lambda_penalty
            candidate_costs_info.append({'tau': tau_candidate, 'cost': current_total_cost, 'C_LS_segment': C_LS_val})
            if current_total_cost < min_cost_for_t:
                min_cost_for_t = current_total_cost
                best_tau_for_t = tau_candidate
        
        if not np.isinf(min_cost_for_t):
            F_costs[t_current_end] = min_cost_for_t
            P_changepoints_array[t_current_end] = best_tau_for_t # Store the chosen tau
        else:
            F_costs[t_current_end] = np.inf 
            P_changepoints_array[t_current_end] = R_candidates[0] if R_candidates else 0

        R_new_candidates = []
        F_prime_t_current_end = F_costs[t_current_end] - lambda_penalty
        for info in candidate_costs_info: 
            tau_k = info['tau']; C_LS_k = info['C_LS_segment'] 
            if F_costs[tau_k] + C_LS_k <= F_prime_t_current_end:
                R_new_candidates.append(tau_k)
        R_new_candidates.append(t_current_end) 
        R_candidates = sorted(list(set(R_new_candidates))) 
        max_R_candidates_size = max(max_R_candidates_size, len(R_candidates))
        
        log_interval = N // 10 if N >=10 else 1
        if log_R_history or (t_current_end % log_interval == 0) or t_current_end == N :
             R_candidates_length_history.append((t_current_end, len(R_candidates)))

    optimal_segment_endpoints = [N] 
    current_cp_val = N 
    while current_cp_val > 0:
        # Use P_changepoints_array for backtracking
        prev_cp_val = P_changepoints_array[current_cp_val] 
        optimal_segment_endpoints.append(prev_cp_val)
        current_cp_val = prev_cp_val
    optimal_segment_endpoints = sorted(list(set(optimal_segment_endpoints))) 
    
    return (optimal_segment_endpoints, F_costs[N], max_R_candidates_size, 
            active_R_sets_for_t, R_candidates_length_history, P_changepoints_array)

if __name__ == '__main__':
    print("Testing segmentation_pelt_proof.py (Returns P_changepoints)...")
    def mock_C_LS_md_simple(s, e, ps, ymin, ymax):
        l = e - s + 1; c = 1.0 if s != e else 0.0; d = ymin.shape[1] if ymin.ndim==2 else 1
        fits = [{'a':0,'b':0,'SSE':c/(2*d if d>0 else 1),'dim_idx':i,'dim_name':f'd{i}'} for i in range(d)]
        return c, fits, fits

    N_test, D_test, lambda_test = 10, 1, 0.1
    ps_md, ymin_NxD, ymax_NxD = {'N_timesteps':N_test,'D_dimensions':D_test}, np.zeros((N_test,D_test)), np.zeros((N_test,D_test))
    
    endpoints, cost, max_R, active_R, R_len_hist, P_cps = pelt_segmentation(
        N_test, lambda_test, mock_C_LS_md_simple, ps_md, ymin_NxD, ymax_NxD, True)

    print(f"Endpoints: {endpoints}, Cost: {cost:.2f}, Max|R|: {max_R}")
    print(f"P_changepoints: {P_cps}")
    # Expected P_changepoints for this cost: P[t] = t-1
    expected_P = np.arange(N_test + 1); expected_P[0]=0 # P[0] is not strictly defined by loop, but P[1]=0
    for t_val in range(1, N_test + 1): expected_P[t_val] = t_val -1
    
    assert np.array_equal(P_cps, expected_P), f"P_changepoints incorrect. Got {P_cps}, Expected {expected_P}"
    print("Assertions for P_changepoints passed.")
