# optimization.py
import numpy as np
import time
import config # Read configuration parameters
import utils # Need utils for transformations and IK/FK helpers
from planner import OMPLPlanner # Need planner for cost evaluation
import traceback # For printing detailed error messages
from typing import Optional, Tuple
# --- Dependency Checks ---
SCIPY_AVAILABLE = utils.SCIPY_AVAILABLE # Check if SciPy (for rotations) is available

# Check for PySwarms (for PSO)
try:
    import pyswarms as ps
    # Import necessary handlers or structures if using bounds or specific features
    # from pyswarms.backend.handlers import BoundaryHandler
    PY_SWARMS_AVAILABLE = True
    print("INFO (optimization.py): PySwarms found.")
except ImportError:
    PY_SWARMS_AVAILABLE = False
    print("WARNING (optimization.py): PySwarms not found. PSO optimization unavailable.")

# --- Cost Function for GMM Waypoint Optimization ---

def gmm_waypoint_cost_function(
    particle_batch: np.ndarray, # Batch of 6D state vectors [N_particles, 6]
    gmm_params: dict, # GMM params for this boundary
    branch_tuple: tuple, # The specific IK branch to use
    q_previous_waypoint: np.ndarray, # C-space config of previous waypoint
    planner: OMPLPlanner, # Planner instance
    H_B_I: np.ndarray # Transform Base to Interface
    ) -> np.ndarray:
    """
    Calculates the combined cost for a batch of candidate waypoints (particles).
    Cost = w_plan * Cost_Plan + w_gmm * Cost_GMM
    (Implementation remains the same as in immersive id="optimization_py_gmm_fix")
    """
    n_particles = particle_batch.shape[0]
    costs = np.full(n_particles, float('inf'))
    weights = gmm_params.get('weights'); means = gmm_params.get('means'); covariances_list = gmm_params.get('covariances')
    if weights is None or means is None or covariances_list is None: return costs
    weights = np.array(weights); means = np.array(means); covariances_list = np.array(covariances_list)
    n_components = len(weights)
    inv_covariances = []
    try:
        reg_covar = 1e-6
        if covariances_list.shape != (n_components, 6, 6): raise ValueError("Covariances shape mismatch")
        for cov in covariances_list: inv_covariances.append(np.linalg.inv(cov + np.identity(6) * reg_covar))
    except Exception as e: print(f"ERROR: Could not process GMM covariances: {e}."); return costs
    eval_planning_time = config.GMM_OPT_EVAL_PLANNING_TIME
    for i in range(n_particles):
        z_particle = particle_batch[i]; p_I = z_particle[:3]; v_log_I = z_particle[3:]
        min_mahalanobis_sq = float('inf'); valid_dist_found = False
        if n_components > 0:
            for k in range(n_components):
                diff = z_particle - means[k]
                try:
                    dist_sq = diff.T @ inv_covariances[k] @ diff
                    if np.isfinite(dist_sq): min_mahalanobis_sq = min(min_mahalanobis_sq, dist_sq); valid_dist_found = True
                except Exception: continue
            cost_gmm = min_mahalanobis_sq if valid_dist_found else float('inf')
        else: cost_gmm = float('inf')
        if not np.isfinite(cost_gmm): costs[i] = float('inf'); continue
        cost_plan = float('inf')
        R_I = utils.exp_map_so3(v_log_I)
        if R_I is None: costs[i] = float('inf'); continue
        T_I_target = np.identity(4); T_I_target[:3, :3] = R_I; T_I_target[:3, 3] = p_I
        T_B_target = H_B_I @ T_I_target
        ik_result = utils.get_ik_solution_for_branch(T_B_target, branch_tuple, q_previous_waypoint)
        if ik_result is None or ik_result[0] is None: q_goal_branch = None
        else: q_goal_branch, _ = ik_result
        if q_goal_branch is None: costs[i] = float('inf'); continue
        _, path_cost = planner.plan(q_previous_waypoint, q_goal_branch, time_limit=eval_planning_time)
        cost_plan = path_cost if np.isfinite(path_cost) else float('inf')
        total_cost = (config.GMM_OPT_WEIGHT_PLANNING * cost_plan + config.GMM_OPT_WEIGHT_GMM_DIST * cost_gmm)
        costs[i] = total_cost if np.isfinite(total_cost) else float('inf')
    return costs


# --- NEW: GMM Waypoint Optimization Runner ---

def run_gmm_waypoint_optimization(
    gmm_params: dict,
    branch_tuple: tuple,
    q_previous_waypoint: np.ndarray,
    planner: OMPLPlanner,
    H_B_I: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, dict]:
    """
    Runs PSO or GD to find the best 6D waypoint state z = [p_I, v_log_I]
    within the GMM distribution for a specific boundary and IK branch.
    MODIFIED to store optimization history in opt_info.
    """
    # Initialize opt_info with algorithm placeholder and history lists
    opt_info = {
        'algorithm': f'GMM_OPT_{config.GMM_OPT_ALGORITHM}_FAILED',
        'cost_history': [],
        'pos_history': [], # To store the sequence of best 6D states [p_I, v_log_I]
        'initial_guess': None # To store the starting 6D state
        }
    z_optimal = None
    q_optimal_branch = None
    optimal_cost = float('inf')

    # --- Determine Initial Guess ---
    z_initial_guess = None
    try:
        weights = gmm_params.get('weights'); means = gmm_params.get('means')
        if weights is not None and means is not None and len(weights) > 0 and len(means) > 0:
            best_k = np.argmax(weights); z_initial_guess = np.array(means[best_k])
        else: z_initial_guess = np.zeros(6)
    except Exception as e_init: print(f"Warning: Could not determine initial guess: {e_init}"); z_initial_guess = np.zeros(6)
    opt_info['initial_guess'] = z_initial_guess # Store initial guess

    bounds = None # Define bounds later if needed

    # --- Run Selected Optimization Algorithm ---
    if config.GMM_OPT_ALGORITHM == "PSO":
        if not PY_SWARMS_AVAILABLE:
            opt_info['algorithm'] = 'GMM_OPT_PSO_UNAVAILABLE'; return None, None, float('inf'), opt_info

        print(f"  Running PSO for GMM Waypoint Optimization (Branch {branch_tuple})...")
        n_particles = config.GMM_PSO_N_PARTICLES; dimensions = 6
        options = config.GMM_PSO_OPTIONS; max_iters = config.GMM_PSO_ITERATIONS
        cost_func_args = {'gmm_params': gmm_params, 'branch_tuple': branch_tuple, 'q_previous_waypoint': q_previous_waypoint, 'planner': planner, 'H_B_I': H_B_I}
        init_pos = z_initial_guess + np.random.randn(n_particles, dimensions) * 0.01 if z_initial_guess is not None else None

        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds, init_pos=init_pos)

        try:
            # Optimize and capture history
            best_cost_pso, best_pos_pso = optimizer.optimize(objective_func=gmm_waypoint_cost_function, iters=max_iters, n_processes=None, verbose=False, **cost_func_args) # verbose=False

            # Store history from the optimizer object
            opt_info['cost_history'] = optimizer.cost_history
            # Store the history of the *best* particle's position found so far at each iteration
            # PySwarms stores this in optimizer.pos_history if using SwarmSaver,
            # otherwise we might need to approximate or use the global best history.
            # Let's store the global best position history for simplicity.
            # Note: Check PySwarms version documentation for exact history attribute names.
            # Assuming optimizer has `global_best_pos` attribute updated internally or similar.
            # We'll store the final swarm positions for visualization instead, as history might be large.
            # opt_info['pos_history'] = optimizer.pos_history # If available and desired
            opt_info['final_swarm_positions_6d'] = optimizer.swarm.position # Store final 6D positions of all particles
            opt_info['initial_guess'] = z_initial_guess # Re-confirm initial guess used

            if np.isfinite(best_cost_pso):
                z_optimal = best_pos_pso; optimal_cost = best_cost_pso
                opt_info['algorithm'] = f'GMM_OPT_PSO_SUCCESS'
                print(f"  PSO finished. Best Cost: {optimal_cost:.4f}")
            else:
                print(f"  PSO finished but found no finite cost solution."); opt_info['algorithm'] = f'GMM_OPT_PSO_NO_FINITE_SOL'

        except Exception as e:
            print(f"ERROR during PSO optimization: {e}"); traceback.print_exc()
            opt_info['algorithm'] = 'GMM_OPT_PSO_ERROR'


    elif config.GMM_OPT_ALGORITHM == "GD":
        print(f"  Running GD for GMM Waypoint Optimization (Branch {branch_tuple})...")
        # --- GD Implementation with History Logging ---
        opt_info['algorithm'] = 'GMM_OPT_GD_PLACEHOLDER' # Update status later
        z_current = np.copy(z_initial_guess)
        opt_info['pos_history'].append(np.copy(z_current)) # Store initial state

        learning_rate = config.GMM_GD_LEARNING_RATE
        tolerance = config.GMM_GD_TOLERANCE
        epsilon = config.GMM_GD_EPSILON # For numerical gradient
        prev_cost = float('inf')

        for iter_num in range(config.GMM_GD_ITERATIONS):
            # Calculate cost at current point
            current_cost_arr = gmm_waypoint_cost_function(z_current.reshape(1, 6), gmm_params, branch_tuple, q_previous_waypoint, planner, H_B_I)
            current_cost = current_cost_arr[0] if len(current_cost_arr) > 0 else float('inf')
            opt_info['cost_history'].append(current_cost)

            if not np.isfinite(current_cost):
                print(f"  GD Iter {iter_num+1}: Cost is infinite. Stopping."); opt_info['algorithm'] = 'GMM_OPT_GD_INF_COST'; break
            if iter_num > 0 and abs(prev_cost - current_cost) < tolerance:
                print(f"  GD Iter {iter_num+1}: Converged. Cost={current_cost:.4f}"); opt_info['algorithm'] = 'GMM_OPT_GD_CONVERGED'; break
            if iter_num > 0 and current_cost > prev_cost: # Cost increased, maybe step back or stop?
                 print(f"  GD Iter {iter_num+1}: Cost increased ({current_cost:.4f} > {prev_cost:.4f}). Stopping.")
                 opt_info['algorithm'] = 'GMM_OPT_GD_COST_INCREASED'
                 # Optionally revert to previous state: z_current = opt_info['pos_history'][-2]
                 break

            prev_cost = current_cost

            # Estimate gradient numerically (finite difference)
            grad = np.zeros_like(z_current)
            for j in range(len(z_current)):
                z_plus = np.copy(z_current); z_plus[j] += epsilon
                z_minus = np.copy(z_current); z_minus[j] -= epsilon
                cost_plus_arr = gmm_waypoint_cost_function(z_plus.reshape(1, 6), gmm_params, branch_tuple, q_previous_waypoint, planner, H_B_I)
                cost_minus_arr = gmm_waypoint_cost_function(z_minus.reshape(1, 6), gmm_params, branch_tuple, q_previous_waypoint, planner, H_B_I)
                cost_plus = cost_plus_arr[0] if len(cost_plus_arr)>0 and np.isfinite(cost_plus_arr[0]) else current_cost # Use current cost if eval fails
                cost_minus = cost_minus_arr[0] if len(cost_minus_arr)>0 and np.isfinite(cost_minus_arr[0]) else current_cost
                grad[j] = (cost_plus - cost_minus) / (2.0 * epsilon)

            if not np.all(np.isfinite(grad)):
                 print(f"  GD Iter {iter_num+1}: Non-finite gradient detected. Stopping."); opt_info['algorithm'] = 'GMM_OPT_GD_BAD_GRAD'; break

            # Update step
            z_current -= learning_rate * grad
            opt_info['pos_history'].append(np.copy(z_current)) # Store state after update

            # print(f"  GD Iter {iter_num+1}: Cost={current_cost:.4f}") # Verbose

        else: # Loop finished without break
             print(f"  GD finished after {config.GMM_GD_ITERATIONS} iterations. Final Cost={current_cost:.4f}")
             opt_info['algorithm'] = 'GMM_OPT_GD_MAX_ITER'

        # Set optimal results based on GD outcome
        if 'CONVERGED' in opt_info['algorithm'] or 'MAX_ITER' in opt_info['algorithm']:
             z_optimal = z_current # Use the last valid state
             optimal_cost = opt_info['cost_history'][-1]
        elif 'COST_INCREASED' in opt_info['algorithm'] and len(opt_info['pos_history']) > 1:
             z_optimal = opt_info['pos_history'][-2] # Use state before cost increase
             optimal_cost = opt_info['cost_history'][-2]
        # Else: Optimization failed, z_optimal remains None, optimal_cost remains inf

    else:
        print(f"ERROR: Unknown GMM optimization algorithm: {config.GMM_OPT_ALGORITHM}")
        opt_info['algorithm'] = 'GMM_OPT_UNKNOWN_ALGO'
        return None, None, float('inf'), opt_info

    # --- Final Step: Get the corresponding q_goal for the optimal z ---
    if z_optimal is not None:
        print(f"  Finding final IK solution for optimal z* (Branch {branch_tuple})...")
        R_I_opt = utils.exp_map_so3(z_optimal[3:])
        T_I_opt = np.identity(4)
        if R_I_opt is not None:
            T_I_opt[:3, :3] = R_I_opt; T_I_opt[:3, 3] = z_optimal[:3]
            T_B_opt = H_B_I @ T_I_opt
            q_optimal_branch, _ = utils.get_ik_solution_for_branch(T_B_opt, branch_tuple, current_config_hint=q_previous_waypoint)
            if q_optimal_branch is None:
                print(f"  Warning: Could not find IK for the optimized z* for branch {branch_tuple}. Opt result invalid.")
                z_optimal = None; optimal_cost = float('inf'); opt_info['algorithm'] += "_FINAL_IK_FAILED"
        else:
            print("  Warning: Could not convert optimal z* orientation back. Opt result invalid.")
            z_optimal = None; optimal_cost = float('inf'); opt_info['algorithm'] += "_FINAL_EXP_MAP_FAILED"
    else:
         q_optimal_branch = None # Ensure q is None if z is None

    # Store final optimal cost regardless of success
    opt_info['final_cost'] = optimal_cost

    return z_optimal, q_optimal_branch, optimal_cost, opt_info


# --- REMOVED: Traditional Optimization Functions ---


print(f"optimization.py loaded (GMM Opt Algo: {config.GMM_OPT_ALGORITHM if config.ENABLE_GMM_OPTIMIZATION else 'DISABLED'}, PSO: {PY_SWARMS_AVAILABLE}, GD: {'Implemented (Basic)'})") # Updated GD status
