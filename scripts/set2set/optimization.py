# optimization.py
import numpy as np
import time
import config # Import config to access parameters
import utils

# --- SciPy Import Check ---
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    print("WARNING: (optimization.py) scipy not found. Install with 'pip install scipy'. Orientation operations will be limited.")
    SCIPY_AVAILABLE = False
    # Define a dummy R class if scipy is not available
    class R:
        @staticmethod
        def from_matrix(matrix): raise ImportError("scipy not available")
        @staticmethod
        def from_quat(quat): raise ImportError("scipy not available")
        def as_quat(self): raise ImportError("scipy not available")
        def as_matrix(self): raise ImportError("scipy not available")

# --- PSO Library Import Check ---
try:
    import pyswarms as ps
    PY_SWARMS_AVAILABLE = True
except ImportError:
    print("WARNING: pyswarms not found. Install with 'pip install pyswarms'. PSO algorithm disabled.")
    PY_SWARMS_AVAILABLE = False

# --- Helper Function: Evaluate Pose (Used by both GD and PSO) ---
def evaluate_target_pose(T_target, q_start, planner, hint_q=None, eval_time_limit=None):
    """
    Helper function: performs IK and Planning for a given target pose.
    Returns the OMPL path cost (or large finite number on failure).
    """
    # Use a default eval time if not provided
    if eval_time_limit is None:
        eval_time_limit = 0.01 # A reasonable default

    q_goal, _, _ = utils.initialize_robot_taskspace(T_target, current_config_hint=hint_q)
    if q_goal is None:
        return None, 1e18 # Return large finite cost if IK fails

    path_np, path_cost = planner.plan(q_start, q_goal, time_limit=eval_time_limit)

    if path_np is None:
        return None, 1e18 # Return large finite cost if Planning fails

    # Ensure cost is finite, return large value otherwise
    if not np.isfinite(path_cost):
        return path_np, 1e18
    else:
        return path_np, path_cost

# --- Quaternion Helper ---
def normalize_quat(q):
    """Normalizes a quaternion [w, x, y, z]."""
    norm = np.linalg.norm(q)
    if norm < 1e-9: # Avoid division by zero
        return np.array([1.0, 0.0, 0.0, 0.0]) # Return identity quaternion
    return q / norm

# --- Fitness Function Wrapper for PSO ---
def pso_fitness_batch(particles, q_start, planner, ik_hint, eval_time_limit):
    """
    Evaluates the fitness (OMPL cost) for a batch of PSO particles.
    Designed for use with PySwarms. Returns large cost on failure.
    """
    n_particles = particles.shape[0]
    costs = np.full(n_particles, 1e18) # Initialize with large cost

    for i in range(n_particles):
        particle = particles[i]
        pos = particle[:3]
        quat_wxyz = normalize_quat(particle[3:]) # Normalize quaternion from particle

        # Construct T_target matrix
        try:
            if not SCIPY_AVAILABLE: raise ImportError("SciPy needed for R.from_quat")
            if np.isclose(np.linalg.norm(quat_wxyz), 1.0):
                 rotation_matrix = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
            else:
                 costs[i] = 1e18; continue # Should not happen, but safeguard
        except (ValueError, ImportError) as e:
            costs[i] = 1e18; continue # Penalize invalid quaternion

        T_target = np.identity(4)
        T_target[:3, :3] = rotation_matrix
        T_target[:3, 3] = pos

        # Evaluate cost using the helper (returns large finite cost on failure)
        _, path_cost = evaluate_target_pose(T_target, q_start, planner,
                                            hint_q=ik_hint,
                                            eval_time_limit=eval_time_limit)
        costs[i] = path_cost
    return costs

# --- PSO Implementation Function ---
def optimize_target_with_pso(initial_T_guess, q_start, planner, set_data):
    """
    Optimizes the target pose using Particle Swarm Optimization (PySwarms).
    Returns: best_T, best_cost, final_swarm_positions
    """
    print("\n" + "-"*40)
    print("--- Starting Particle Swarm Optimization (PSO) ---")

    if not PY_SWARMS_AVAILABLE:
        print("ERROR: PySwarms library not available. Cannot run PSO.")
        return None, float('inf'), None
    if not SCIPY_AVAILABLE:
        print("ERROR: SciPy library not available (needed for rotation conversion). Cannot run PSO.")
        return None, float('inf'), None

    # --- Setup ---
    dimensions = 7 # x, y, z, qw, qx, qy, qz
    pos_bounds = np.array(set_data['pos_bounds'])
    min_bounds = np.concatenate((pos_bounds[:, 0], [-1.0] * 4))
    max_bounds = np.concatenate((pos_bounds[:, 1], [1.0] * 4))
    bounds = (min_bounds, max_bounds)
    print(f"PSO Search Bounds: Pos={np.round(pos_bounds.T, 3)}, Quat=[-1, 1]")

    q_center_hint = set_data.get('q_center')
    initial_q_goal, _, _ = utils.initialize_robot_taskspace(initial_T_guess, current_config_hint=q_center_hint)
    ik_hint = q_center_hint if initial_q_goal is None else initial_q_goal
    if ik_hint is None: print("Warning: No valid IK hint available for PSO fitness.")

    fitness_func = lambda p: pso_fitness_batch(p, q_start=q_start, planner=planner,
                                                ik_hint=ik_hint,
                                                eval_time_limit=config.PSO_EVAL_PLANNING_TIME_LIMIT)

    options = config.PSO_OPTIONS
    n_particles = config.PSO_N_PARTICLES
    iters = config.PSO_ITERATIONS
    print(f"PSO Parameters: Particles={n_particles}, Iterations={iters}, Options={options}")

    # Instantiate PSO Optimizer
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions,
                                        options=options, bounds=bounds,
                                        bh_strategy='periodic',
                                        velocity_clamp=(-0.5, 0.5), # Example clamp
                                        vh_strategy='unmodified',
                                        init_pos=None)

    # --- Run Optimization ---
    print("Running PSO optimization...")
    start_time = time.time()
    best_cost, best_pos_params = optimizer.optimize(fitness_func, iters=iters, verbose=True)
    end_time = time.time()
    print(f"PSO finished in {end_time - start_time:.2f} seconds.")

    # --- Get final swarm positions ---
    final_swarm_positions = optimizer.swarm.position

    # --- Result processing ---
    # Use large finite number check consistent with fitness function
    if not np.isfinite(best_cost) or best_cost >= 1e17 or best_pos_params is None:
        print("PSO did not find a finite/valid solution.")
        return None, float('inf'), final_swarm_positions # Return swarm pos even on failure

    print(f"PSO Best Cost Found: {best_cost:.5f}")
    best_pos_xyz = best_pos_params[:3]
    best_quat_wxyz = normalize_quat(best_pos_params[3:]) # Normalize final best quat
    print(f"PSO Best Params (normalized quat): {np.round(best_quat_wxyz, 4)}")

    try:
        best_R_matrix = R.from_quat(best_quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        best_T_target = np.identity(4)
        best_T_target[:3,:3] = best_R_matrix
        best_T_target[:3, 3] = best_pos_xyz
    except Exception as e:
        print(f"ERROR: Could not convert best PSO parameters back to pose matrix: {e}")
        return None, float('inf'), final_swarm_positions # Return failure

    print("-" * 40)
    return best_T_target, best_cost, final_swarm_positions




# --- Gradient Descent Implementation ---
def refine_target_pose_with_gd(initial_T, initial_cost_from_sampling, q_start, planner, set_data):
    """
    Refines the target pose (position and orientation) using Gradient Descent
    with dimension-specific step sizes. Uses quaternions for orientation.
    (Full implementation from previous response)
    """
    print("\n" + "-"*40)
    print("--- Starting Gradient Descent Refinement (Pos + Quat, Dim-Specific Steps) ---")
    print(f"Cost from Sampling: {initial_cost_from_sampling:.5f}")
    print(f"Max Iterations: {config.GD_ITERATIONS}, Pos Steps: {np.round(config.GD_STEP_SIZES_POS, 4)}, Rot Steps: {np.round(config.GD_STEP_SIZES_ROT, 4)}, Epsilon: {config.GD_EPSILON}")
    print("-" * 40)

    # Initialize return history lists
    pose_history_xyz = []
    quat_history_wxyz = [] # Store quaternion history [w, x, y, z]
    cost_history = []
    gradient_history_7d = [] # Store 7D gradients [gx, gy, gz, gw, gqx, gqy, gqz]

    if not config.ENABLE_GD_REFINEMENT: # Check if GD itself is enabled
        print("GD refinement disabled in config. Returning initial values.")
        pose_history_xyz.append(initial_T[:3, 3])
        try:
             initial_quat = R.from_matrix(initial_T[:3, :3]).as_quat()[[3, 0, 1, 2]] # Convert to w,x,y,z
        except Exception: # Handle case where R is dummy class
             initial_quat = np.array([1.0, 0.0, 0.0, 0.0]) # Default
        quat_history_wxyz.append(initial_quat)
        cost_history.append(initial_cost_from_sampling) # Report the sampling cost if GD disabled
        gradient_history_7d.append(np.zeros(7))
        return initial_T, initial_cost_from_sampling, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d

    if not SCIPY_AVAILABLE:
        print("ERROR: SciPy library not available (needed for rotation conversion). Cannot run GD.")
        # Return initial values, maybe indicate failure more strongly?
        # For now, return initial state as per ENABLE_GD_REFINEMENT check
        pose_history_xyz.append(initial_T[:3, 3])
        quat_history_wxyz.append(np.array([1.0, 0.0, 0.0, 0.0]))
        cost_history.append(initial_cost_from_sampling)
        gradient_history_7d.append(np.zeros(7))
        return initial_T, initial_cost_from_sampling, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d


    # --- Verification Step: Evaluate initial pose with GD's short time limit ---
    initial_q_goal_for_gd, _, _ = utils.initialize_robot_taskspace(initial_T)
    ik_hint_for_gd = set_data.get('q_center') if initial_q_goal_for_gd is None else initial_q_goal_for_gd
    if initial_q_goal_for_gd is None: print("Warning: IK failed for initial GD pose, using set center as hint.")

    print("Verifying initial cost using GD's evaluation method...")
    _, initial_cost_re_evaluated = evaluate_target_pose(initial_T, q_start, planner,
                                                       hint_q=ik_hint_for_gd,
                                                       eval_time_limit=config.GD_EVAL_PLANNING_TIME_LIMIT) # Use specific limit
    print(f"Initial Cost (Re-evaluated with GD eval time limit): {initial_cost_re_evaluated:.5f}")

    large_cost = 1e18 # Use consistent large cost value
    if not np.isfinite(initial_cost_re_evaluated) or initial_cost_re_evaluated >= large_cost:
        print("GD refinement skipped: Initial pose cost is non-finite/very large with GD settings.")
        pose_history_xyz.append(initial_T[:3, 3])
        try: initial_quat = R.from_matrix(initial_T[:3, :3]).as_quat()[[3, 0, 1, 2]]
        except: initial_quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat_history_wxyz.append(initial_quat)
        cost_history.append(initial_cost_re_evaluated) # Store bad cost
        gradient_history_7d.append(np.zeros(7))
        return initial_T, initial_cost_re_evaluated, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d
    # --- End Verification Step ---

    # Initialize GD state
    current_T = np.copy(initial_T)
    current_cost = initial_cost_re_evaluated # Start GD from the re-evaluated cost
    current_pos = current_T[:3, 3]
    try:
        current_quat_wxyz = R.from_matrix(current_T[:3, :3]).as_quat()[[3, 0, 1, 2]] # Scipy quat is [x,y,z,w] -> convert
    except Exception as e:
         print(f"ERROR converting initial rotation to quaternion: {e}. Cannot start GD.")
         pose_history_xyz.append(initial_T[:3, 3])
         quat_history_wxyz.append(np.array([1.0, 0.0, 0.0, 0.0]))
         cost_history.append(initial_cost_re_evaluated)
         gradient_history_7d.append(np.zeros(7))
         return initial_T, initial_cost_re_evaluated, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d

    ik_hint = ik_hint_for_gd # Use the hint determined earlier

    # Initialize history with the starting state
    pose_history_xyz.append(np.copy(current_pos))
    quat_history_wxyz.append(np.copy(current_quat_wxyz))
    cost_history.append(current_cost)
    gradient_history_7d.append(np.zeros(7)) # Placeholder for gradient *leading to* this state (zero)

    # Get bounds from set_data
    pos_bounds = np.array(set_data['pos_bounds'])

    # Track best state found *during GD*
    best_gd_cost = current_cost
    best_gd_T = np.copy(current_T)

    # --- GD Loop ---
    for iteration in range(config.GD_ITERATIONS):
        # print(f"\nGD Iteration {iteration + 1}/{config.GD_ITERATIONS}") # Less verbose output
        # print(f"  Current Pos: {np.round(current_pos, 5)}, Quat (wxyz): {np.round(current_quat_wxyz, 4)}, Cost: {current_cost:.5f}")

        gradient_7d = np.zeros(7) # [gx, gy, gz, gw, gqx, gqy, gqz]
        valid_gradient = True # Flag to track if all gradient components are valid

        # --- Gradient Calculation ---
        try:
             current_R_matrix = R.from_quat(current_quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        except Exception as e:
             print(f"  ERROR: GD Iter {iteration+1}: Could not convert current quaternion to matrix: {e}. Stopping GD.")
             if gradient_history_7d: gradient_history_7d.pop() # Remove last placeholder
             break

        # Position Gradients (x, y, z)
        for i in range(3):
            pos_plus = np.copy(current_pos); pos_plus[i] += config.GD_EPSILON
            T_plus = np.identity(4); T_plus[:3,:3] = current_R_matrix; T_plus[:3, 3] = pos_plus
            _, cost_plus = evaluate_target_pose(T_plus, q_start, planner, hint_q=ik_hint,
                                                eval_time_limit=config.GD_EVAL_PLANNING_TIME_LIMIT)

            if not np.isfinite(cost_plus) or cost_plus >= large_cost:
                # print(f"      Pos Dim {i}: Warning - Planning failed/cost high for positive perturbation.") # Debug
                bound_center_i = np.mean(pos_bounds[i])
                grad_component = np.sign(current_pos[i] - bound_center_i) * 1e6 # Heuristic gradient
            else:
                grad_component = (cost_plus - current_cost) / config.GD_EPSILON

            if not np.isfinite(grad_component):
                # print(f"      Pos Dim {i}: Warning - Non-finite gradient component. Using zero.") # Debug
                gradient_7d[i] = 0.0
                valid_gradient = False
            else: gradient_7d[i] = grad_component

        # Orientation Gradients (w, qx, qy, qz)
        for i in range(4):
            quat_plus_unnormalized = np.copy(current_quat_wxyz)
            quat_plus_unnormalized[i] += config.GD_EPSILON
            quat_plus = normalize_quat(quat_plus_unnormalized)

            try:
                R_plus = R.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
            except ValueError as e:
                # print(f"      Quat Dim {i}: Warning - Invalid quaternion after perturbation? {e}. Using zero gradient.") # Debug
                gradient_7d[3+i] = 0.0
                valid_gradient = False
                continue

            T_plus = np.identity(4); T_plus[:3,:3] = R_plus; T_plus[:3, 3] = current_pos
            _, cost_plus = evaluate_target_pose(T_plus, q_start, planner, hint_q=ik_hint,
                                                eval_time_limit=config.GD_EVAL_PLANNING_TIME_LIMIT)

            if not np.isfinite(cost_plus) or cost_plus >= large_cost:
                # print(f"      Quat Dim {i}: Warning - Planning failed/cost high for positive perturbation.") # Debug
                grad_component = 1e6 # Arbitrary large gradient penalty
            else:
                grad_component = (cost_plus - current_cost) / config.GD_EPSILON

            if not np.isfinite(grad_component):
                # print(f"      Quat Dim {i}: Warning - Non-finite gradient component. Using zero.") # Debug
                gradient_7d[3+i] = 0.0
                valid_gradient = False
            else: gradient_7d[3+i] = grad_component

        # --- Store and Check Gradient ---
        gradient_history_7d[-1] = np.copy(gradient_7d) # Store calculated gradient
        gradient_norm = np.linalg.norm(gradient_7d)

        if not valid_gradient or np.any(np.isnan(gradient_7d)) or gradient_norm < 1e-9:
            # print(f"  GD Iter {iteration+1}: Gradient calculation failed, invalid, or near-zero (Norm: {gradient_norm:.4e}). Stopping GD.") # Debug
            break

        # --- Apply Gradient Clipping ---
        clipped_gradient_7d = np.copy(gradient_7d)
        if config.GD_ENABLE_CLIPPING and gradient_norm > config.GD_MAX_GRAD_NORM:
            scale = config.GD_MAX_GRAD_NORM / gradient_norm
            clipped_gradient_7d = gradient_7d * scale
            # print(f"  GD Iter {iteration+1}: Gradient clipped.") # Debug
        # --- End Clipping ---

        # --- Update Step ---
        grad_pos_effective = clipped_gradient_7d[:3]
        delta_pos = -config.GD_STEP_SIZES_POS * grad_pos_effective
        next_pos = current_pos + delta_pos
        next_pos = np.clip(next_pos, pos_bounds[:, 0], pos_bounds[:, 1]) # Clamp position

        grad_quat_effective = clipped_gradient_7d[3:]
        delta_quat = -config.GD_STEP_SIZES_ROT * grad_quat_effective
        next_quat_unnormalized = current_quat_wxyz + delta_quat
        next_quat_wxyz = normalize_quat(next_quat_unnormalized) # Normalize quaternion

        # --- Evaluate New Pose ---
        try:
            next_R_matrix = R.from_quat(next_quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        except ValueError as e:
            print(f"  Warning: GD Iter {iteration+1}: Invalid quaternion after update? {e}. Stopping GD.")
            break

        next_T = np.identity(4); next_T[:3,:3] = next_R_matrix; next_T[:3, 3] = next_pos
        _, next_cost = evaluate_target_pose(next_T, q_start, planner, hint_q=ik_hint,
                                            eval_time_limit=config.GD_EVAL_PLANNING_TIME_LIMIT)
        # print(f"  GD Iter {iteration+1}: Evaluated Cost at Next Pose: {next_cost:.5f}") # Debug

        # --- Check Improvement ---
        cost_improvement = current_cost - next_cost

        if np.isfinite(next_cost) and next_cost < current_cost and next_cost < large_cost:
            # print(f"  GD Iter {iteration+1}: Cost improved by: {cost_improvement:.5f}") # Debug
            # Update state
            current_pos = next_pos
            current_quat_wxyz = next_quat_wxyz
            current_T = next_T
            current_cost = next_cost

            if current_cost < best_gd_cost: # Update best if improved
                 best_gd_cost = current_cost
                 best_gd_T = np.copy(current_T)

            # Update IK hint
            current_q_goal, _, _ = utils.initialize_robot_taskspace(current_T)
            if current_q_goal is not None: ik_hint = current_q_goal

            # Add history
            pose_history_xyz.append(np.copy(current_pos))
            quat_history_wxyz.append(np.copy(current_quat_wxyz))
            cost_history.append(current_cost)
            gradient_history_7d.append(np.zeros(7)) # Placeholder for next

            # Check convergence
            if cost_improvement < config.GD_TOLERANCE:
                # print(f"  GD Iter {iteration+1}: Convergence detected.") # Debug
                gradient_history_7d.pop()
                break
        else:
             # print(f"  GD Iter {iteration+1}: Cost did not improve or became non-finite/large. Stopping GD.") # Debug
             gradient_history_7d.pop()
             break
    # End of GD loop
    else: # Loop finished naturally
         if gradient_history_7d: gradient_history_7d.pop()
         print(f"\nGD reached maximum iterations ({config.GD_ITERATIONS}).")

    # --- Final GD Results ---
    print("\n" + "-"*40)
    print("--- Gradient Descent Finished ---")
    if len(cost_history) > 1:
        try: final_best_quat = R.from_matrix(best_gd_T[:3,:3]).as_quat()[[3,0,1,2]]
        except: final_best_quat = [np.nan]*4
        print(f"Best GD Pos: {np.round(best_gd_T[:3,3], 5)}, Quat (wxyz): {np.round(final_best_quat, 4)}")
        print(f"Best GD Cost: {best_gd_cost:.5f}")
        print(f"Improvement vs Initial Re-evaluated: {initial_cost_re_evaluated - best_gd_cost:.5f}")
        print(f"Total Iterations performed: {len(cost_history) - 1}")
    else:
        print("GD did not run or failed on the first step.")
        print(f"Initial Re-evaluated Cost: {initial_cost_re_evaluated:.5f}")
    print("-" * 40)

    return best_gd_T, best_gd_cost, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d
# --- End of refine_target_pose_with_gd function ---




# --- Main Optimization Dispatcher ---
def run_optimization(initial_T, initial_cost, q_start, planner, set_data):
    """ Runs the selected optimization algorithm (GD or PSO). """
    algo = config.OPTIMIZATION_ALGORITHM.upper()
    result_info = {'algorithm': algo} # Start building info dict

    if algo == "GD":
        result_info['algorithm'] = 'GD' # Set specific algo
        if not config.ENABLE_GD_REFINEMENT:
             print("GD selected but disabled in config (ENABLE_GD_REFINEMENT=False). Skipping refinement.")
             result_info['algorithm'] = 'GD_DISABLED'
             return initial_T, initial_cost, result_info

        # Call GD
        best_T, best_cost, pose_hist, quat_hist, cost_hist, grad_hist = refine_target_pose_with_gd(
            initial_T, initial_cost, q_start, planner, set_data
        )
        # Store GD history if it ran
        if len(cost_hist) > 1: # Check if more than just initial state exists
            result_info['history'] = (pose_hist, quat_hist, cost_hist, grad_hist)
        else: # GD failed very early or didn't run
            result_info['algorithm'] = 'GD_FAILED_INIT'
            # Return initial state from sampling
            return initial_T, initial_cost, result_info
        return best_T, best_cost, result_info

    elif algo == "PSO":
        result_info['algorithm'] = 'PSO' # Set specific algo
        # Call PSO - now returns 3 values
        best_T, best_cost, final_swarm_pos = optimize_target_with_pso(
            initial_T, q_start, planner, set_data
        )
        # Add swarm position to result info if available
        if final_swarm_pos is not None:
            result_info['final_swarm_positions'] = final_swarm_pos

        if best_T is None: # PSO function returns None on failure
             print("PSO failed to find a solution. Returning initial guess.")
             result_info['algorithm'] = 'PSO_FAILED'
             return initial_T, initial_cost, result_info # Return initial state
        else:
             return best_T, best_cost, result_info # Return PSO result

    else:
        print(f"ERROR: Unknown optimization algorithm '{config.OPTIMIZATION_ALGORITHM}' selected in config. Skipping refinement.")
        result_info['algorithm'] = 'UNKNOWN'
        return initial_T, initial_cost, result_info

# --- Final module print statement ---
print(f"optimization.py loaded (Algorithm selection: {config.OPTIMIZATION_ALGORITHM}, PSO Available: {PY_SWARMS_AVAILABLE}, GD Enabled: {config.ENABLE_GD_REFINEMENT})")