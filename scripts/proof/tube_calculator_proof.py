# proof/tube_calculator_proof.py
import numpy as np

def calculate_multidim_tube(list_of_NxD_aligned_trajectories):
    """
    Calculates the min/max envelope (tube) for each dimension across multiple
    aligned NxD trajectories.

    Args:
        list_of_NxD_aligned_trajectories (list of np.ndarray): 
            A list where each element is an (N x D) NumPy array. 
            All arrays MUST be of the same shape (N x D).

    Returns:
        tuple: (T_min_values_NxD, T_max_values_NxD, N_timesteps, D_dimensions)
               T_min_values_NxD: (N x D) NumPy array of minimum values.
               T_max_values_NxD: (N x D) NumPy array of maximum values.
               N_timesteps: Number of time steps (N).
               D_dimensions: Number of dimensions (D).
               Returns (None, None, 0, 0) if input is invalid or empty.
    """
    if not list_of_NxD_aligned_trajectories:
        print("Warning (calculate_multidim_tube): No aligned trajectories provided.")
        return None, None, 0, 0

    # Validate shapes and consistency
    first_traj_shape = None
    valid_trajectories = []

    for i, traj_NxD in enumerate(list_of_NxD_aligned_trajectories):
        if not isinstance(traj_NxD, np.ndarray) or traj_NxD.ndim != 2:
            print(f"Warning (calculate_multidim_tube): Trajectory {i} is not a 2D NumPy array. Skipping.")
            continue
        if traj_NxD.shape[0] == 0 or traj_NxD.shape[1] == 0:
            print(f"Warning (calculate_multidim_tube): Trajectory {i} has zero length or zero dimensions {traj_NxD.shape}. Skipping.")
            continue
            
        if first_traj_shape is None:
            first_traj_shape = traj_NxD.shape
        elif traj_NxD.shape != first_traj_shape:
            print(f"Warning (calculate_multidim_tube): Trajectory {i} has inconsistent shape {traj_NxD.shape} (expected {first_traj_shape}). Skipping.")
            continue
        valid_trajectories.append(traj_NxD)

    if not valid_trajectories:
        print("Warning (calculate_multidim_tube): No valid NxD trajectories remaining after checks.")
        return None, None, 0, 0
    
    # All valid trajectories now have the same shape: first_traj_shape
    N_timesteps = first_traj_shape[0]
    D_dimensions = first_traj_shape[1]

    try:
        # Stack trajectories along a new axis (K x N x D)
        # K is the number of valid trajectories
        stacked_data_KxNxD = np.stack(valid_trajectories, axis=0)
    except ValueError as e:
        print(f"Error (calculate_multidim_tube): Stacking NxD trajectories failed: {e}. Check consistency.")
        return None, None, 0, 0

    # Calculate min and max along the K-axis (axis=0)
    T_min_values_NxD = np.min(stacked_data_KxNxD, axis=0) # Result is (N x D)
    T_max_values_NxD = np.max(stacked_data_KxNxD, axis=0) # Result is (N x D)

    # print(f"Calculated multi-dimensional tube: {N_timesteps} time steps, {D_dimensions} dimensions.") # Less verbose
    return T_min_values_NxD, T_max_values_NxD, N_timesteps, D_dimensions

if __name__ == '__main__':
    print("Testing tube_calculator_proof.py (Multi-Dimensional)...")
    # D=2 dimensions, N=3 timesteps, K=3 trajectories
    traj1_md = np.array([[1, 10], [2, 11], [3, 12]])
    traj2_md = np.array([[0, 12], [3, 9],  [2, 13]])
    traj3_md = np.array([[2, 9],  [1, 10], [4, 11]])
    
    aligned_trajs_md_test = [traj1_md, traj2_md, traj3_md]
    
    T_min_NxD, T_max_NxD, N_out, D_out = calculate_multidim_tube(aligned_trajs_md_test)

    if T_min_NxD is not None:
        print(f"\nTube calculated: N={N_out}, D={D_out}")
        print(f"  T_min_values (NxD):\n{T_min_NxD}")
        # Expected T_min: [[0, 9], [1, 9], [2, 11]]
        assert np.array_equal(T_min_NxD, np.array([[0,9],[1,9],[2,11]]))
        print(f"  T_max_values (NxD):\n{T_max_NxD}")
        # Expected T_max: [[2, 12], [3, 11], [4, 13]]
        assert np.array_equal(T_max_NxD, np.array([[2,12],[3,11],[4,13]]))
        print("Assertions for T_min and T_max passed.")
    else:
        print("\nMulti-dimensional tube calculation failed.")

    print("\nTesting with inconsistent shapes:")
    traj4_md_wrong_N = np.array([[1,1],[2,2]]) # N=2, D=2
    inconsistent_trajs = [traj1_md, traj4_md_wrong_N]
    T_min_NxD_err, _, _, _ = calculate_multidim_tube(inconsistent_trajs)
    assert T_min_NxD_err is None, "Failed to handle inconsistent N"
    print("Correctly handled inconsistent N (returned None).")

    traj5_md_wrong_D = np.array([[1,1,1],[2,2,2],[3,3,3]]) # N=3, D=3
    inconsistent_trajs2 = [traj1_md, traj5_md_wrong_D]
    T_min_NxD_err2, _, _, _ = calculate_multidim_tube(inconsistent_trajs2)
    assert T_min_NxD_err2 is None, "Failed to handle inconsistent D"
    print("Correctly handled inconsistent D (returned None).")
