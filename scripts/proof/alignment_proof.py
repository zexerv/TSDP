# proof/alignment_proof.py
import numpy as np
import config_proof as config 
import time
from scipy.spatial.distance import euclidean # For MD-DTW point-wise distance

# Conditionally import fastdtw
_SHOULD_USE_FASTDTW_CONFIG = config.USE_FASTDTW_APPROXIMATION
_FASTDTW_IMPORTED_SUCCESSFULLY = False
if _SHOULD_USE_FASTDTW_CONFIG:
    try:
        from fastdtw import fastdtw
        _FASTDTW_IMPORTED_SUCCESSFULLY = True
        # print("Successfully imported fastdtw module for alignment_proof.py.") # Less verbose
    except ImportError:
        print("\n*** WARNING (alignment_proof.py): fastdtw library not found, though config.USE_FASTDTW_APPROXIMATION is True. ***")
        print("*** Please install fastdtw (`pip install fastdtw scipy`). fastDTW will NOT be used. ***\n")
        _FASTDTW_IMPORTED_SUCCESSFULLY = False
# else:
    # print("fastDTW usage is disabled in config_proof.py for alignment_proof.py.") # Less verbose

CAN_USE_FASTDTW_LOGIC = _SHOULD_USE_FASTDTW_CONFIG and _FASTDTW_IMPORTED_SUCCESSFULLY

def md_dtw_align_trajectories(list_of_multidim_trajectories):
    """
    Aligns a list of multi-dimensional (N_m x D) trajectories to a reference 
    trajectory (longest one) using multi-dimensional DTW.
    The distance between D-dimensional points is Euclidean.

    Args:
        list_of_multidim_trajectories (list of np.ndarray): 
            Each element is an (N_m x D) NumPy array.

    Returns:
        tuple: (aligned_trajectories_NxD, reference_index)
               aligned_trajectories_NxD: List of (N_ref x D) NumPy arrays.
               reference_index: Index of the trajectory used as reference.
               Returns ([], -1) if input is invalid.
    """
    n_trajectories = len(list_of_multidim_trajectories)
    if n_trajectories == 0:
        print("Error (md_dtw_align): No trajectories provided for alignment.")
        return [], -1

    # Validate trajectory shapes and get lengths
    lengths = []
    num_dimensions = -1
    valid_trajectories_for_alignment = []

    for i, traj_md in enumerate(list_of_multidim_trajectories):
        if not isinstance(traj_md, np.ndarray) or traj_md.ndim != 2:
            print(f"Warning (md_dtw_align): Trajectory {i} is not a 2D NumPy array. Skipping.")
            continue
        if traj_md.shape[0] == 0: # Empty trajectory (no time steps)
            print(f"Warning (md_dtw_align): Trajectory {i} has 0 time steps. Skipping.")
            continue
        
        if num_dimensions == -1:
            num_dimensions = traj_md.shape[1]
        elif traj_md.shape[1] != num_dimensions:
            print(f"Error (md_dtw_align): Trajectory {i} has {traj_md.shape[1]} dimensions, expected {num_dimensions}. Skipping.")
            continue
        
        lengths.append(traj_md.shape[0])
        valid_trajectories_for_alignment.append(traj_md)

    if not valid_trajectories_for_alignment:
        print("Error (md_dtw_align): No valid trajectories remaining after initial checks.")
        return [], -1
    
    if num_dimensions == 0 : # Should be caught by shape[1] check
        print("Error (md_dtw_align): Trajectories have 0 dimensions.")
        return [], -1

    # Use the new list of valid trajectories
    list_of_multidim_trajectories = valid_trajectories_for_alignment
    n_trajectories = len(list_of_multidim_trajectories)
    lengths = [t.shape[0] for t in list_of_multidim_trajectories] # Re-calculate lengths

    reference_index_in_valid_list = np.argmax(lengths)
    # Find original index if needed, but for processing, use index from valid list
    
    ref_trajectory_NxD = list_of_multidim_trajectories[reference_index_in_valid_list]
    ref_len_N = ref_trajectory_NxD.shape[0]
    
    # print(f"\nUsing trajectory (valid_list_idx={reference_index_in_valid_list}) with shape {ref_trajectory_NxD.shape} as reference for MD-DTW.") # Less verbose

    aligned_trajectories_NxD = []
    dtw_times = []

    for i, current_traj_NxD in enumerate(list_of_multidim_trajectories):
        current_len_Nm = current_traj_NxD.shape[0]
        
        if i == reference_index_in_valid_list:
            aligned_trajectories_NxD.append(ref_trajectory_NxD.copy())
            # print(f"  Trajectory {i} is the reference. Shape: {current_traj_NxD.shape}") # Less verbose
            continue
        
        # print(f"  Aligning trajectory {i} (shape {current_traj_NxD.shape}) to reference (shape {ref_trajectory_NxD.shape})...") # Less verbose
        
        dtw_start_time = time.time()
        distance, path = None, []

        if CAN_USE_FASTDTW_LOGIC:
            # fastdtw uses the 'dist' function to compare elements from the two sequences.
            # Here, elements are D-dimensional vectors (rows of the trajectory arrays).
            # scipy.spatial.distance.euclidean is suitable.
            distance, path = fastdtw(ref_trajectory_NxD, current_traj_NxD, 
                                     radius=config.FASTRADIUS, dist=euclidean)
        else:
            # TODO: Implement manual MD-DTW if fastdtw is not available/desired.
            print("    ERROR (md_dtw_align): Manual MD-DTW not implemented. fastdtw is required or config.USE_FASTDTW_APPROXIMATION must be True.")
            # Fallback: return unaligned or zero-padded to indicate failure for this trajectory
            aligned_trajectories_NxD.append(np.zeros_like(ref_trajectory_NxD)) 
            continue 
            
        dtw_end_time = time.time()
        dtw_times.append(dtw_end_time - dtw_start_time)
        # print(f"    MD-DTW distance: {distance:.4f} (took {dtw_end_time - dtw_start_time:.4f}s)") # Less verbose

        if not path:
            print(f"    Warning (md_dtw_align): DTW returned empty path for trajectory {i}. Filling with zeros.")
            aligned_trajectories_NxD.append(np.zeros_like(ref_trajectory_NxD))
            continue

        # Warp the trajectory (N_m x D) to (N_ref x D)
        warped_traj_NxD = np.zeros_like(ref_trajectory_NxD) # Shape (N_ref, D)
        
        ref_to_current_map = {}
        for ref_idx_p, current_idx_p in path:
            ref_to_current_map[ref_idx_p] = current_idx_p
        
        last_valid_current_idx = 0
        if 0 in ref_to_current_map: 
            last_valid_current_idx = min(ref_to_current_map[0], current_len_Nm -1)

        for ref_idx_warp in range(ref_len_N):
            if ref_idx_warp in ref_to_current_map:
                current_idx_warp = min(ref_to_current_map[ref_idx_warp], current_len_Nm - 1)
                warped_traj_NxD[ref_idx_warp, :] = current_traj_NxD[current_idx_warp, :]
                last_valid_current_idx = current_idx_warp
            else:
                valid_idx_fill = min(last_valid_current_idx, current_len_Nm - 1)
                if valid_idx_fill >=0 : 
                     warped_traj_NxD[ref_idx_warp, :] = current_traj_NxD[valid_idx_fill, :]
        
        aligned_trajectories_NxD.append(warped_traj_NxD)

    # if dtw_times: # Less verbose
        # print(f"  Average MD-DTW time per trajectory: {np.mean(dtw_times):.4f}s")
        
    # The reference_index to return should ideally be the original index from the input list,
    # not reference_index_in_valid_list if some trajectories were skipped.
    # However, if trajectories are skipped, the caller needs to handle the mismatch.
    # For now, returning the index within the list of trajectories *actually processed* for alignment.
    # A more robust solution would map this back or ensure all input trajs are valid.
    # Given the prompt assumes K aligned NxD trajectories as input to the *next* step (tube calc),
    # this alignment module's primary job is to produce that.
    # If the input `list_of_multidim_trajectories` was already the K NxD trajectories,
    # this function would effectively just find the reference and "align" others to it (which would be trivial if all N are same).
    # The current implementation handles variable N_m.

    return aligned_trajectories_NxD, reference_index_in_valid_list # Return index from the (potentially filtered) list

if __name__ == '__main__':
    print("Testing alignment_proof.py (Multi-Dimensional DTW)...")
    
    # Create dummy multi-dimensional trajectories (Nm x D)
    # D = 2
    trajA_md = np.array([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0], [4.0, 13.0], [5.0, 14.0]]) # 5x2
    trajB_md = np.array([[1.2, 10.5], [2.5, 11.3], [3.3, 12.1], [4.8, 13.5]])          # 4x2
    trajC_md = np.array([[0.9, 9.8], [1.8, 10.8], [2.9, 11.9], [3.9, 12.9], [5.1, 14.2], [6.0, 15.0]]) # 6x2
    trajD_md_empty_time = np.array([[],[]]).T # 0x2 (or ensure it's shape (0,D))
    if trajD_md_empty_time.ndim == 1 and trajD_md_empty_time.size == 0: # Handle np.array([]) case
        trajD_md_empty_time = np.empty((0,2))
    trajE_md_wrong_dim = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]) # 2x3

    all_trajs_md_test = [trajA_md, trajB_md, trajC_md, trajD_md_empty_time, trajE_md_wrong_dim, trajA_md] # Add A again

    if not CAN_USE_FASTDTW_LOGIC:
        print("Skipping MD-DTW test as fastdtw is not available/configured.")
    else:
        print("\nOriginal Multi-Dimensional Trajectories (shapes):")
        for i, t_md in enumerate(all_trajs_md_test):
            print(f"  Traj {i}: {t_md.shape}")

        aligned_trajs_md, ref_idx_md = md_dtw_align_trajectories(all_trajs_md_test)

        print("\nAligned Multi-Dimensional Trajectories (shapes):")
        if aligned_trajs_md:
            print(f"Reference trajectory was (valid list) index: {ref_idx_md}")
            for i, t_aligned_md in enumerate(aligned_trajs_md):
                print(f"  Aligned Traj {i}: shape {t_aligned_md.shape}")
                if t_aligned_md.shape[0] > 0:
                    print(f"    First point: {t_aligned_md[0,:]}")
        else:
            print("Multi-dimensional alignment failed.")

