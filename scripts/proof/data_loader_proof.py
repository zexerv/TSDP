# proof/data_loader_proof.py
import os
import pandas as pd
import numpy as np
import config_proof as config # Use the new config

def load_selected_dimensions_from_csv(parent_folder, dimensions_to_use, 
                                      load_all=True, file_list=None):
    """
    Loads specified dimensions from trajectories in CSV files.

    Args:
        parent_folder (str): Path to the folder containing CSV files.
        dimensions_to_use (list of str): List of column names to extract.
        load_all (bool): Whether to load all CSVs in the folder.
        file_list (list, optional): Specific list of filenames to load if load_all is False.
        
    Returns:
        tuple: (list_of_multidim_trajectories, list_of_loaded_filenames)
               Each trajectory is a NumPy array of shape (Nm x D).
               Returns ([], []) if errors occur or no data is loaded.
    """
    all_multidim_trajectories = []
    loaded_filenames = []

    if not os.path.isdir(parent_folder):
        print(f"Error (load_selected_dimensions): Parent folder not found - {parent_folder}")
        return [], []

    if not dimensions_to_use:
        print("Error (load_selected_dimensions): No dimensions specified to load.")
        return [], []

    if load_all:
        try:
            all_files_in_dir = [f for f in os.listdir(parent_folder) if os.path.isfile(os.path.join(parent_folder, f))]
            filenames = [f for f in all_files_in_dir if f.endswith('.csv')]
            try: # Try numerical sort, fallback to alphabetical
                filenames.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError:
                filenames.sort()
            # print(f"Found {len(filenames)} CSV files to load: {filenames}") # Less verbose
            if not filenames:
                print(f"Warning (load_selected_dimensions): No CSV files found in {parent_folder}")
                return [], []
        except FileNotFoundError:
            print(f"Error (load_selected_dimensions): Parent folder not found during listing - {parent_folder}")
            return [], []
    elif file_list:
        filenames = file_list
        # print(f"Loading specified files: {filenames}") # Less verbose
    else:
        print("Error (load_selected_dimensions): No files specified to load (and load_all is False).")
        return [], []

    for filename in filenames:
        file_path = os.path.join(parent_folder, filename)
        try:
            # Read only the necessary columns
            df_full = pd.read_csv(file_path, usecols=dimensions_to_use)
            df_full.columns = [col.strip() for col in df_full.columns] 

            # Ensure all requested dimensions were actually found and in correct order
            # This reorders and checks for missing columns after loading.
            # If usecols already filters, this is more of a sanity check / reordering.
            missing_cols_in_df = [col for col in dimensions_to_use if col not in df_full.columns]
            if missing_cols_in_df:
                print(f"Warning (load_selected_dimensions): File '{filename}' is missing some requested dimensions: {missing_cols_in_df}. Skipping file.")
                continue
            
            # Select and reorder columns to match dimensions_to_use
            df_selected_dims = df_full[dimensions_to_use]
            
            # Convert to numeric, coercing errors. Then drop rows with any NaN in selected dimensions.
            # This ensures all data points are valid across all D dimensions.
            numeric_df = df_selected_dims.apply(pd.to_numeric, errors='coerce').dropna()
            
            if numeric_df.empty:
                print(f"Warning (load_selected_dimensions): File '{filename}' resulted in empty data after NA drop for selected dimensions. Skipping.")
                continue

            trajectory_data_Nd = numeric_df.values # Shape (Nm, D)
            all_multidim_trajectories.append(trajectory_data_Nd)
            loaded_filenames.append(filename)
            # print(f"Successfully loaded {trajectory_data_Nd.shape[0]}x{trajectory_data_Nd.shape[1]} data from {filename}") # Less verbose

        except FileNotFoundError:
            print(f"Warning (load_selected_dimensions): File not found - {file_path}")
        except pd.errors.EmptyDataError:
            print(f"Warning (load_selected_dimensions): File '{filename}' is empty. Skipping.")
        except ValueError as ve: 
            if "is not in list" in str(ve): # More specific error for missing columns in usecols
                 print(f"Warning (load_selected_dimensions): File '{filename}' missing one of columns {dimensions_to_use}. Error: {ve}. Skipping.")
            else:
                 print(f"Error (load_selected_dimensions): ValueError processing {file_path}: {ve}")
        except Exception as e:
            print(f"Error (load_selected_dimensions): loading or processing {file_path}: {e}")

    if not all_multidim_trajectories:
        print(f"Warning (load_selected_dimensions): No valid multi-dimensional trajectory data loaded for dimensions: {dimensions_to_use}.")

    return all_multidim_trajectories, loaded_filenames

if __name__ == '__main__':
    print("Testing data_loader_proof.py (Multi-Dimensional)...")
    
    # Test with dimensions from config
    dims_to_load_test = config.DIMENSIONS_TO_USE
    print(f"Attempting to load dimensions: {dims_to_load_test}")

    trajectories_md, files_md = load_selected_dimensions_from_csv(
        parent_folder=config.PARENT_FOLDER_PATH,
        dimensions_to_use=dims_to_load_test,
        load_all=config.LOAD_ALL_FILES,
        file_list=config.FILE_LIST if not config.LOAD_ALL_FILES else None
    )

    if trajectories_md:
        print(f"\nSuccessfully loaded {len(trajectories_md)} multi-dimensional trajectories.")
        for i, traj_md in enumerate(trajectories_md):
            print(f"  Trajectory {i} (from {files_md[i]}): shape {traj_md.shape}")
            if traj_md.shape[0] > 0:
                print(f"    First point: {traj_md[0,:]}")
    else:
        print(f"\nNo multi-dimensional trajectories were loaded for dimensions: {dims_to_load_test}.")
        print("Please check:")
        print(f"1. `PARENT_FOLDER_PATH` in `config_proof.py`: Currently '{config.PARENT_FOLDER_PATH}'")
        print(f"2. `DIMENSIONS_TO_USE` in `config_proof.py`: Currently '{dims_to_load_test}' (must be columns in CSVs)")
        print(f"3. CSV files exist in the target path and are not empty or malformed.")
