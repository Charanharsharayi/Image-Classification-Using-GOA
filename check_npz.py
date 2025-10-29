import numpy as np
import os

# --- IMPORTANT: Make sure this name matches your file EXACTLY ---
filename = 'xray_features.npz'

# 1. Check if the file exists
if not os.path.exists(filename):
    print(f"--- DIAGNOSIS ---")
    print(f"Error: File not found at '{os.path.abspath(filename)}'")
    print("FIX: Make sure your .npz file is in the same folder as this script.")
    
else:
    # 2. Check if the file can be loaded and what keys it has
    try:
        data = np.load(filename)
        print(f"--- DIAGNOSIS ---")
        print(f"Successfully loaded '{filename}'")
        
        # 3. List all arrays found inside
        print(f"Keys (arrays) found inside: {data.files}")

        # 4. Check for the required keys
        if 'features' in data and 'labels' in data:
            print("\nRESULT: SUCCESS! This file looks valid.")
            print(f"  - 'features' shape: {data['features'].shape}")
            print(f"  - 'labels' shape:   {data['labels'].shape}")
            if data['features'].shape[0] == 0:
                print("\nWARNING: The arrays are empty! The feature extraction script found no images.")
                
        else:
            print("\nRESULT: FAILED! The file is 'invalid' for your script.")
            if 'features' not in data:
                print("  - ERROR: The required key 'features' was NOT found.")
            if 'labels' not in data:
                print("  - ERROR: The required key 'labels' was NOT found.")
            print("\nFIX: Re-run your feature extraction script (the *first* script you sent) to create this file correctly.")
            print("     Make sure it saves using: np.savez_compressed(..., features=features, labels=labels)")

    except Exception as e:
        print(f"--- DIAGNOSIS ---")
        print(f"Error loading '{filename}'. The file is likely empty or corrupted.")
        print(f"Python error: {e}")
        print("\nFIX: Delete this .npz file and re-run your feature extraction script.")