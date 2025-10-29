import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os
import numpy as np
import cv2  # OpenCV for handcrafted features
import math
import argparse  # *** ADDED ***

# --- Check for optional dependencies ---
try:
    from skimage.feature import local_binary_pattern, greycomatrix, greycoprops, hog
    from skimage.color import rgb2gray
    SKIMAGE_AVAILABLE = True
    print("scikit-image found. LBP, GLCM, and HOG features will be enabled.")
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not found. LBP, GLCM, and HOG features will be disabled (vectors of zeros).")

try:
    import pywt
    PYWT_AVAILABLE = True
    print("PyWavelets found. Wavelet features will be enabled.")
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: PyWavelets not found. Wavelet features will be disabled (vectors of zeros).")

# Change directory to the script's location
try:
    script_dir = os.path.dirname(os.path.abspath(_file_))
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")
except NameError:
    script_dir = os.getcwd()
    print(f"Working directory is: {script_dir} (running in interactive environment?)")


# --- *** REMOVED HARDCODED CONFIGURATION *** ---
# DATASET_PATH, OUTFILE_PATH, and CONCAT_EXTRA_FEATURES
# will now be handled by argparse in the main block.
# ------------------------------------------------


HOG_DIM = 6084


# --- Load Pre-trained ConvNeXt model ---
print("Loading ConvNeXt_Base pre-trained model...")
convnext = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
feature_extractor = convnext.features
feature_extractor.eval()


# Check for CUDA availability and move the model to GPU if possible
if torch.cuda.is_available():
    print("Using GPU for deep feature extraction.")
    feature_extractor.cuda()
else:
    print("Using CPU for deep feature extraction.")


# --- Image Transformations ---
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


# *** MODIFIED FUNCTION SIGNATURE ***
def get_vector(image_path, concat_extra=False):
    """
    This function takes the path of an image, applies the necessary transformations,
    and returns a concatenated vector of deep features (ConvNeXt-GAP)
    and (optionally) various handcrafted features.
    
    Args:
        image_path (str): Path to the input image.
        concat_extra (bool): If True, compute and concatenate handcrafted features.
                             If False, return only deep features.
    """
    try:
        # 1. Open the image
        img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB

        # 2. Apply transformations for ConvNeXt
        img_224 = scaler(img)  # Save the 224x224 PIL image for later
        t_img = Variable(normalize(to_tensor(img_224))).unsqueeze(0)

        # 3. Move tensor to GPU if available
        if torch.cuda.is_available():
            t_img = t_img.cuda()

        # 4. Get the convolutional feature map from ConvNeXt
        with torch.no_grad():
            features = feature_extractor(t_img)

        # 5. Convert conv feature map into a 1024-dim Global Average Pooling (GAP) descriptor
        with torch.no_grad():
            gp = features.mean(dim=(2, 3)).squeeze(0)
        deep_features = gp.cpu().data.numpy()

        # *** MODIFIED LOGIC ***
        if not concat_extra:
            return deep_features

        # --- 6. Compute handcrafted features ---
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_224_cv = cv2.cvtColor(np.array(img_224), cv2.COLOR_RGB2BGR)
        gray_224 = cv2.cvtColor(img_224_cv, cv2.COLOR_BGR2GRAY)
        handcrafted = []

        # 6.a Color histograms (48 dims)
        chans = cv2.split(img_cv)
        color_hist_features = []
        for ch in chans:
            hist = cv2.calcHist([ch], [0], None, [16], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-8)
            color_hist_features.append(hist)
        handcrafted.append(np.concatenate(color_hist_features))

        # 6.b Grayscale histogram (32 dims)
        gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        gray_hist = gray_hist / (gray_hist.sum() + 1e-8)
        handcrafted.append(gray_hist)

        # 6.c LBP histogram (59 dims)
        if SKIMAGE_AVAILABLE:
            try:
                lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
                n_bins = int(lbp.max() + 1)
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
                lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)
            except Exception:
                lbp_hist = np.zeros(59)
        else:
            lbp_hist = np.zeros(59)
        handcrafted.append(lbp_hist)

        # 6.d GLCM properties (24 dims)
        if SKIMAGE_AVAILABLE:
            try:
                gray_q = (gray / 32).astype('uint8')
                glcm = greycomatrix(gray_q, distances=[1], angles=[0, math.pi / 4, math.pi / 2, 3 * math.pi / 4], levels=8, symmetric=True, normed=True)
                props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
                glcm_feats = []
                for p in props:
                    v = greycoprops(glcm, p).ravel()
                    glcm_feats.append(v)
                glcm_feats = np.concatenate(glcm_feats)
            except Exception:
                glcm_feats = np.zeros(6 * 4)
        else:
            glcm_feats = np.zeros(6 * 4)
        handcrafted.append(glcm_feats)

        # 6.e Hu moments (7 dims)
        try:
            moments = cv2.moments(gray)
            hu = cv2.HuMoments(moments).flatten()
            for i in range(len(hu)):
                if hu[i] != 0:
                    hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]))
            hu = np.nan_to_num(hu)
        except Exception:
            hu = np.zeros(7)
        handcrafted.append(hu)

        # 6.f HOG (6084 dims)
        if SKIMAGE_AVAILABLE:
            try:
                hog_feat = hog(gray_224, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
                if hog_feat.shape[0] != HOG_DIM:
                    if hog_feat.shape[0] < HOG_DIM:
                        hog_feat = np.pad(hog_feat, (0, HOG_DIM - hog_feat.shape[0]), 'constant')
                    else:
                        hog_feat = hog_feat[:HOG_DIM]
            except Exception as e:
                hog_feat = np.zeros(HOG_DIM)
        else:
            hog_feat = np.zeros(HOG_DIM)
        handcrafted.append(hog_feat)

        # 6.g Wavelet energies (6 dims)
        if PYWT_AVAILABLE:
            try:
                arr = gray.astype(float)
                coeffs = pywt.wavedec2(arr, 'db1', level=2)
                energies = []
                for level in coeffs[1:]:
                    for band in level:
                        energies.append(np.sum(np.abs(band)))
                wave_feats = np.array(energies)
                if wave_feats.size != 6:
                    wave_feats = np.zeros(6)
            except Exception:
                wave_feats = np.zeros(6)
        else:
            wave_feats = np.zeros(6)
        handcrafted.append(wave_feats)

        # concatenate all handcrafted features
        handcrafted_vec = np.concatenate([np.ravel(x) for x in handcrafted])

        # 7. Final feature vector: deep features + handcrafted
        final = np.concatenate([deep_features.ravel(), handcrafted_vec.ravel()])
        return final

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# --------------------------------------------------------------------
# --- *** MODIFIED FUNCTION SIGNATURE *** ---
# --------------------------------------------------------------------
def extract_features(dataset_path, concat_extra=False):
    """
    Recursively iterates through the dataset folders, extracts features 
    for each image, and returns the features and corresponding labels.
    
    Args:
        dataset_path (str): Path to the root dataset directory.
        concat_extra (bool): Passed to get_vector to control feature extraction.
    """
    print("Starting recursive feature extraction...")
    all_features = []
    all_labels = []
    
    # A dictionary to map class names (like 'agricultural') to integers (0, 1, 2...)
    class_to_label = {}
    current_label = 0
    
    # Define valid image file extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # os.walk traverses the directory tree top-down
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        
        # Check if this directory contains any images
        image_files = [f for f in filenames if f.lower().endswith(image_extensions)]

        # If this directory contains image files, process them
        if image_files:
            class_name = os.path.basename(dirpath)

            if class_name not in class_to_label:
                class_to_label[class_name] = current_label
                print(f"Discovered new class: {class_name} (Label: {current_label})")
                current_label += 1
            class_label = class_to_label[class_name]

            print(f"Processing folder: {dirpath} (Class: {class_name})")

            img_count = 0
            for image_name in image_files:
                image_path = os.path.join(dirpath, image_name)

                # *** MODIFIED LOGIC ***
                # Pass the concat_extra flag to get_vector
                feature_vector = get_vector(image_path, concat_extra=concat_extra)

                if feature_vector is not None:
                    all_features.append(feature_vector)
                    all_labels.append(class_label)
                    img_count += 1
            
            if img_count > 0:
                print(f"   ...extracted features from {img_count} images.")

    print("Feature extraction completed.")
    
    if not class_to_label:
        print(f"Error: No images found in any subdirectories of '{dataset_path}'.")
        return np.array([]), np.array([])
        
    print("\nDiscovered classes and total images processed:")
    for name, label in class_to_label.items():
        count = np.sum(np.array(all_labels) == label)
        print(f"  - Class '{name}' (Label {label}): {count} images")

    return np.array(all_features), np.array(all_labels)
# --------------------------------------------------------------------
# --- END MODIFIED FUNCTION ---
# --------------------------------------------------------------------


# *** MODIFIED MAIN BLOCK ***
if __name__ == '__main__':
    
    # 1. Setup argparse
    parser = argparse.ArgumentParser(description="Extract deep and handcrafted features from an image dataset.")
    
    parser.add_argument(
        '--data', 
        type=str, 
        default="AID",
        help="Name of the root dataset directory (e.g., 'AID'). Assumed to be in the same folder as the script."
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default="features_aid.npz",
        help="Name for the output .npz file (e.g., 'features_aid.npz')."
    )
    
    parser.add_argument(
        '--extra',
        action='store_true',
        help="Include handcrafted features (LBP, GLCM, HOG, etc.). If not set, only ConvNeXt features are extracted."
    )
    
    args = parser.parse_args()

    # 2. Resolve paths based on script location
    target_dataset_path = os.path.join(script_dir, args.data)
    target_outfile_path = os.path.join(script_dir, args.output)

    # 3. Print configuration
    print("---")
    print("Feature Extraction Configuration:")
    print(f"  Dataset Directory: {target_dataset_path}")
    print(f"  Output File:       {target_outfile_path}")
    print(f"  Extra Features:    {args.extra}")
    if not args.extra:
            print("  (Note: Only extracting deep features. To add handcrafted features, use the --extra flag)")
    print("---")

    # 4. Run extraction
    if not os.path.isdir(target_dataset_path):
        print(f"Error: Dataset directory not found at '{target_dataset_path}'")
        print(f"Please make sure the '{args.data}' folder is in the same directory as this script.")
    else:
        # Pass the --extra flag to the function
        features, labels = extract_features(target_dataset_path, concat_extra=args.extra)
        
        if features.size > 0 and labels.size > 0:
            print(f"\nTotal features extracted: {features.shape[0]}")
            print(f"Dimension of each feature vector: {features.shape[1]}")
            
            print(f"\nSaving features and labels to '{target_outfile_path}'...")
            # Save to the user-specified output file
            np.savez_compressed(target_outfile_path, features=features, labels=labels)
            print("Done.")
        else:
            print("\nNo features were extracted. Please check the dataset directory and image files.")