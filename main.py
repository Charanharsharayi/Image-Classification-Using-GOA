# Make sure to install optuna: pip install optuna
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time
import joblib
from tqdm import tqdm
import optuna
import os
import argparse  # Added for terminal input

# --- Get the base directory of the script ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants for Persistence (Updated for SVM-only) ---
BEST_SOLUTION_FILE = os.path.join(BASE_DIR, "goa_best_solution_aid_svm_v1.npz")
FINAL_MODEL_FILE = os.path.join(BASE_DIR, "final_svm_model_aid_v1.joblib")
CONVERGENCE_PLOT_FILE = os.path.join(BASE_DIR, "goa_convergence_runs_svm_v1.png")


# --- Helper Functions ---
def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def get_fitness(X_train, y_train, X_val, y_val, feature_subset, alpha=0.99):
    num_selected = np.sum(feature_subset)
    total_features = len(feature_subset)

    if num_selected == 0:
        return 1.0

    selected_indices = np.where(feature_subset == 1)[0]
    X_train_subset = X_train[:, selected_indices]
    X_val_subset = X_val[:, selected_indices]

    # Using a simple, fast classifier for fitness evaluation
    dt_fitness = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_fitness.fit(X_train_subset, y_train)
    accuracy = dt_fitness.score(X_val_subset, y_val)
    error_rate = 1 - accuracy

    # Fitness aims to minimize error and number of features
    fitness = alpha * error_rate + (1 - alpha) * (num_selected / total_features)
    return fitness


# --- Grasshopper Optimization Algorithm (GOA) ---
def run_goa_feature_selection(X_train, y_train, X_val, y_val, num_agents=30, max_iter=50):
    num_features = X_train.shape[1]
    c_max, c_min = 1, 0.00004
    positions = np.random.rand(num_agents, num_features)
    target_position = np.zeros(num_features)
    target_fitness = float('inf')
    convergence_curve = []
    stagnation_counter = 0
    stagnation_limit = 15 # Number of iterations without improvement to trigger shake-up

    # Initialize target position and fitness
    for i in range(num_agents):
        binary_position = (sigmoid(positions[i, :]) > 0.5).astype(int)
        current_fitness = get_fitness(X_train, y_train, X_val, y_val, binary_position)
        if current_fitness < target_fitness:
            target_fitness = current_fitness
            target_position = positions[i, :].copy()

    iterator = tqdm(range(max_iter), desc="GOA Iterations", leave=False)
    for t in iterator:
        c = c_max - t * ((c_max - c_min) / max_iter)
        prev_fitness = target_fitness
        new_positions = np.zeros_like(positions)

        for i in range(num_agents):
            S_i = np.zeros(num_features)
            for j in range(num_agents):
                if i != j:
                    dist = np.linalg.norm(positions[i, :] - positions[j, :])
                    r_ij_vec = (positions[j, :] - positions[i, :]) / (dist + 1e-8)
                    # S_i function (social interaction)
                    f, l = 0.5, 1.5
                    s_r = f * np.exp(-dist / l) - np.exp(-dist)
                    S_i += s_r * r_ij_vec

            temp_position = c * S_i + target_position
            new_positions[i, :] = temp_position

        # Apply bounds [0, 1]
        positions = np.clip(new_positions, 0, 1)

        # Update target
        for i in range(num_agents):
            binary_position = (sigmoid(positions[i, :]) > 0.5).astype(int)
            current_fitness = get_fitness(X_train, y_train, X_val, y_val, binary_position)
            if current_fitness < target_fitness:
                target_fitness = current_fitness
                target_position = positions[i, :].copy()

        convergence_curve.append(target_fitness)
        iterator.set_postfix(best_fitness=f"{target_fitness:.6f}")
        
        # --- Stagnation Handling ---
        if target_fitness >= prev_fitness:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter >= stagnation_limit:
            tqdm.write(f"Stagnation detected at iteration {t+1}. Applying population shake-up.")
            # Perturb a portion of the population (e.g., 30%)
            num_to_perturb = int(num_agents * 0.3)
            perturb_indices = np.random.choice(num_agents, num_to_perturb, replace=False)
            for idx in perturb_indices:
                # Don't perturb the best solution
                if not np.array_equal(positions[idx], target_position):
                    noise = np.random.normal(0, 0.1, num_features) # Add small Gaussian noise
                    positions[idx] += noise
            positions = np.clip(positions, 0, 1) # Re-apply bounds
            stagnation_counter = 0 # Reset counter

    best_feature_subset = (sigmoid(target_position) > 0.5).astype(int)
    print("GOA Run Finished.")
    return best_feature_subset, target_position, target_fitness, convergence_curve


# --- Baseline Model (SVM only) ---
def run_baseline_model_svm(X_train, y_train, X_test, y_test, target_names):
    print("\n--- Training Baseline Model (SVM with All Features) ---")
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nBaseline SVM Performance (on Test Set):")
    print(f"Number of features used: {X_train.shape[1]}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Removed run_baseline_model_rf and run_baseline_model_knn as requested


# --- Main Classification Function ---
def run_classification(dataset_path, num_runs=10):
    print(f"Loading dataset from: {dataset_path}")
    
    # *** MODIFICATION START ***
    # Simplified loading process for a single file.
    try:
        # Check if the file exists and is an npz file
        if not (os.path.isfile(dataset_path) and dataset_path.endswith('.npz')):
             print(f"Error: Invalid dataset file '{dataset_path}'. Not a .npz file or not found.")
             return

        data = np.load(dataset_path)
        
        # Check for the required keys within the .npz file
        if 'features' not in data or 'labels' not in data:
            print(f"Error: File {dataset_path} is missing 'features' or 'labels' keys.")
            return
            
        X = data['features']
        y = data['labels']
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    # *** MODIFICATION END ***


    # --- Preprocessing ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    target_names = [str(c) for c in le.classes_] # For classification reports

    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {num_classes} classes.")
    print(f"Original labels: {le.classes_}")
    print(f"Using encoded labels (0 to {num_classes - 1}) for classification.")

    # --- Split ---
    # 80% train_val, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    # From 80% train_val, take 12.5% (which is 10% of total) for validation
    # This results in 70% train, 10% val, 20% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
    )

    print(f"\nData split:")
    print(f"  Training set:    {X_train.shape[0]} samples")
    print(f"  Validation set:  {X_val.shape[0]} samples")
    print(f"  Test set:        {X_test.shape[0]} samples")

    # --- Scale data ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    # We also need a scaler for the entire train_val set for final model training
    X_train_val_scaled = scaler.fit_transform(X_train_val)

    # --- Baseline Model ---
    run_baseline_model_svm(X_train_scaled, y_train, X_test_scaled, y_test, target_names)
    # Removed RF and KNN baselines

    # --- Load Previous Best Solution ---
    overall_best_position = None
    overall_best_fitness = float('inf')
    try:
        data = np.load(BEST_SOLUTION_FILE)
        overall_best_position = data['position']
        overall_best_fitness = data['fitness']
        print(f"\nLoaded previous best fitness: {overall_best_fitness:.6f}")
    except FileNotFoundError:
        print("\nNo previous GOA solution found. Starting fresh.")

    # --- Run GOA ---
    start_time_goa = time.time()
    all_run_curves = []

    run_iterator = tqdm(range(num_runs), desc="Overall GOA Progress")
    for run in run_iterator:
        run_iterator.set_description(f"GOA Run {run + 1}/{num_runs}")
        _, position, fitness, convergence = run_goa_feature_selection(
            X_train_scaled, y_train, X_val_scaled, y_val, num_agents=30, max_iter=60 # 60 iterations per run
        )
        all_run_curves.append(convergence)
        if fitness < overall_best_fitness:
            tqdm.write(f"New best fitness: {fitness:.6f}")
            overall_best_fitness = fitness
            overall_best_position = position

    print(f"\nTotal GOA time: {time.time() - start_time_goa:.2f}s")

    if overall_best_position is not None:
        np.savez_compressed(BEST_SOLUTION_FILE, position=overall_best_position, fitness=overall_best_fitness)
        best_features = (sigmoid(overall_best_position) > 0.5).astype(int)
    else:
        print("GOA failed to find any solution.")
        return

    # --- Train Final Model (SVM with Optuna) ---
    print("\n--- Training Final SVM Classifier ---")
    selected_indices = np.where(best_features == 1)[0]
    
    if len(selected_indices) == 0:
        print("Error: GOA selected 0 features. Cannot proceed.")
        return
        
    X_train_for_tuning = X_train_scaled[:, selected_indices]

    print(f"Selected {len(selected_indices)} / {X.shape[1]} features.")

    # Optuna objective function for SVM
    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'linear'])
        c = trial.suggest_float('C', 1e-2, 1e2, log=True)
        
        params = {'kernel': kernel, 'C': c, 'random_state': 42}
        
        if kernel in ['rbf', 'poly']:
            params['gamma'] = trial.suggest_float('gamma', 1e-4, 1.0, log=True)
        
        if kernel == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
            
        model = SVC(**params)
        
        # Cross-validate on the training data with selected features
        score = cross_val_score(model, X_train_for_tuning, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest Optuna Params for SVM: {best_params}")

    # Prepare final data subsets with selected features
    X_train_val_final = X_train_val_scaled[:, selected_indices]
    X_test_final = X_test_scaled[:, selected_indices]

    # Create and train the final model using best params on the entire train_val set
    final_model_params = best_params.copy()
    final_model_params['random_state'] = 42
    final_model_params['probability'] = True # Good practice, though not strictly needed here
    
    final_model = SVC(**final_model_params)

    final_model.fit(X_train_val_final, y_train_val)
    y_pred_final = final_model.predict(X_test_final)

    print("\nFinal GOA-SVM Model Performance (on Test Set):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
    print(classification_report(y_test, y_pred_final, target_names=target_names, zero_division=0))

    joblib.dump(final_model, FINAL_MODEL_FILE)
    print(f"\nFinal model saved to '{FINAL_MODEL_FILE}'")

    # --- Convergence Plot ---
    plt.figure(figsize=(12, 7))
    for i, curve in enumerate(all_run_curves):
        plt.plot(curve, alpha=0.7, label=f'Run {i+1}')
    if all_run_curves:
        # Plot average convergence
        avg_curve = np.mean(all_run_curves, axis=0)
        plt.plot(avg_curve, color='black', linewidth=2, linestyle='--', label='Average')
    plt.title(f"GOA Convergence Curves ({num_runs} Runs)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Value")
    plt.grid(True)
    plt.legend()
    plt.savefig(CONVERGENCE_PLOT_FILE)
    print(f"\nConvergence plot saved as '{CONVERGENCE_PLOT_FILE}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GOA Feature Selection with SVM Classification.")
    
    # *** MODIFICATION START ***
    # Argument for data path (updated help text)
    parser.add_argument(
        '--data', 
        type=str, 
        default="features_ucl.npz",
        help="Filename of the .npz dataset file. "
             "Must be in the same directory as this script."
    )
    
    # Argument for number of runs
    parser.add_argument(
        '--runs', 
        type=int, 
        default=8,
        help="Number of GOA runs to perform."
    )
    
    args = parser.parse_args()

    RUNS = args.runs
    
    # Resolve the data path
    # The file is assumed to be in the same directory as the script.
    DATA_PATH = os.path.join(BASE_DIR, args.data)

    print(f"Script base directory: {BASE_DIR}")
    print(f"Number of GOA runs: {RUNS}")
    print(f"Target data file: {DATA_PATH}")
    
    # Check if the specific file exists
    if not os.path.isfile(DATA_PATH):
        print(f"Error: Data file not found: {DATA_PATH}")
    else:
        run_classification(DATA_PATH, num_runs=RUNS)
    # *** MODIFICATION END ***