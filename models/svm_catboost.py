# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import hamming_loss, jaccard_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import time
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import uniform, loguniform
import warnings
from IPython.display import display, Markdown, HTML
import matplotlib.ticker as mtick
from scipy.stats import uniform, loguniform
from matplotlib.colors import LinearSegmentedColormap

# Suppress warnings for cleaner notebook output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Create custom colormap for heatmaps
colors = ["#f7fbff", "#08306b"]  # Light blue to dark blue
cmap_blue = LinearSegmentedColormap.from_list("custom_blue", colors)

# Display title
display(HTML("<h1 style='color:#0066CC;'>Multi-Label Disease Classification with Stacked SVM+CatBoost Model</h1>"))

# 1. Data Loading
def load_data(file_path):
    """
    Load the dataset from a CSV file with visualization of sample data.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    display(Markdown("## 1. Data Loading"))
    print(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        
        # Display sample data
        display(Markdown("### Sample data:"))
        display(df.head())
        
        # Create visualization of dataset size
        plt.figure(figsize=(8, 2))
        plt.barh(['Rows', 'Columns'], [df.shape[0], df.shape[1]], color=['#3498db', '#2ecc71'])
        plt.title('Dataset Dimensions')
        plt.tight_layout()
        plt.show()
        
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# 2. Data Preprocessing
def preprocess_data(df):
    """
    Preprocess the dataset with visualizations of data characteristics.
    
    Args:
        df (pd.DataFrame): The raw dataset.
    
    Returns:
        tuple: Processed features (X) and labels (y).
    """
    display(Markdown("## 2. Data Preprocessing and Exploratory Analysis"))
    
    # Define feature and target columns
    feature_cols = ['DO', 'pH', 'Alkalinity', 'Hardness', 'Nitrite', 'H2S', 
                   'Salinity', 'Ammonia', 'Temperature']
    target_cols = ['WSS', 'AHPND', 'TSV', 'YHV']
    
    # Basic statistics of the dataset
    display(Markdown("### Feature Statistics:"))
    display(df[feature_cols].describe().round(2))
    
    # Visualize feature distributions
    display(Markdown("### Feature Distributions:"))
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#3498db')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Class distribution
    display(Markdown("### Class Distribution:"))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(target_cols):
        value_counts = df[col].value_counts()
        axes[i].pie(value_counts, labels=['Negative (0)', 'Positive (1)'], 
                 autopct='%1.1f%%', colors=['#3498db', '#e74c3c'],
                 textprops={'fontsize': 12})
        axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()
    
    # Check for missing values
    missing_values = df.isnull().sum()
    
    if missing_values.sum() > 0:
        display(Markdown("### Missing Values:"))
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values': missing_values.values,
            'Percentage': (missing_values.values / len(df) * 100).round(2)
        })
        display(missing_df[missing_df['Missing Values'] > 0])
        
        # Visualize missing values
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_df[missing_df['Missing Values'] > 0]['Column'], 
                   y=missing_df[missing_df['Missing Values'] > 0]['Percentage'],
                   color='#e74c3c')
        plt.title('Percentage of Missing Values by Column')
        plt.xticks(rotation=45)
        plt.ylabel('Percentage (%)')
        plt.tight_layout()
        plt.show()
        
        print("Filling missing values with median values")
        df = df.fillna(df.median())
    else:
        print("No missing values detected.")
    
    # Feature correlations
    display(Markdown("### Feature Correlations:"))
    plt.figure(figsize=(12, 10))
    corr_matrix = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
               vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title('Feature Correlation Matrix', fontsize=15)
    plt.tight_layout()
    plt.show()
    
    # Disease co-occurrence
    display(Markdown("### Disease Co-occurrence:"))
    plt.figure(figsize=(10, 8))
    disease_corr = df[target_cols].corr()
    sns.heatmap(disease_corr, annot=True, fmt='.2f', cmap='coolwarm',
               vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title('Disease Co-occurrence Matrix', fontsize=15)
    plt.tight_layout()
    plt.show()
    
    # Extract features and labels
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Visualize features before and after scaling
    display(Markdown("### Feature Scaling Effect:"))
    feature_idx = 0  # First feature for demonstration
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(X[:, feature_idx], kde=True, color='#3498db')
    plt.title(f'{feature_cols[feature_idx]} Before Scaling')
    
    plt.subplot(1, 2, 2)
    sns.histplot(X_scaled[:, feature_idx], kde=True, color='#2ecc71')
    plt.title(f'{feature_cols[feature_idx]} After Scaling')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFeatures shape after preprocessing: {X_scaled.shape}")
    print(f"Labels shape after preprocessing: {y.shape}")
    
    return X_scaled, y, scaler, feature_cols, target_cols

# 3. Hyperparameter Tuning for SVM
def hyperparameter_tune_svm(X_train, y_train, random_state=42):
    """
    Perform hyperparameter tuning for SVM model with visualization of search results.
    
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels (multi-label binary format).
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: Best SVM parameters and best estimator.
    """
    display(Markdown("## 3. Hyperparameter Tuning for SVM"))
    
    # Define hyperparameter grid for SVM
    param_grid = {
        'estimator__C': loguniform(1e-2, 1e2),
        'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'estimator__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'estimator__degree': [2, 3, 4],  # for poly kernel
        'estimator__class_weight': ['balanced', None],
        'estimator__probability': [True],  # needed for stacking
    }
    
    # Create base SVM model
    base_model = SVC(random_state=random_state)
    
    # Wrap with OneVsRestClassifier for multi-label
    ovr_svm = OneVsRestClassifier(base_model)
    
    # Set up RandomizedSearchCV
    print("Starting hyperparameter search for SVM (this may take a while)...")
    
    # Create progress tracker for visual feedback
    start_time = time.time()
    
    search = RandomizedSearchCV(
        estimator=ovr_svm,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings sampled
        cv=3,  # 3-fold cross-validation
        scoring='f1_weighted',
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    # Fit the model - using first target only for tuning to save time
    search.fit(X_train, y_train[:, 0])
    
    end_time = time.time()
    
    print(f"\nHyperparameter tuning for SVM completed in {end_time - start_time:.2f} seconds.")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation score: {search.best_score_:.4f}")
    
    # Visualize hyperparameter search results
    display(Markdown("### Hyperparameter Search Results:"))
    
    # Convert results to DataFrame for visualization
    results_df = pd.DataFrame(search.cv_results_)
    
    # Filter for key parameters
    param_cols = [col for col in results_df.columns if col.startswith('param_estimator__')]
    
    # Prepare data for plotting
    plot_df = pd.DataFrame({
        'Iteration': range(1, len(results_df) + 1),
        'F1 Score': results_df['mean_test_score'],
        'Kernel': results_df['param_estimator__kernel'],
        'C': results_df['param_estimator__C'],
        'Gamma': results_df['param_estimator__gamma'].astype(str)
    })
    
    # Plot score vs iteration
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    sns.lineplot(x='Iteration', y='F1 Score', data=plot_df, marker='o', markersize=8)
    plt.axhline(y=search.best_score_, color='r', linestyle='--', label=f'Best score: {search.best_score_:.4f}')
    plt.title('F1 Score by Search Iteration')
    plt.legend()
    
    # Plot score by kernel
    plt.subplot(2, 2, 2)
    sns.boxplot(x='Kernel', y='F1 Score', data=plot_df)
    plt.title('F1 Score Distribution by Kernel Type')
    
    # Plot score vs C (log scale)
    plt.subplot(2, 2, 3)
    plt.scatter(plot_df['C'], plot_df['F1 Score'], c=plot_df.index, cmap='viridis', s=50)
    plt.xscale('log')
    plt.xlabel('C Parameter (log scale)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. C Parameter')
    plt.colorbar(label='Iteration', ticks=[0, len(plot_df)-1])
    
    # Plot score by gamma
    plt.subplot(2, 2, 4)
    sns.boxplot(x='Gamma', y='F1 Score', data=plot_df)
    plt.title('F1 Score Distribution by Gamma Value')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Extract best parameters
    best_params = {k.replace('estimator__', ''): v for k, v in search.best_params_.items()}
    
    # Display best parameters in a table
    display(Markdown("### Best SVM Hyperparameters:"))
    best_params_df = pd.DataFrame({
        'Parameter': list(best_params.keys()),
        'Value': list(best_params.values())
    })
    display(best_params_df)
    
    return best_params, search.best_estimator_

# 4. Training Stacked SVM+CatBoost Model
def train_stacked_svm_catboost(X_train, X_test, y_train, y_test, target_cols, n_folds=5, random_state=42):
    """
    Train a stacked model with SVM as base learner and CatBoost as meta-learner with
    progress visualization and detailed performance metrics.
    
    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training labels (multi-label binary format).
        y_test (np.ndarray): Testing labels (multi-label binary format).
        target_cols (list): Names of target columns.
        n_folds (int): Number of folds for cross-validation in stacking.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: Trained stacked model and evaluation results.
    """
    display(Markdown("## 4. Training Stacked SVM+CatBoost Model"))
    start_time = time.time()
    
    # Get the best SVM parameters first
    best_svm_params, _ = hyperparameter_tune_svm(X_train, y_train, random_state)
    
    # Create storage for metadata (stacking features)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_targets = y_train.shape[1]
    
    # Arrays to store SVM predictions for training (out-of-fold) and test data
    train_meta_features = np.zeros((n_train, n_targets))
    test_meta_features = np.zeros((n_test, n_targets))
    
    # Initialize the KFold cross-validator
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Initialize dictionary to store base SVM models
    base_models = {}
    
    # Set up progress tracking
    display(Markdown("### Training SVM Base Models:"))
    
    # Display model architecture visualization
    plt.figure(figsize=(12, 6))
    # Create a 2x2 grid for illustration
    plt.subplot(1, 1, 1)
    plt.text(0.5, 0.9, 'Stacked Model Architecture', fontsize=15, ha='center', va='center', fontweight='bold')
    
    # Base models
    plt.text(0.25, 0.7, 'SVM Base Models', fontsize=12, ha='center', va='center', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    # Meta models
    plt.text(0.75, 0.7, 'CatBoost Meta Models', fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    # Connect with arrows
    plt.arrow(0.30, 0.7, 0.35, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Original features
    plt.text(0.2, 0.4, 'Original Features', fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    # Meta features
    plt.text(0.5, 0.4, 'Meta Features\n(Base Model Predictions)', fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    # Final prediction
    plt.text(0.75, 0.4, 'Final Predictions', fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    # Connect with arrows
    plt.arrow(0.25, 0.67, 0, -0.22, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.75, 0.67, 0, -0.22, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.25, 0.4, 0.2, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.55, 0.4, 0.15, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Train base SVM models for each disease target
    print("\nTraining SVM base models for each disease...")
    
    # Create progress bar visualization
    progress_data = []
    
    for i, disease in enumerate(target_cols):
        print(f"Training for disease: {disease}")
        disease_start_time = time.time()
        
        # Initialize temporary arrays for this target
        train_meta_preds = np.zeros(n_train)
        test_meta_preds = np.zeros((n_test, n_folds))
        
        # Create SVM model with best params
        base_model = SVC(
            C=best_svm_params['C'],
            kernel=best_svm_params['kernel'],
            gamma=best_svm_params['gamma'],
            degree=best_svm_params.get('degree', 3),
            class_weight=best_svm_params.get('class_weight', None),
            probability=True,
            random_state=random_state
        )
        
        # Perform k-fold stacking
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            # Split data for this fold
            fold_X_train, fold_X_val = X_train[train_idx], X_train[val_idx]
            fold_y_train = y_train[train_idx, i]
            fold_y_val = y_train[val_idx, i]
            
            # Train SVM on this fold
            base_model.fit(fold_X_train, fold_y_train)
            
            # Get out-of-fold predictions (probabilities)
            val_preds = base_model.predict_proba(fold_X_val)[:, 1]
            train_meta_preds[val_idx] = val_preds
            
            # Calculate fold validation score
            val_pred_binary = (val_preds >= 0.5).astype(int)
            fold_f1 = f1_score(fold_y_val, val_pred_binary, zero_division=0)
            fold_scores.append(fold_f1)
            
            # Get predictions for test set
            test_fold_preds = base_model.predict_proba(X_test)[:, 1]
            test_meta_preds[:, fold] = test_fold_preds
        
        # Store the out-of-fold predictions as meta-features
        train_meta_features[:, i] = train_meta_preds
        
        # Average the test predictions across folds
        test_meta_features[:, i] = np.mean(test_meta_preds, axis=1)
        
        # Train a final SVM model on all training data for this target
        final_model = SVC(
            C=best_svm_params['C'],
            kernel=best_svm_params['kernel'],
            gamma=best_svm_params['gamma'],
            degree=best_svm_params.get('degree', 3),
            class_weight=best_svm_params.get('class_weight', None),
            probability=True,
            random_state=random_state
        )
        final_model.fit(X_train, y_train[:, i])
        
        # Store the trained model
        base_models[disease] = final_model
        
        # Update progress data
        disease_end_time = time.time()
        progress_data.append({
            'Disease': disease,
            'Time (s)': disease_end_time - disease_start_time,
            'Avg F1': np.mean(fold_scores)
        })
    
    # Visualize base model training progress
    progress_df = pd.DataFrame(progress_data)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='Disease', y='Time (s)', data=progress_df, palette='viridis')
    plt.title('SVM Training Time by Disease')
    plt.ylabel('Time (seconds)')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='Disease', y='Avg F1', data=progress_df, palette='viridis')
    plt.title('SVM Cross-Validation F1 by Disease')
    plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.show()
    
    # Now train CatBoost meta-models for each disease using meta-features
    display(Markdown("### Training CatBoost Meta-Models:"))
    meta_models = {}
    stacked_predictions = np.zeros((n_test, n_targets))
    
    # Initialize dictionary to store results
    results = {}
    
    print("\nTraining CatBoost meta-models...")
    catboost_progress = []
    
    for i, disease in enumerate(target_cols):
        meta_start_time = time.time()
        print(f"Training meta-model for disease: {disease}")
        
        # Create CatBoost model (with faster training parameters)
        meta_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            random_seed=random_state,
            verbose=100,  # Report every 100 iterations
            allow_writing_files=False  # Disable writing files
        )
        
        # Create expanded feature set for meta-learner (original features + meta-features)
        X_train_expanded = np.column_stack([X_train, train_meta_features])
        X_test_expanded = np.column_stack([X_test, test_meta_features])
        
        # Train meta-model
        meta_model.fit(X_train_expanded, y_train[:, i])
        
        # Make predictions
        y_pred_proba = meta_model.predict_proba(X_test_expanded)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        stacked_predictions[:, i] = y_pred
        
        # Store meta-model
        meta_models[disease] = meta_model
        
        # Calculate performance metrics
        precision = precision_score(y_test[:, i], y_pred, zero_division=0)
        recall = recall_score(y_test[:, i], y_pred, zero_division=0)
        f1 = f1_score(y_test[:, i], y_pred, zero_division=0)
        
        # Update progress data
        meta_end_time = time.time()
        catboost_progress.append({
            'Disease': disease,
            'Time (s)': meta_end_time - meta_start_time,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
        
        # Store results with confusion matrix
        cm = confusion_matrix(y_test[:, i], y_pred)
        results[disease] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_true': y_test[:, i],
            'y_pred': y_pred,
            'y_prob': y_pred_proba
        }
        
        print(f"{disease}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    # Visualize CatBoost meta-model training results
    catboost_df = pd.DataFrame(catboost_progress)
    
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    sns.barplot(x='Disease', y='Time (s)', data=catboost_df, palette='viridis')
    plt.title('CatBoost Training Time by Disease')
    plt.ylabel('Time (seconds)')
    
    plt.subplot(2, 2, 2)
    sns.barplot(x='Disease', y='F1', data=catboost_df, palette='viridis')
    plt.title('F1 Score by Disease')
    plt.ylabel('F1 Score')
    
    plt.subplot(2, 2, 3)
    sns.barplot(x='Disease', y='Precision', data=catboost_df, palette='viridis')
    plt.title('Precision by Disease')
    plt.ylabel('Precision')
    
    plt.subplot(2, 2, 4)
    sns.barplot(x='Disease', y='Recall', data=catboost_df, palette='viridis')
    plt.title('Recall by Disease')
    plt.ylabel('Recall')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize confusion matrices for each disease
    display(Markdown("### Confusion Matrices by Disease:"))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, disease in enumerate(target_cols):
        cm = results[disease]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_blue, ax=axes[i])
        axes[i].set_title(f'Confusion Matrix: {disease}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        axes[i].set_xticklabels(['Negative', 'Positive'])
        axes[i].set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.show()
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_test, stacked_predictions)
    hamming = hamming_loss(y_test, stacked_predictions)
    jaccard = jaccard_score(y_test, stacked_predictions, average='samples')
    
    display(Markdown("### Overall Model Performance:"))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Jaccard Score: {jaccard:.4f}")
    
    # Visualize overall metrics
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Jaccard Score', '1 - Hamming Loss']
    values = [accuracy, jaccard, 1 - hamming]  # Invert Hamming Loss for visualization
    
    # Create gauge chart-like visualization
    sns.barplot(x=metrics, y=values, palette='viridis')
    plt.ylim(0, 1)
    plt.title('Overall Model Performance Metrics')
    plt.ylabel('Score (higher is better)')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=0.75, color='g', linestyle='--', alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Store overall results
    results['overall'] = {
        'accuracy': accuracy,
        'hamming_loss': hamming,
        'jaccard_score': jaccard,
        'model_name': 'SVM+CatBoost Stack'
    }
    
    # Create a container for the complete stacked model
    stacked_model = {
        'base_models': base_models,
        'meta_models': meta_models,
        'best_svm_params': best_svm_params
    }
    
    # Calculate total runtime
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nStacked model training completed in {total_time:.2f} seconds")
    
    # Visualize feature importance from meta models
    display(Markdown("### Feature Importance Analysis:"))
    
    # Get feature importance from the first meta model as an example
    if hasattr(meta_models[target_cols[0]], 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        
        # Original feature names plus meta-features
        expanded_features = feature_cols + [f"{col}_meta" for col in target_cols]
        
        # Get importances from the first model
        importances = meta_models[target_cols[0]].feature_importances_
        # Get importances from the first model
        importances = meta_models[target_cols[0]].feature_importances_
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': expanded_features[:len(importances)],
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f'Feature Importance for {target_cols[0]} Meta-Model')
        plt.tight_layout()
        plt.show()
        
        # Compare feature importance across all diseases
        plt.figure(figsize=(14, 10))
        
        # Create a DataFrame to store importance values for all diseases
        all_importance = pd.DataFrame({'Feature': expanded_features[:len(importances)]})
        
        for disease in target_cols:
            if hasattr(meta_models[disease], 'feature_importances_'):
                imp = meta_models[disease].feature_importances_
                all_importance[disease] = imp
        
        # Melt the DataFrame for seaborn
        melted_importance = pd.melt(all_importance, id_vars=['Feature'], var_name='Disease', value_name='Importance')
        
        # Plot heatmap of feature importance across diseases
        pivot_importance = melted_importance.pivot(index='Feature', columns='Disease', values='Importance')
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_importance, cmap=cmap_blue, annot=True, fmt='.3f')
        plt.title('Feature Importance Across All Diseases')
        plt.tight_layout()
        plt.show()
    
    return stacked_model, results

# 5. Save Model and Evaluate Performance
def save_and_evaluate_model(stacked_model, results, scaler, target_cols, output_dir='models'):
    """
    Save the trained model and visualize comprehensive performance evaluation.
    
    Args:
        stacked_model (dict): Dictionary containing base and meta models.
        results (dict): Dictionary of evaluation results.
        scaler (StandardScaler): The feature scaler.
        target_cols (list): Names of target columns.
        output_dir (str): Directory to save model files.
    
    Returns:
        None
    """
    display(Markdown("## 5. Model Evaluation and Saving"))
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Save models
    print("Saving trained models...")
    
    # Save base models
    for disease, model in stacked_model['base_models'].items():
        model_path = os.path.join(output_dir, f'svm_base_{disease}.pkl')
        joblib.dump(model, model_path)
    
    # Save meta models
    for disease, model in stacked_model['meta_models'].items():
        model_path = os.path.join(output_dir, f'catboost_meta_{disease}.pkl')
        joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Save model metadata
    metadata = {
        'best_svm_params': stacked_model['best_svm_params'],
        'target_columns': target_cols,
        'results': {k: {k2: v2 for k2, v2 in v.items() if not isinstance(v2, np.ndarray)} 
                   for k, v in results.items() if k != 'confusion_matrix'}
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    
    print(f"All models successfully saved to {output_dir}")
    
    # Create detailed evaluation report
    display(Markdown("### Detailed Model Evaluation"))
    
    # Classification report for each disease
    for disease in target_cols:
        display(Markdown(f"#### Classification Report for {disease}:"))
        y_true = results[disease]['y_true']
        y_pred = results[disease]['y_pred']
        
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        display(report_df.round(3))
    
    # ROC curve and AUC for each disease
    display(Markdown("### ROC Curves and AUC"))
    
    plt.figure(figsize=(12, 8))
    
    for i, disease in enumerate(target_cols):
        y_true = results[disease]['y_true']
        y_prob = results[disease]['y_prob']
        
        # Calculate false positive rate and true positive rate
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{disease} (AUC = {roc_auc:.3f})')
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Precision-Recall curves
    display(Markdown("### Precision-Recall Curves"))
    
    plt.figure(figsize=(12, 8))
    
    for disease in target_cols:
        y_true = results[disease]['y_true']
        y_prob = results[disease]['y_prob']
        
        # Calculate precision and recall
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        # Plot Precision-Recall curve
        plt.plot(recall, precision, lw=2, label=f'{disease} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Compare model performance with baseline
    display(Markdown("### Performance Comparison with Baseline"))
    
    # Prepare data for plotting
    metrics_data = []
    
    for disease in target_cols:
        # Stacked model metrics
        metrics_data.append({
            'Disease': disease,
            'Model': 'Stacked SVM+CatBoost',
            'Metric': 'Precision',
            'Value': results[disease]['precision']
        })
        metrics_data.append({
            'Disease': disease,
            'Model': 'Stacked SVM+CatBoost',
            'Metric': 'Recall',
            'Value': results[disease]['recall']
        })
        metrics_data.append({
            'Disease': disease,
            'Model': 'Stacked SVM+CatBoost',
            'Metric': 'F1 Score',
            'Value': results[disease]['f1']
        })
        
        # Add baseline metrics (assume majority class prediction)
        y_true = results[disease]['y_true']
        majority_class = int(np.mean(y_true) >= 0.5)
        y_pred_baseline = np.full_like(y_true, majority_class)
        
        baseline_precision = precision_score(y_true, y_pred_baseline, zero_division=0)
        baseline_recall = recall_score(y_true, y_pred_baseline, zero_division=0)
        baseline_f1 = f1_score(y_true, y_pred_baseline, zero_division=0)
        
        metrics_data.append({
            'Disease': disease,
            'Model': 'Baseline (Majority)',
            'Metric': 'Precision',
            'Value': baseline_precision
        })
        metrics_data.append({
            'Disease': disease,
            'Model': 'Baseline (Majority)',
            'Metric': 'Recall',
            'Value': baseline_recall
        })
        metrics_data.append({
            'Disease': disease,
            'Model': 'Baseline (Majority)',
            'Metric': 'F1 Score',
            'Value': baseline_f1
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create grouped bar plot
    g = sns.catplot(
        data=metrics_df, kind="bar",
        x="Disease", y="Value", hue="Model",
        col="Metric", col_wrap=3, height=4, aspect=1.2,
        palette=["#3498db", "#e74c3c"],
        legend_out=False
    )
    
    # Customize plot
    g.set_axis_labels("", "Score")
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1))
    
    # Add value labels on the bars
    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    plt.show()
    
    # Create a summary table
    display(Markdown("### Performance Summary"))
    
    summary_data = []
    
    for disease in target_cols:
        summary_data.append({
            'Disease': disease,
            'Precision': results[disease]['precision'],
            'Recall': results[disease]['recall'],
            'F1 Score': results[disease]['f1'],
            'Accuracy': np.mean(results[disease]['y_true'] == results[disease]['y_pred'])
        })
    
    # Add overall metrics
    summary_data.append({
        'Disease': 'Overall',
        'Precision': np.mean([results[d]['precision'] for d in target_cols]),
        'Recall': np.mean([results[d]['recall'] for d in target_cols]),
        'F1 Score': np.mean([results[d]['f1'] for d in target_cols]),
        'Accuracy': results['overall']['accuracy']
    })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create styled summary table
    display(summary_df.style.background_gradient(cmap='Blues', subset=['Precision', 'Recall', 'F1 Score', 'Accuracy'])
            .format({'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1 Score': '{:.3f}', 'Accuracy': '{:.3f}'})
            .set_caption('Model Performance Summary by Disease'))
    
    # Create visual summary
    plt.figure(figsize=(10, 6))
    
    # Create a spider/radar chart for each disease
    from matplotlib.path import Path
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D
    
    def radar_factory(num_vars, frame='circle'):
        # Calculate evenly-spaced axis angles
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        
        # Rotate theta such that the first axis is at the top
        theta += np.pi/2
        
        def draw_poly_patch(ax):
            # Draw polygon connecting the axis lines
            verts = unit_poly_verts(theta)
            return plt.Polygon(verts, closed=True, edgecolor='k', facecolor='none')
        
        def draw_poly_lines(ax):
            # Draw axis lines connecting to the vertices
            for i in range(num_vars):
                ax.plot([0, np.cos(theta[i])], [0, np.sin(theta[i])], 'k-', lw=1)
        
        def unit_poly_verts(theta):
            # Return vertices of polygon for subplot axes
            x0, y0, r = [0.5] * 3
            verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
            return verts
        
        return theta, draw_poly_patch, draw_poly_lines
    
    def radar_plot(ax, theta, data, color, label):
        # Plot the data for a single disease
        values = data[['Precision', 'Recall', 'F1 Score', 'Accuracy']].values.flatten()
        ax.plot(theta, values, color=color, linestyle='-', marker='o', label=label)
        ax.fill(theta, values, facecolor=color, alpha=0.25)
    
    # Set up radar plot
    n_metrics = 4  # Precision, Recall, F1, Accuracy
    theta, draw_poly_patch, draw_poly_lines = radar_factory(n_metrics)
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Draw the background circles and spokes
    ax.grid(True)
    
    # Draw concentric circles
    for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(np.linspace(0, 2*np.pi, 100), [level] * 100, 'k-', alpha=0.1)
        if level < 1.0:
            ax.text(0, level, f'{level:.1f}', ha='right', va='center', alpha=0.7)
    
    # Draw axis lines
    draw_poly_lines(ax)
    
    # Set the labels for each metric
    metric_labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    ax.set_xticks(theta)
    ax.set_xticklabels(metric_labels)
    
    # Remove y-axis labels
    ax.set_yticklabels([])
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Plot each disease with a different color
    colors = plt.cm.viridis(np.linspace(0, 1, len(target_cols)))
    for i, disease in enumerate(target_cols):
        radar_plot(ax, theta, summary_df[summary_df['Disease'] == disease], colors[i], disease)
    
    # Add overall performance
    radar_plot(ax, theta, summary_df[summary_df['Disease'] == 'Overall'], 'r', 'Overall')
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Performance Radar Plot by Disease', size=15)
    plt.tight_layout()
    plt.show()
    
    return

# 6. Prediction Function for New Data
def predict_with_stacked_model(X_new, stacked_model, scaler, target_cols, feature_cols):
    """
    Make predictions on new data using the stacked model.
    
    Args:
        X_new (np.ndarray or pd.DataFrame): New features to predict on.
        stacked_model (dict): Dictionary containing base and meta models.
        scaler (StandardScaler): The feature scaler.
        target_cols (list): Names of target columns.
        feature_cols (list): Names of feature columns.
        
    Returns:
        tuple: Predicted binary labels and prediction probabilities.
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(X_new, pd.DataFrame):
        X_new = X_new[feature_cols].values
    
    # Scale the features
    X_new_scaled = scaler.transform(X_new)
    
    # Get base models and meta models
    base_models = stacked_model['base_models']
    meta_models = stacked_model['meta_models']
    
    # Number of samples and targets
    n_samples = X_new_scaled.shape[0]
    n_targets = len(target_cols)
    
    # Generate base model predictions (meta-features)
    meta_features = np.zeros((n_samples, n_targets))
    
    for i, disease in enumerate(target_cols):
        # Get base model for this disease
        base_model = base_models[disease]
        
        # Get predicted probabilities
        meta_features[:, i] = base_model.predict_proba(X_new_scaled)[:, 1]
    
    # Create expanded feature matrix
    X_new_expanded = np.column_stack([X_new_scaled, meta_features])
    
    # Make final predictions using meta-models
    final_predictions = np.zeros((n_samples, n_targets))
    final_probabilities = np.zeros((n_samples, n_targets))
    
    for i, disease in enumerate(target_cols):
        # Get meta model for this disease
        meta_model = meta_models[disease]
        
        # Get predicted probabilities
        y_proba = meta_model.predict_proba(X_new_expanded)[:, 1]
        final_probabilities[:, i] = y_proba
        
        # Convert to binary predictions
        final_predictions[:, i] = (y_proba >= 0.5).astype(int)
    
    return final_predictions, final_probabilities

# 7. Interactive Prediction Function
def interactive_prediction(stacked_model, scaler, feature_cols, target_cols):
    """
    Interactive function to make predictions on custom inputs with visualization.
    
    Args:
        stacked_model (dict): Dictionary containing base and meta models.
        scaler (StandardScaler): The feature scaler.
        feature_cols (list): Names of feature columns.
        target_cols (list): Names of target columns.
        
    Returns:
        None
    """
    display(Markdown("## 7. Interactive Prediction Tool"))
    
    print("Enter values for each feature:")
    
    # Collect user input for each feature
    input_values = {}
    
    for feature in feature_cols:
        valid_input = False
        while not valid_input:
            try:
                value = float(input(f"{feature}: "))
                input_values[feature] = value
                valid_input = True
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    
    # Create a DataFrame from the input
    input_df = pd.DataFrame([input_values])
    
    # Make prediction
    predictions, probabilities = predict_with_stacked_model(
        input_df, stacked_model, scaler, target_cols, feature_cols
    )
    
    print("\nPrediction Results:")
    
    # Display results in a table
    results_data = []
    
    for i, disease in enumerate(target_cols):
        results_data.append({
            'Disease': disease,
            'Prediction': 'Positive' if predictions[0, i] == 1 else 'Negative',
            'Probability': probabilities[0, i]
        })
    
    results_df = pd.DataFrame(results_data)
    display(results_df)
    
    # Visualize prediction probabilities
    plt.figure(figsize=(10, 6))
    
    # Create probability bars
    bars = plt.barh(
        results_df['Disease'], 
        results_df['Probability'], 
        color=[
            '#e74c3c' if pred == 'Positive' else '#3498db' 
            for pred in results_df['Prediction']
        ]
    )
    
    # Add probability labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}', 
            va='center'
        )
    
    # Add a threshold line
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    
    # Set chart properties
    plt.xlim(0, 1)
    plt.title('Disease Prediction Probabilities')
    plt.xlabel('Probability')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return

# Main function to run the complete workflow
def main():
    """
    Main function to execute the complete workflow of loading data, training models,
    evaluating performance, and making predictions.
    """
    display(Markdown("# Complete Disease Classification Workflow"))
    
    # 1. Load the dataset
    data_path = r'C:\Users\naren\Desktop\4thyear models\Untitled Folder\dataset.csv'
    df = load_data(data_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # 2. Preprocess the data
    X, y, scaler, feature_cols, target_cols = preprocess_data(df)
    
    # 3. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # 4. Train the stacked model
    stacked_model, results = train_stacked_svm_catboost(
        X_train, X_test, y_train, y_test, target_cols
    )
    
    # 5. Save and evaluate the model
    save_and_evaluate_model(stacked_model, results, scaler, target_cols)
    
    # 6. Interactive prediction
    while True:
        answer = input("\nWould you like to make predictions on custom input? (y/n): ")
        if answer.lower() == 'y':
            interactive_prediction(stacked_model, scaler, feature_cols, target_cols)
        else:
            break
    
    print("\nWorkflow completed successfully!")

# Run the main function if script is executed directly
if __name__ == "__main__":
    main()