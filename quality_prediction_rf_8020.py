"""
Random Forest with Forward Selection
=====================================
NAČÍTÁ UŽ ROZDĚLENÁ DATA (train_80.csv, test_20.csv)
Forward selection běží POUZE na train datech!
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - ZMĚŇ CESTY PODLE POTŘEBY
# ============================================================================

TRAIN_PATH = "train_80.csv"
TEST_PATH = "test_20.csv"
MAX_FEATURES = 20
CV_FOLDS = 5
N_TRIALS = 10
SAVE_PLOTS = True

# ============================================================================
# LOAD SPLIT DATA
# ============================================================================

def load_split_data(train_path, test_path):
    """Načte už rozdělená data."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("=" * 60)
    print("LOADING PRE-SPLIT DATA")
    print("=" * 60)
    print(f"Train: {len(train_df)} samples")
    print(f"Test:  {len(test_df)} samples")
    
    # Create class labels if not present
    def create_class_label(row):
        if row['Label'] == 'iO':
            return 'ok'
        else:
            return row['Error_Description'].lower()
    
    if 'Class' not in train_df.columns:
        train_df['Class'] = train_df.apply(create_class_label, axis=1)
        test_df['Class'] = test_df.apply(create_class_label, axis=1)
    
    print(f"\nTrain class distribution:")
    print(train_df['Class'].value_counts())
    print(f"\nTest class distribution:")
    print(test_df['Class'].value_counts())
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns 
                    if col not in ['Filename', 'Label', 'Error_Description', 'Class']]
    
    # Prepare X and y
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df['Class'].copy()
    y_test = test_df['Class'].copy()
    
    # Handle missing values
    for X in [X_train, X_test]:
        X.fillna(X.mean(), inplace=True)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(), inplace=True)
    
    print(f"\nFeatures: {len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test, feature_cols


# ============================================================================
# FORWARD SELECTION (on TRAIN only!)
# ============================================================================

def forward_selection(X_train, y_train, model, cv=5, max_features=20, verbose=True, random_state=123):
    """
    Forward Selection na TRAIN datech s Pipeline pro zamezení Data Leakage.
    """
    feature_names = list(X_train.columns)
    selected_features = []
    remaining_features = feature_names.copy()
    selection_history = []
    best_overall_score = -np.inf # Iniciální hodnota pro sledování zlepšení
    
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    if verbose:
        print("\n" + "=" * 60)
        print("FORWARD SELECTION (FIXED - using Pipeline)")
        print("=" * 60)
    
    iteration = 0
    no_improvement_count = 0
    
    while remaining_features and len(selected_features) < max_features:
        iteration += 1
        best_score = -np.inf
        best_feature = None
        
        if verbose:
            print(f"\nIteration {iteration}: Testing {len(remaining_features)} features...")
        
        for feature in remaining_features:
            test_features = selected_features + [feature]
            X_subset = X_train[test_features] # Používáme neškálovaná data
            
            # Vytvoření Pipeline pro zajištění škálování uvnitř CV
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            try:
                # cross_val_score nyní volá fit/transform na Pipeline
                # Toto zajistí, že StandardScaler se fituje POUZE na tréninkovém foldu
                scores = cross_val_score(pipeline, X_subset, y_train, 
                                         cv=cv_splitter, scoring='f1_macro', n_jobs=-1)
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_feature = feature
            except Exception as e:
                # print(f"Error: {e}") 
                continue
        
        if best_feature and best_score > best_overall_score:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_overall_score = best_score
            selection_history.append((best_feature, best_score))
            no_improvement_count = 0
            
            if verbose:
                print(f"  -> Added: {best_feature}")
                print(f"     CV F1: {best_score:.4f}")
        else:
            no_improvement_count += 1
            if verbose:
                print(f"  -> No improvement ({no_improvement_count}/3)")
            if no_improvement_count >= 3:
                break
    
    if verbose:
        print(f"\n>>> Selected {len(selected_features)} features")
    
    return selected_features, selection_history


# ============================================================================
# EVALUATE ON TEST
# ============================================================================

def evaluate_on_test(X_train, X_test, y_train, y_test, selected_features, model):
    """Trénuj na train, evaluuj na test."""
    
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    # Train & predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro'),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro')
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm, y_pred


def run_trials(X_train, X_test, y_train, y_test, selected_features, n_trials=10):
    """Více trialů s různými random states."""
    
    results = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
    all_cms = []
    
    print(f"\nRunning {n_trials} trials on TEST set...")
    print("-" * 40)
    
    for trial in range(n_trials):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=trial * 10,
            n_jobs=-1
        )
        
        metrics, cm, _ = evaluate_on_test(
            X_train, X_test, y_train, y_test, selected_features, model
        )
        
        for k in results:
            results[k].append(metrics[k])
        all_cms.append(cm)
        
        print(f"Trial {trial+1:2d}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    summary = {f'{k}_mean': np.mean(v) for k, v in results.items()}
    summary.update({f'{k}_std': np.std(v) for k, v in results.items()})
    
    return summary, np.mean(all_cms, axis=0), all_cms[-1]


# ============================================================================
# PLOTTING
# ============================================================================

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 14})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_selection_history(history, save_path=None):
    features, scores = zip(*history)
    steps = range(1, len(history) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, scores, 'bo-')
    
    for i, (f, s) in enumerate(zip(features, scores)):
        plt.annotate(f, (steps[i], s), textcoords="offset points",
                     xytext=(0, 10), ha='center', rotation=45, fontsize=8,
                     bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    
    plt.title('Forward Selection History (on TRAIN)', fontsize=14, fontweight='bold')
    plt.xlabel('Step')
    plt.ylabel('CV F1-Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("RANDOM FOREST - NO DATA LEAKAGE")
    print("Forward selection on TRAIN only, evaluation on TEST")
    print("=" * 70)
    
    # 1. Load pre-split data
    X_train, X_test, y_train, y_test, feature_cols = load_split_data(TRAIN_PATH, TEST_PATH)
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    class_names = le.classes_
    
    # 2. Forward selection on TRAIN only
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    selected_features, history = forward_selection(
        X_train, y_train_enc,
        model=rf_model,
        cv=CV_FOLDS,
        max_features=MAX_FEATURES,
        verbose=True
    )
    
    print(f"\n>>> Selected Features ({len(selected_features)}):")
    for i, f in enumerate(selected_features, 1):
        print(f"    {i}. {f}")
    
    if SAVE_PLOTS and history:
        plot_selection_history(history, "forward_selection_rf.png")
    
    # 3. Evaluate on TEST
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    
    summary, avg_cm, last_cm = run_trials(
        X_train, X_test, y_train_enc, y_test_enc, selected_features, N_TRIALS
    )
    
    print("\n" + "-" * 40)
    print(">>> TEST Results:")
    print(f"    Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"    F1-Score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"    Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
    print(f"    Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    
    if SAVE_PLOTS:
        plot_confusion_matrix(avg_cm, class_names, 
                              'RF Confusion Matrix (TEST)', 'confusion_matrix_rf.png')
    
        # Last trial confusion matrix (integer values)
        plt.figure(figsize=(10, 8))
        sns.heatmap(last_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 14})
        plt.title('RF Confusion Matrix (Last Trial)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix_rf_last_trial.png', dpi=150, bbox_inches='tight')
        print("Saved: confusion_matrix_rf_last_trial.png")
        plt.show()

    # 4. Baseline (all features)
    print("\n" + "=" * 70)
    print("BASELINE (all features)")
    print("=" * 70)
    
    summary_all, _, _ = run_trials(
        X_train, X_test, y_train_enc, y_test_enc, feature_cols, N_TRIALS
    )
    
    print(f"\n>>> Baseline ({len(feature_cols)} features):")
    print(f"    Accuracy:  {summary_all['accuracy_mean']:.4f} ± {summary_all['accuracy_std']:.4f}")
    print(f"    F1-Score:  {summary_all['f1_mean']:.4f} ± {summary_all['f1_std']:.4f}")
    print(f"    Precision: {summary_all['precision_mean']:.4f} ± {summary_all['precision_std']:.4f}")
    print(f"    Recall:    {summary_all['recall_mean']:.4f} ± {summary_all['recall_std']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Selected Features: {len(selected_features)} (from {len(feature_cols)})
    
    TEST Performance:
    - Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}
    - F1-Score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}
    - Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}
    - Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}
    
    Baseline (all features):
    - Accuracy:  {summary_all['accuracy_mean']:.4f} ± {summary_all['accuracy_std']:.4f}
    - F1-Score:  {summary_all['f1_mean']:.4f} ± {summary_all['f1_std']:.4f}
    - Precision: {summary_all['precision_mean']:.4f} ± {summary_all['precision_std']:.4f}
    - Recall:    {summary_all['recall_mean']:.4f} ± {summary_all['recall_std']:.4f}
    """)
    
    print("Selection History:")
    for i, (f, s) in enumerate(history, 1):
        print(f"  {i}. {f} -> {s:.4f}")


if __name__ == "__main__":
    main()