# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 21: Bootstrap Stability Analysis for Optimal GSHC Scenario ---
# 
# Purpose: Perform comprehensive bootstrap stability analysis for the optimal
# GSHC definition (Liberal_Healthy: BMI<27, SBP<130, DBP<85, n=1024)
# 
# Scientific Question: Does the increased sample size (447→1024) improve:
# 1. Individual assignment stability
# 2. Pathway prediction stability  
# 3. Overall clustering robustness
# 
# Method:
# 1. Run bootstrap analysis on optimal scenario (n=1024)
# 2. Compare with original scenario stability (n=447)
# 3. Quantify improvements in all stability metrics
# 4. Validate that significant results are robust and reproducible
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
import pickle
from tqdm import tqdm
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')
plt.style.use('default')

# =============================================================================
# --- Configuration ---
# =============================================================================

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = '.'

DATA_FILES = {
    'shhs1': os.path.join(SCRIPT_DIR, 'shhs1-dataset-0.21.0.csv'),
    'shhs2': os.path.join(SCRIPT_DIR, 'shhs2-dataset-0.21.0.csv')
}

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_bootstrap_optimal_scenario')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load previous results for comparison
ORIGINAL_BOOTSTRAP_DIR = os.path.join(SCRIPT_DIR, 'output_bootstrap_stability')
SENSITIVITY_RESULTS_DIR = os.path.join(SCRIPT_DIR, 'output_sample_size_sensitivity')

# Variable mapping
VAR_MAP = {
    'bmi_v1': 'bmi_s1', 'sbp_v1': 'systbp', 'dbp_v1': 'diasbp', 'age_v1': 'age_s1', 
    'gender': 'gender', 'ess_v1': 'ess_s1', 'arousal_index': 'ai_all', 
    'n3_percent': 'times34p', 'n1_percent': 'timest1p', 'n2_percent': 'timest2p', 
    'rem_percent': 'timeremp', 'sleep_efficiency': 'slpeffp', 'waso': 'waso', 
    'rdi': 'rdi4p', 'min_spo2': 'minsat', 'avg_spo2': 'avgsat', 
    'bmi_v2': 'bmi_s2', 'sbp_v2': 'avg23bps_s2', 'dbp_v2': 'avg23bpd_s2'
}
RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

# Bootstrap parameters
N_BOOTSTRAP = 1000  # Number of bootstrap iterations
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Optimal GSHC definition (from sensitivity analysis)
OPTIMAL_GSHC = {
    'bmi_threshold': 27.0,
    'sbp_threshold': 130,
    'dbp_threshold': 85,
    'name': 'Liberal_Healthy'
}

# =============================================================================
# --- Utility Functions ---
# =============================================================================

def load_and_map_data(filepaths, id_col='nsrrid'):
    """Load and merge SHHS datasets"""
    try:
        df1 = pd.read_csv(filepaths['shhs1'], encoding='ISO-8859-1', low_memory=False)
        df2 = pd.read_csv(filepaths['shhs2'], encoding='ISO-8859-1', low_memory=False)
        merged_df = pd.merge(df1, df2, on=id_col, how='left', suffixes=('', '_dup'))
        return merged_df.rename(columns=RENAME_MAP)
    except Exception as e:
        raise FileNotFoundError(f"Error loading data: {e}")

def has_transitioned(row):
    """Check if participant transitioned to unhealthy state"""
    if pd.isna(row['bmi_v2']) or pd.isna(row['sbp_v2']) or pd.isna(row['dbp_v2']):
        return np.nan
    return 1 if any([row['bmi_v2'] >= 25, row['sbp_v2'] >= 120, row['dbp_v2'] >= 80]) else 0

def has_hypertension(row):
    """Check if participant developed hypertension"""
    if pd.isna(row['sbp_v2']) or pd.isna(row['dbp_v2']):
        return np.nan
    return 1 if (row['sbp_v2'] >= 120 or row['dbp_v2'] >= 80) else 0

def has_overweight(row):
    """Check if participant became overweight"""
    if pd.isna(row['bmi_v2']):
        return np.nan
    return 1 if row['bmi_v2'] >= 25 else 0

def create_optimal_gshc(df):
    """Create GSHC with optimal definition"""
    gshc_criteria = (
        (df['bmi_v1'] < OPTIMAL_GSHC['bmi_threshold']) & 
        (df['sbp_v1'] < OPTIMAL_GSHC['sbp_threshold']) & 
        (df['dbp_v1'] < OPTIMAL_GSHC['dbp_threshold'])
    )
    return df[gshc_criteria].copy()

def engineer_hypoxemia_features(df):
    """Engineer hypoxemia features"""
    df_features = df.copy()
    
    # Basic range and variability
    df_features['spo2_range'] = df_features['avg_spo2'] - df_features['min_spo2']
    
    # Coefficient of variation (proxy for variability)
    df_features['spo2_variability'] = df_features['spo2_range'] / df_features['avg_spo2']
    
    # Hypoxemia burden (composite score)
    df_features['hypoxemia_burden'] = (100 - df_features['min_spo2']) * (1 + df_features['spo2_variability'])
    
    # Desaturation severity index
    df_features['desaturation_severity'] = (100 - df_features['min_spo2']) * np.log1p(df_features['rdi'])
    
    # Relative hypoxemia
    df_features['relative_hypoxemia'] = (df_features['avg_spo2'] - df_features['min_spo2']) / df_features['avg_spo2']
    
    return df_features

def get_hypoxemia_feature_matrix(df):
    """Extract hypoxemia feature matrix"""
    hypox_features = [
        'min_spo2', 'avg_spo2', 'spo2_range', 'spo2_variability',
        'hypoxemia_burden', 'desaturation_severity', 'relative_hypoxemia', 'rdi'
    ]
    
    feature_matrix = df[hypox_features].copy()
    return feature_matrix, hypox_features

# =============================================================================
# --- Bootstrap Analysis Functions ---
# =============================================================================

def perform_single_bootstrap_iteration(original_data, original_labels, iteration):
    """Perform one bootstrap iteration for optimal scenario"""
    
    # Bootstrap sampling (with replacement)
    n_samples = len(original_data)
    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    bootstrap_data = original_data.iloc[bootstrap_indices].copy()
    
    # Engineer features
    bootstrap_data = engineer_hypoxemia_features(bootstrap_data)
    
    # Extract feature matrix
    X_hypox, feature_names = get_hypoxemia_feature_matrix(bootstrap_data)
    
    # Remove missing data
    complete_mask = X_hypox.notna().all(axis=1)
    X_complete = X_hypox[complete_mask]
    data_complete = bootstrap_data[complete_mask].copy()
    
    if len(X_complete) < 50:  # Too few samples
        return None
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_complete)
    
    # Apply clustering (K-means with 2 clusters, consistent with sensitivity analysis)
    try:
        model = KMeans(n_clusters=2, random_state=iteration)
        labels = model.fit_predict(X_scaled)
        
        # Calculate outcomes
        data_complete['Y_Hypertension'] = data_complete.apply(has_hypertension, axis=1)
        data_complete['Y_Overweight'] = data_complete.apply(has_overweight, axis=1)
        data_complete = data_complete.dropna(subset=['Y_Hypertension', 'Y_Overweight'])
        
        if len(data_complete) < 20:
            return None
        
        # Align labels with complete data
        labels_aligned = labels[:len(data_complete)]
        
        # Calculate pathway predictions
        pathway_results = {}
        
        for outcome in ['Y_Hypertension', 'Y_Overweight']:
            y = data_complete[outcome]
            
            if y.sum() >= 10 and len(np.unique(labels_aligned)) >= 2:  # Minimum cases
                try:
                    # Create dummy variables for subtypes
                    subtypes = pd.get_dummies(labels_aligned, prefix='subtype')
                    
                    # Fit logistic regression
                    lr = LogisticRegression(random_state=iteration, class_weight='balanced', max_iter=1000)
                    cv_scores = cross_val_score(lr, subtypes, y, cv=5, scoring='roc_auc')
                    
                    # Fit full model for coefficients
                    lr.fit(subtypes, y)
                    
                    pathway_results[outcome] = {
                        'auc': np.mean(cv_scores),
                        'coefficients': dict(zip(subtypes.columns, lr.coef_[0])),
                        'n_positive': int(y.sum())
                    }
                except:
                    pathway_results[outcome] = None
            else:
                pathway_results[outcome] = None
        
        # Calculate cluster quality
        try:
            silhouette = silhouette_score(X_scaled, labels)
        except:
            silhouette = -1
        
        return {
            'iteration': iteration,
            'labels': labels_aligned,
            'bootstrap_indices': bootstrap_indices,
            'n_clusters': len(np.unique(labels)),
            'silhouette_score': silhouette,
            'pathway_results': pathway_results,
            'sample_size': len(data_complete)
        }
        
    except Exception as e:
        return None

def run_bootstrap_analysis_optimal(original_data, original_labels):
    """Run complete bootstrap analysis for optimal scenario"""
    
    print(f"Running {N_BOOTSTRAP} bootstrap iterations for optimal scenario...")
    print(f"Optimal GSHC definition: BMI<{OPTIMAL_GSHC['bmi_threshold']}, SBP<{OPTIMAL_GSHC['sbp_threshold']}, DBP<{OPTIMAL_GSHC['dbp_threshold']}")
    print(f"Sample size: {len(original_data)}")
    
    bootstrap_results = []
    
    for i in tqdm(range(N_BOOTSTRAP), desc="Bootstrap iterations"):
        result = perform_single_bootstrap_iteration(original_data, original_labels, i)
        if result is not None:
            bootstrap_results.append(result)
    
    print(f"Successful bootstrap iterations: {len(bootstrap_results)}/{N_BOOTSTRAP}")
    
    return bootstrap_results

# =============================================================================
# --- Stability Analysis Functions ---
# =============================================================================

def analyze_clustering_stability_optimal(bootstrap_results):
    """Analyze clustering stability for optimal scenario"""
    
    # Extract cluster counts
    cluster_counts = [result['n_clusters'] for result in bootstrap_results]
    cluster_counter = Counter(cluster_counts)
    
    # Extract silhouette scores
    silhouette_scores = [result['silhouette_score'] for result in bootstrap_results 
                        if result['silhouette_score'] > -1]
    
    stability_metrics = {
        'cluster_count_distribution': dict(cluster_counter),
        'most_common_n_clusters': cluster_counter.most_common(1)[0] if cluster_counter else (0, 0),
        'cluster_stability_rate': cluster_counter.most_common(1)[0][1] / len(bootstrap_results) if cluster_counter else 0,
        'silhouette_mean': np.mean(silhouette_scores) if silhouette_scores else -1,
        'silhouette_std': np.std(silhouette_scores) if silhouette_scores else 0,
        'silhouette_ci': np.percentile(silhouette_scores, [2.5, 97.5]) if silhouette_scores else [-1, -1]
    }
    
    return stability_metrics

def analyze_individual_assignment_stability_optimal(bootstrap_results, original_data, original_labels):
    """Analyze individual assignment stability for optimal scenario"""
    
    n_original = len(original_data)
    assignment_matrix = np.full((n_original, len(bootstrap_results)), -1, dtype=int)
    
    # Fill assignment matrix
    for boot_idx, result in enumerate(bootstrap_results):
        bootstrap_indices = result['bootstrap_indices']
        labels = result['labels']
        
        # Map bootstrap labels back to original indices
        for i, orig_idx in enumerate(bootstrap_indices):
            if i < len(labels):
                assignment_matrix[orig_idx, boot_idx] = labels[i]
    
    # Calculate stability for each individual
    individual_stability = []
    
    for i in range(n_original):
        assignments = assignment_matrix[i, :]
        valid_assignments = assignments[assignments >= 0]
        
        if len(valid_assignments) > 0:
            # Most common assignment
            assignment_counter = Counter(valid_assignments)
            most_common_assignment, count = assignment_counter.most_common(1)[0]
            stability_rate = count / len(valid_assignments)
            
            individual_stability.append({
                'original_index': i,
                'original_label': original_labels[i] if i < len(original_labels) else -1,
                'most_common_bootstrap_label': most_common_assignment,
                'stability_rate': stability_rate,
                'n_bootstrap_appearances': len(valid_assignments)
            })
        else:
            individual_stability.append({
                'original_index': i,
                'original_label': original_labels[i] if i < len(original_labels) else -1,
                'most_common_bootstrap_label': -1,
                'stability_rate': 0,
                'n_bootstrap_appearances': 0
            })
    
    return individual_stability

def analyze_pathway_conclusion_stability_optimal(bootstrap_results):
    """Analyze pathway prediction stability for optimal scenario"""
    
    # Extract pathway results
    htn_aucs = []
    overweight_aucs = []
    htn_stronger_count = 0
    overweight_stronger_count = 0
    valid_comparisons = 0
    
    for result in bootstrap_results:
        pathway_results = result['pathway_results']
        
        htn_result = pathway_results.get('Y_Hypertension')
        overweight_result = pathway_results.get('Y_Overweight')
        
        if htn_result and overweight_result:
            htn_auc = htn_result['auc']
            overweight_auc = overweight_result['auc']
            
            htn_aucs.append(htn_auc)
            overweight_aucs.append(overweight_auc)
            
            # Compare which pathway is stronger
            if htn_auc > overweight_auc:
                htn_stronger_count += 1
            else:
                overweight_stronger_count += 1
            
            valid_comparisons += 1
    
    pathway_stability = {
        'valid_comparisons': valid_comparisons,
        'htn_stronger_rate': htn_stronger_count / valid_comparisons if valid_comparisons > 0 else 0,
        'overweight_stronger_rate': overweight_stronger_count / valid_comparisons if valid_comparisons > 0 else 0,
        'htn_auc_mean': np.mean(htn_aucs) if htn_aucs else 0,
        'htn_auc_std': np.std(htn_aucs) if htn_aucs else 0,
        'htn_auc_ci': np.percentile(htn_aucs, [2.5, 97.5]) if htn_aucs else [0, 0],
        'overweight_auc_mean': np.mean(overweight_aucs) if overweight_aucs else 0,
        'overweight_auc_std': np.std(overweight_aucs) if overweight_aucs else 0,
        'overweight_auc_ci': np.percentile(overweight_aucs, [2.5, 97.5]) if overweight_aucs else [0, 0]
    }
    
    return pathway_stability

def load_original_bootstrap_results():
    """Load original bootstrap results for comparison"""
    
    original_file = os.path.join(ORIGINAL_BOOTSTRAP_DIR, 'bootstrap_stability_results.pkl')
    
    if os.path.exists(original_file):
        with open(original_file, 'rb') as f:
            original_results = pickle.load(f)
        return original_results
    else:
        print("⚠️  Original bootstrap results not found, will skip comparison")
        return None

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Bootstrap Stability Analysis for Optimal GSHC Scenario ===")
    print("Analyzing stability improvements with increased sample size...")
    
    # Load data
    print("\n--- Loading and preparing data ---")
    base_df = load_and_map_data(DATA_FILES)
    base_df['Y_Transition'] = base_df.apply(has_transitioned, axis=1)
    full_cohort_df = base_df.dropna(subset=['Y_Transition']).copy()
    
    # Create optimal GSHC
    optimal_gshc_df = create_optimal_gshc(full_cohort_df)
    print(f"Optimal GSHC size: {len(optimal_gshc_df)}")
    
    # Engineer features and prepare for clustering
    optimal_gshc_df = engineer_hypoxemia_features(optimal_gshc_df)
    X_hypox, feature_names = get_hypoxemia_feature_matrix(optimal_gshc_df)
    
    # Remove missing data
    complete_mask = X_hypox.notna().all(axis=1)
    X_complete = X_hypox[complete_mask]
    gshc_complete = optimal_gshc_df[complete_mask].copy()
    
    print(f"Complete data available for {len(gshc_complete)} participants")
    
    # Perform initial clustering to get original labels
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_complete)
    
    # Use K-means with 2 clusters (consistent with sensitivity analysis)
    kmeans = KMeans(n_clusters=2, random_state=42)
    original_labels = kmeans.fit_predict(X_scaled)
    
    print(f"Original clustering: {len(np.unique(original_labels))} clusters")
    
    # Run bootstrap analysis
    print(f"\n--- Running Bootstrap Analysis for Optimal Scenario ---")
    bootstrap_results = run_bootstrap_analysis_optimal(gshc_complete, original_labels)
    
    if len(bootstrap_results) < 100:
        print(f"⚠️  Warning: Only {len(bootstrap_results)} successful bootstrap iterations")
        print("Results may be less reliable.")
    
    # Analyze stability
    print(f"\n--- Analyzing Stability ---")
    
    # 1. Clustering stability
    clustering_stability = analyze_clustering_stability_optimal(bootstrap_results)
    print(f"Clustering stability: {clustering_stability['cluster_stability_rate']:.1%}")
    
    # 2. Individual assignment stability
    individual_stability = analyze_individual_assignment_stability_optimal(
        bootstrap_results, gshc_complete, original_labels
    )
    mean_individual_stability = np.mean([ind['stability_rate'] for ind in individual_stability])
    print(f"Mean individual assignment stability: {mean_individual_stability:.1%}")
    
    # 3. Pathway conclusion stability
    pathway_stability = analyze_pathway_conclusion_stability_optimal(bootstrap_results)
    print(f"Pathway conclusion stability: {pathway_stability['valid_comparisons']} valid comparisons")
    print(f"HTN pathway preference: {pathway_stability['htn_stronger_rate']:.1%}")
    
    # Load original results for comparison
    print(f"\n--- Loading Original Results for Comparison ---")
    original_bootstrap_results = load_original_bootstrap_results()
    
    # Save all results
    optimal_stability_results = {
        'bootstrap_results': bootstrap_results,
        'clustering_stability': clustering_stability,
        'individual_stability': individual_stability,
        'pathway_stability': pathway_stability,
        'original_comparison': original_bootstrap_results,
        'optimal_gshc_config': OPTIMAL_GSHC,
        'n_bootstrap': N_BOOTSTRAP,
        'n_successful': len(bootstrap_results),
        'sample_size': len(gshc_complete),
        'gshc_data': gshc_complete,
        'original_labels': original_labels
    }
    
    output_file = os.path.join(OUTPUT_DIR, 'bootstrap_stability_optimal_results.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(optimal_stability_results, f)
    
    print(f"\nOptimal scenario bootstrap results saved to: {output_file}")
    
    # Print comparison summary
    print(f"\n--- Stability Comparison Summary ---")
    print(f"📊 Optimal Scenario (n={len(gshc_complete)}):")
    print(f"   Clustering Stability: {clustering_stability['cluster_stability_rate']:.1%}")
    print(f"   Individual Stability: {mean_individual_stability:.1%}")
    print(f"   HTN Pathway Preference: {pathway_stability['htn_stronger_rate']:.1%}")
    
    if original_bootstrap_results:
        orig_clustering = original_bootstrap_results['clustering_stability']['cluster_stability_rate']
        orig_individual = np.mean([ind['stability_rate'] for ind in original_bootstrap_results['individual_stability']])
        orig_pathway = original_bootstrap_results['pathway_stability']['htn_stronger_rate']
        
        print(f"\n📊 Original Scenario (n≈447):")
        print(f"   Clustering Stability: {orig_clustering:.1%}")
        print(f"   Individual Stability: {orig_individual:.1%}")
        print(f"   HTN Pathway Preference: {orig_pathway:.1%}")
        
        print(f"\n📈 Improvements:")
        print(f"   Clustering: {clustering_stability['cluster_stability_rate'] - orig_clustering:+.1%}")
        print(f"   Individual: {mean_individual_stability - orig_individual:+.1%}")
        print(f"   Pathway: {pathway_stability['htn_stronger_rate'] - orig_pathway:+.1%}")
    
    print(f"\n🎯 Bootstrap Stability Analysis for Optimal Scenario Complete!")
    print(f"📊 Next: Run visualization script to see the stability improvements!")
    print(f"💡 Key Question: How much did the larger sample size improve stability?")
