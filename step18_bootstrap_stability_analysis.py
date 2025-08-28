# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 18: Bootstrap Stability Analysis for Hypoxemia Sub-phenotypes ---
# 
# Purpose: Perform rigorous internal stability analysis using bootstrap 
# resampling to validate sub-phenotype discoveries
# 
# Scientific Question: Are the discovered hypoxemia sub-phenotypes and their 
# differential pathway predictions robust and reproducible?
# 
# Method: 
# 1. Bootstrap resampling (1000 iterations)
# 2. Repeat clustering + pathway analysis on each bootstrap sample
# 3. Assess stability of: clusters, individual assignments, pathway conclusions
# 4. Generate confidence intervals and stability metrics
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
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
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

INPUT_DIR = os.path.join(SCRIPT_DIR, 'output_hypoxemia_subphenotypes')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_bootstrap_stability')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bootstrap parameters
N_BOOTSTRAP = 1000  # Number of bootstrap iterations
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# --- Utility Functions ---
# =============================================================================

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

def engineer_hypoxemia_features(df):
    """Engineer hypoxemia features (same as step17)"""
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

def perform_single_bootstrap_iteration(original_data, original_labels, best_config, iteration):
    """Perform one bootstrap iteration"""
    
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
    
    if len(X_complete) < 20:  # Too few samples
        return None
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_complete)
    
    # Apply clustering (use same config as original)
    try:
        if 'KMeans' in best_config:
            n_clusters = int(best_config.split('_')[1])
            model = KMeans(n_clusters=n_clusters, random_state=iteration)
            labels = model.fit_predict(X_scaled)
        elif 'GMM' in best_config:
            n_components = int(best_config.split('_')[1])
            model = GaussianMixture(n_components=n_components, random_state=iteration)
            labels = model.fit_predict(X_scaled)
        else:
            return None
        
        # Calculate outcomes
        data_complete['Y_Hypertension'] = data_complete.apply(has_hypertension, axis=1)
        data_complete['Y_Overweight'] = data_complete.apply(has_overweight, axis=1)
        data_complete = data_complete.dropna(subset=['Y_Hypertension', 'Y_Overweight'])
        
        if len(data_complete) < 10:
            return None
        
        # Align labels with complete data
        labels_aligned = labels[:len(data_complete)]
        
        # Calculate pathway predictions
        pathway_results = {}
        
        for outcome in ['Y_Hypertension', 'Y_Overweight']:
            y = data_complete[outcome]
            
            if y.sum() >= 5 and len(np.unique(labels_aligned)) >= 2:  # Minimum cases
                try:
                    # Create dummy variables for subtypes
                    subtypes = pd.get_dummies(labels_aligned, prefix='subtype')
                    
                    # Fit logistic regression
                    lr = LogisticRegression(random_state=iteration, class_weight='balanced')
                    cv_scores = cross_val_score(lr, subtypes, y, cv=3, scoring='roc_auc')
                    
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

def run_bootstrap_analysis(original_data, original_labels, best_config):
    """Run complete bootstrap analysis"""
    
    print(f"Running {N_BOOTSTRAP} bootstrap iterations...")
    print(f"Original sample size: {len(original_data)}")
    print(f"Using clustering config: {best_config}")
    
    bootstrap_results = []
    
    for i in tqdm(range(N_BOOTSTRAP), desc="Bootstrap iterations"):
        result = perform_single_bootstrap_iteration(original_data, original_labels, best_config, i)
        if result is not None:
            bootstrap_results.append(result)
    
    print(f"Successful bootstrap iterations: {len(bootstrap_results)}/{N_BOOTSTRAP}")
    
    return bootstrap_results

# =============================================================================
# --- Stability Analysis Functions ---
# =============================================================================

def analyze_clustering_stability(bootstrap_results):
    """Analyze stability of clustering solutions"""
    
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

def analyze_individual_assignment_stability(bootstrap_results, original_data, original_labels):
    """Analyze stability of individual cluster assignments"""
    
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

def analyze_pathway_conclusion_stability(bootstrap_results):
    """Analyze stability of pathway prediction conclusions"""
    
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

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Bootstrap Stability Analysis for Hypoxemia Sub-phenotypes ===")
    print("Performing rigorous internal validation using bootstrap resampling...")
    
    # Load original results
    results_file = os.path.join(INPUT_DIR, 'hypoxemia_subphenotype_results.pkl')
    
    if not os.path.exists(results_file):
        print(f"❌ Error: Original results file not found at {results_file}")
        print("Please run step17_hypoxemia_subphenotype_discovery.py first!")
        exit(1)
    
    print(f"Loading original results from: {results_file}")
    with open(results_file, 'rb') as f:
        original_results = pickle.load(f)
    
    # Extract original data and results
    original_data = original_results['gshc_data']
    original_labels = original_results['best_labels']
    best_config = original_results['best_config']
    
    print(f"Original analysis found {len(np.unique(original_labels))} clusters using {best_config}")
    
    # Run bootstrap analysis
    print(f"\n--- Running Bootstrap Analysis ---")
    bootstrap_results = run_bootstrap_analysis(original_data, original_labels, best_config)
    
    if len(bootstrap_results) < 100:
        print(f"⚠️  Warning: Only {len(bootstrap_results)} successful bootstrap iterations")
        print("Results may be less reliable. Consider investigating data quality.")
    
    # Analyze stability
    print(f"\n--- Analyzing Stability ---")
    
    # 1. Clustering stability
    clustering_stability = analyze_clustering_stability(bootstrap_results)
    print(f"Clustering stability: {clustering_stability['cluster_stability_rate']:.1%}")
    
    # 2. Individual assignment stability
    individual_stability = analyze_individual_assignment_stability(
        bootstrap_results, original_data, original_labels
    )
    mean_individual_stability = np.mean([ind['stability_rate'] for ind in individual_stability])
    print(f"Mean individual assignment stability: {mean_individual_stability:.1%}")
    
    # 3. Pathway conclusion stability
    pathway_stability = analyze_pathway_conclusion_stability(bootstrap_results)
    print(f"Pathway conclusion stability: {pathway_stability['valid_comparisons']} valid comparisons")
    
    # Save all results
    stability_results = {
        'bootstrap_results': bootstrap_results,
        'clustering_stability': clustering_stability,
        'individual_stability': individual_stability,
        'pathway_stability': pathway_stability,
        'original_results': original_results,
        'n_bootstrap': N_BOOTSTRAP,
        'n_successful': len(bootstrap_results)
    }
    
    output_file = os.path.join(OUTPUT_DIR, 'bootstrap_stability_results.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(stability_results, f)
    
    print(f"\nBootstrap stability results saved to: {output_file}")
    
    # Print summary
    print(f"\n--- Bootstrap Stability Summary ---")
    print(f"📊 Clustering Stability: {clustering_stability['cluster_stability_rate']:.1%}")
    print(f"👤 Individual Assignment Stability: {mean_individual_stability:.1%}")
    print(f"🎯 Pathway Conclusion Stability: {pathway_stability['htn_stronger_rate']:.1%} favor HTN")
    print(f"📈 HTN Prediction AUC: {pathway_stability['htn_auc_mean']:.3f} ± {pathway_stability['htn_auc_std']:.3f}")
    print(f"📈 Overweight Prediction AUC: {pathway_stability['overweight_auc_mean']:.3f} ± {pathway_stability['overweight_auc_std']:.3f}")
    
    print(f"\n🎯 Bootstrap Analysis Complete!")
    print(f"📊 Next: Run visualization script to see detailed stability plots!")
    print(f"💡 Key Question: Are your findings stable across {len(bootstrap_results)} bootstrap samples?")
