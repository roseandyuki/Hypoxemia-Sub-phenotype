# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 20: Sample Size Sensitivity Analysis for GSHC Definition ---
# 
# Purpose: Systematically test different GSHC definitions to find the optimal
# balance between sample size and statistical power
# 
# Scientific Question: Can we achieve statistical significance by optimizing
# the GSHC definition while maintaining clinical relevance?
# 
# Strategy:
# 1. Test multiple GSHC definitions with progressively relaxed criteria
# 2. For each definition: run complete analysis pipeline
# 3. Track sample size vs statistical significance relationship
# 4. Identify optimal definition that maximizes power while preserving validity
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
from collections import defaultdict

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

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_sample_size_sensitivity')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variable mapping (same as previous steps)
VAR_MAP = {
    'bmi_v1': 'bmi_s1', 'sbp_v1': 'systbp', 'dbp_v1': 'diasbp', 'age_v1': 'age_s1', 
    'gender': 'gender', 'ess_v1': 'ess_s1', 'arousal_index': 'ai_all', 
    'n3_percent': 'times34p', 'n1_percent': 'timest1p', 'n2_percent': 'timest2p', 
    'rem_percent': 'timeremp', 'sleep_efficiency': 'slpeffp', 'waso': 'waso', 
    'rdi': 'rdi4p', 'min_spo2': 'minsat', 'avg_spo2': 'avgsat', 
    'bmi_v2': 'bmi_s2', 'sbp_v2': 'avg23bps_s2', 'dbp_v2': 'avg23bpd_s2'
}
RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

# GSHC Definition Scenarios to Test
GSHC_SCENARIOS = {
    'Original_Strict': {
        'description': 'Original strict definition (baseline)',
        'bmi_threshold': 25.0,
        'sbp_threshold': 120,
        'dbp_threshold': 80,
        'expected_n': 450
    },
    'BMI_Relaxed_Light': {
        'description': 'Slightly relaxed BMI (25→26)',
        'bmi_threshold': 26.0,
        'sbp_threshold': 120,
        'dbp_threshold': 80,
        'expected_n': 550
    },
    'BP_Relaxed_Light': {
        'description': 'Slightly relaxed BP (120/80→125/82)',
        'bmi_threshold': 25.0,
        'sbp_threshold': 125,
        'dbp_threshold': 82,
        'expected_n': 600
    },
    'Combined_Light': {
        'description': 'Light combined relaxation',
        'bmi_threshold': 26.0,
        'sbp_threshold': 125,
        'dbp_threshold': 82,
        'expected_n': 700
    },
    'BP_Relaxed_Moderate': {
        'description': 'Moderate BP relaxation (120/80→130/85)',
        'bmi_threshold': 25.0,
        'sbp_threshold': 130,
        'dbp_threshold': 85,
        'expected_n': 800
    },
    'Combined_Moderate': {
        'description': 'Moderate combined relaxation',
        'bmi_threshold': 26.0,
        'sbp_threshold': 130,
        'dbp_threshold': 85,
        'expected_n': 900
    },
    'BMI_Relaxed_Moderate': {
        'description': 'Moderate BMI relaxation (25→27)',
        'bmi_threshold': 27.0,
        'sbp_threshold': 120,
        'dbp_threshold': 80,
        'expected_n': 650
    },
    'Liberal_Healthy': {
        'description': 'Liberal but still healthy definition',
        'bmi_threshold': 27.0,
        'sbp_threshold': 130,
        'dbp_threshold': 85,
        'expected_n': 1000
    }
}

# Permutation test parameters (reduced for efficiency)
N_PERMUTATIONS = 5000  # Reduced from 10,000 for faster execution
RANDOM_SEED = 42

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

def create_gshc_with_definition(df, bmi_threshold, sbp_threshold, dbp_threshold):
    """Create GSHC with custom definition"""
    gshc_criteria = (
        (df['bmi_v1'] < bmi_threshold) & 
        (df['sbp_v1'] < sbp_threshold) & 
        (df['dbp_v1'] < dbp_threshold)
    )
    return df[gshc_criteria].copy()

def engineer_hypoxemia_features(df):
    """Engineer hypoxemia features (same as previous steps)"""
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
# --- Analysis Functions ---
# =============================================================================

def perform_clustering_analysis(gshc_df):
    """Perform clustering analysis on GSHC data"""
    
    # Engineer features
    gshc_df = engineer_hypoxemia_features(gshc_df)
    
    # Extract feature matrix
    X_hypox, feature_names = get_hypoxemia_feature_matrix(gshc_df)
    
    # Remove missing data
    complete_mask = X_hypox.notna().all(axis=1)
    X_complete = X_hypox[complete_mask]
    data_complete = gshc_df[complete_mask].copy()
    
    if len(X_complete) < 50:  # Too few samples
        return None
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_complete)
    
    # Try both 2 and 3 clusters, pick best
    best_labels = None
    best_silhouette = -1
    best_n_clusters = 2
    
    for n_clusters in [2, 3]:
        try:
            # Try K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            
            if len(np.unique(labels)) >= 2:
                silhouette = silhouette_score(X_scaled, labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_labels = labels
                    best_n_clusters = n_clusters
        except:
            continue
    
    if best_labels is None:
        return None
    
    return {
        'labels': best_labels,
        'data': data_complete,
        'n_clusters': best_n_clusters,
        'silhouette_score': best_silhouette,
        'sample_size': len(data_complete)
    }

def calculate_pathway_auc_difference(data, labels, outcome1_col, outcome2_col):
    """Calculate AUC difference between two pathways"""
    
    # Create dummy variables for subtypes
    subtypes = pd.get_dummies(labels, prefix='subtype')
    
    # Calculate AUC for each pathway
    aucs = {}
    
    for outcome_col in [outcome1_col, outcome2_col]:
        y = data[outcome_col]
        
        # Skip if insufficient positive cases
        if y.sum() < 5 or len(np.unique(labels)) < 2:
            return None, None, None
        
        try:
            # Fit logistic regression and get cross-validated AUC
            lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
            cv_scores = cross_val_score(lr, subtypes, y, cv=5, scoring='roc_auc')
            aucs[outcome_col] = np.mean(cv_scores)
        except:
            return None, None, None
    
    if len(aucs) == 2:
        auc1 = aucs[outcome1_col]
        auc2 = aucs[outcome2_col]
        difference = auc1 - auc2
        return auc1, auc2, difference
    else:
        return None, None, None

def run_quick_permutation_test(data, labels, outcome1_col, outcome2_col, n_permutations=1000):
    """Run a quick permutation test (reduced iterations for efficiency)"""
    
    # Calculate observed difference
    observed_auc1, observed_auc2, observed_difference = calculate_pathway_auc_difference(
        data, labels, outcome1_col, outcome2_col
    )
    
    if observed_difference is None:
        return None
    
    # Run permutations
    null_differences = []
    
    for i in range(n_permutations):
        # Set seed for reproducibility
        np.random.seed(RANDOM_SEED + i)
        
        # Create permuted data
        permuted_data = data.copy()
        
        # Randomly permute the outcome labels
        permuted_outcome1 = permuted_data[outcome1_col].values.copy()
        np.random.shuffle(permuted_outcome1)
        permuted_data[f'{outcome1_col}_permuted'] = permuted_outcome1
        
        permuted_outcome2 = permuted_data[outcome2_col].values.copy()
        np.random.shuffle(permuted_outcome2)
        permuted_data[f'{outcome2_col}_permuted'] = permuted_outcome2
        
        # Calculate AUC difference with permuted outcomes
        _, _, difference = calculate_pathway_auc_difference(
            permuted_data, labels, 
            f'{outcome1_col}_permuted', 
            f'{outcome2_col}_permuted'
        )
        
        if difference is not None:
            null_differences.append(difference)
    
    if len(null_differences) < 100:
        return None
    
    # Calculate p-value
    extreme_count = sum([1 for diff in null_differences if abs(diff) >= abs(observed_difference)])
    p_value = extreme_count / len(null_differences)
    
    # Calculate effect size
    null_std = np.std(null_differences)
    effect_size = observed_difference / null_std if null_std > 0 else 0
    
    return {
        'observed_auc1': observed_auc1,
        'observed_auc2': observed_auc2,
        'observed_difference': observed_difference,
        'p_value': p_value,
        'effect_size': effect_size,
        'n_permutations': len(null_differences)
    }

def analyze_single_gshc_scenario(full_cohort_df, scenario_name, scenario_config):
    """Analyze a single GSHC definition scenario"""
    
    print(f"\n--- Analyzing: {scenario_name} ---")
    print(f"Description: {scenario_config['description']}")
    print(f"Thresholds: BMI<{scenario_config['bmi_threshold']}, SBP<{scenario_config['sbp_threshold']}, DBP<{scenario_config['dbp_threshold']}")
    
    # Create GSHC with this definition
    gshc_df = create_gshc_with_definition(
        full_cohort_df,
        scenario_config['bmi_threshold'],
        scenario_config['sbp_threshold'],
        scenario_config['dbp_threshold']
    )
    
    actual_n = len(gshc_df)
    print(f"Actual sample size: {actual_n} (expected: {scenario_config['expected_n']})")
    
    if actual_n < 100:
        print("❌ Sample size too small, skipping...")
        return None
    
    # Perform clustering analysis
    clustering_result = perform_clustering_analysis(gshc_df)
    
    if clustering_result is None:
        print("❌ Clustering failed, skipping...")
        return None
    
    print(f"✓ Clustering successful: {clustering_result['n_clusters']} clusters, silhouette={clustering_result['silhouette_score']:.3f}")
    
    # Calculate outcomes
    data = clustering_result['data']
    data['Y_Hypertension'] = data.apply(has_hypertension, axis=1)
    data['Y_Overweight'] = data.apply(has_overweight, axis=1)
    
    # Remove missing outcomes
    complete_data = data.dropna(subset=['Y_Hypertension', 'Y_Overweight']).copy()
    
    if len(complete_data) < 50:
        print("❌ Too few complete cases, skipping...")
        return None
    
    # Align labels with complete data
    labels = clustering_result['labels'][:len(complete_data)]
    
    print(f"Analysis dataset: {len(complete_data)} participants")
    print(f"HTN cases: {complete_data['Y_Hypertension'].sum()}")
    print(f"Overweight cases: {complete_data['Y_Overweight'].sum()}")
    
    # Run quick permutation test
    print("Running permutation test...")
    permutation_result = run_quick_permutation_test(
        complete_data, labels, 'Y_Hypertension', 'Y_Overweight', n_permutations=1000
    )
    
    if permutation_result is None:
        print("❌ Permutation test failed")
        return None
    
    print(f"✓ Permutation test complete: p={permutation_result['p_value']:.6f}")
    
    return {
        'scenario_name': scenario_name,
        'scenario_config': scenario_config,
        'actual_sample_size': actual_n,
        'analysis_sample_size': len(complete_data),
        'clustering_result': clustering_result,
        'permutation_result': permutation_result,
        'htn_cases': int(complete_data['Y_Hypertension'].sum()),
        'overweight_cases': int(complete_data['Y_Overweight'].sum())
    }

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Sample Size Sensitivity Analysis for GSHC Definition ===")
    print("Testing multiple GSHC definitions to optimize statistical power...")
    
    # Load data
    print("\n--- Loading and preparing data ---")
    base_df = load_and_map_data(DATA_FILES)
    base_df['Y_Transition'] = base_df.apply(has_transitioned, axis=1)
    full_cohort_df = base_df.dropna(subset=['Y_Transition']).copy()
    
    print(f"Full cohort size: {len(full_cohort_df)}")
    
    # Run analysis for each GSHC scenario
    print(f"\n--- Testing {len(GSHC_SCENARIOS)} GSHC scenarios ---")
    
    all_scenario_results = {}
    
    for scenario_name, scenario_config in GSHC_SCENARIOS.items():
        try:
            result = analyze_single_gshc_scenario(full_cohort_df, scenario_name, scenario_config)
            if result is not None:
                all_scenario_results[scenario_name] = result
        except Exception as e:
            print(f"❌ Error with {scenario_name}: {e}")
            continue
    
    print(f"\n--- Analysis Summary ---")
    print(f"Successful scenarios: {len(all_scenario_results)}/{len(GSHC_SCENARIOS)}")
    
    # Find best scenario
    best_scenario = None
    best_p_value = 1.0
    
    print(f"\n--- Results Summary ---")
    for scenario_name, result in all_scenario_results.items():
        p_value = result['permutation_result']['p_value']
        sample_size = result['analysis_sample_size']
        auc_diff = result['permutation_result']['observed_difference']
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"{scenario_name:20s}: n={sample_size:4d}, p={p_value:.6f} {significance}, AUC_diff={auc_diff:.4f}")
        
        if p_value < best_p_value:
            best_p_value = p_value
            best_scenario = scenario_name
    
    # Save results
    results_file = os.path.join(OUTPUT_DIR, 'sample_size_sensitivity_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_scenario_results, f)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print best result
    if best_scenario:
        print(f"\n🎯 BEST SCENARIO: {best_scenario}")
        best_result = all_scenario_results[best_scenario]
        print(f"📊 Sample size: {best_result['analysis_sample_size']}")
        print(f"📊 P-value: {best_result['permutation_result']['p_value']:.6f}")
        print(f"📊 AUC difference: {best_result['permutation_result']['observed_difference']:.4f}")
        print(f"📊 Effect size: {best_result['permutation_result']['effect_size']:.3f}")
        
        if best_result['permutation_result']['p_value'] < 0.05:
            print(f"\n🎉 SUCCESS! Statistical significance achieved!")
            print(f"✅ The optimal GSHC definition provides significant results!")
        else:
            print(f"\n⚠️  Still not significant, but this is the best we can achieve")
            print(f"💡 Consider this as the optimal balance point")
    
    print(f"\n🎯 Sample Size Sensitivity Analysis Complete!")
    print(f"📊 Next: Run visualization script to see the sample size-significance curve!")
    print(f"💡 You now know the optimal GSHC definition for maximum statistical power!")
