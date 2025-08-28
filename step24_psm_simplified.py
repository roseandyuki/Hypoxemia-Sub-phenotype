# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 24: Simplified Propensity Score Matching Analysis ---
# 
# Purpose: Simplified PSM analysis without problematic dependencies
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle
from scipy import stats

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

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_psm_simplified')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Optimal GSHC definition
OPTIMAL_GSHC = {
    'bmi_threshold': 27.0,
    'sbp_threshold': 130,
    'dbp_threshold': 85,
    'name': 'Liberal_Healthy'
}

PSM_CALIPER = 0.1
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
    df_features['spo2_range'] = df_features['avg_spo2'] - df_features['min_spo2']
    df_features['spo2_variability'] = df_features['spo2_range'] / df_features['avg_spo2']
    df_features['hypoxemia_burden'] = (100 - df_features['min_spo2']) * (1 + df_features['spo2_variability'])
    df_features['desaturation_severity'] = (100 - df_features['min_spo2']) * np.log1p(df_features['rdi'])
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

def perform_clustering_for_psm(gshc_df):
    """Perform clustering analysis for PSM"""
    print("Performing clustering for PSM analysis...")
    
    gshc_df = engineer_hypoxemia_features(gshc_df)
    X_hypox, _ = get_hypoxemia_feature_matrix(gshc_df)
    
    complete_mask = X_hypox.notna().all(axis=1)
    X_complete = X_hypox[complete_mask]
    data_complete = gshc_df[complete_mask].copy()
    
    print(f"Complete data available for {len(data_complete)} participants")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_complete)
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    print(f"Clustering complete: {len(np.unique(labels))} clusters")
    
    # Calculate outcomes
    data_complete['Y_Hypertension'] = data_complete.apply(
        lambda row: 1 if (not pd.isna(row['sbp_v2']) and not pd.isna(row['dbp_v2']) and 
                         (row['sbp_v2'] >= 120 or row['dbp_v2'] >= 80)) else 0, axis=1
    )
    data_complete['Y_Overweight'] = data_complete.apply(
        lambda row: 1 if (not pd.isna(row['bmi_v2']) and row['bmi_v2'] >= 25) else 0, axis=1
    )
    data_complete['Y_Composite'] = data_complete.apply(
        lambda row: 1 if (row['Y_Hypertension'] == 1 or row['Y_Overweight'] == 1) else 0, axis=1
    )
    
    final_data = data_complete.dropna(subset=['Y_Hypertension', 'Y_Overweight']).copy()
    final_labels = labels[:len(final_data)]
    
    print(f"Final PSM dataset: {len(final_data)} participants")
    print(f"Sub-phenotype distribution: {np.bincount(final_labels)}")
    
    return final_data, final_labels

def calculate_propensity_scores(data, labels):
    """Calculate propensity scores for sub-phenotype assignment"""
    print("Calculating propensity scores...")
    
    confounders = [
        'age_v1', 'gender', 'bmi_v1', 'sbp_v1', 'dbp_v1',
        'ess_v1', 'sleep_efficiency', 'arousal_index'
    ]
    
    psm_data = data[confounders + ['Y_Hypertension', 'Y_Overweight', 'Y_Composite']].copy()
    psm_data['treatment'] = labels
    
    psm_data = psm_data.dropna(subset=confounders)
    
    print(f"PSM analysis dataset: {len(psm_data)} participants")
    print(f"Treatment distribution: {np.bincount(psm_data['treatment'])}")
    
    X = psm_data[confounders]
    y = psm_data['treatment']
    
    scaler = StandardScaler()
    X_scaled = X.copy()
    continuous_vars = ['age_v1', 'bmi_v1', 'sbp_v1', 'dbp_v1', 'ess_v1', 'sleep_efficiency', 'arousal_index']
    X_scaled[continuous_vars] = scaler.fit_transform(X[continuous_vars])
    
    ps_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    ps_model.fit(X_scaled, y)
    
    propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
    psm_data['propensity_score'] = propensity_scores
    
    ps_auc = roc_auc_score(y, propensity_scores)
    print(f"Propensity score model AUC: {ps_auc:.3f}")
    
    return psm_data, ps_model, scaler, confounders

def perform_simple_matching(psm_data, caliper=PSM_CALIPER):
    """Perform simple 1:1 propensity score matching"""
    print(f"Performing 1:1 propensity score matching with caliper {caliper}...")
    
    treated = psm_data[psm_data['treatment'] == 1].copy()
    control = psm_data[psm_data['treatment'] == 0].copy()
    
    print(f"Before matching: Treated={len(treated)}, Control={len(control)}")
    
    matched_pairs = []
    used_control_indices = set()
    
    for _, treated_row in treated.iterrows():
        treated_ps = treated_row['propensity_score']
        
        # Find best match in control group
        available_control = control[~control.index.isin(used_control_indices)]
        if len(available_control) == 0:
            continue
            
        distances = abs(available_control['propensity_score'] - treated_ps)
        best_match_idx = distances.idxmin()
        best_distance = distances.min()
        
        if best_distance <= caliper:
            matched_pairs.append((treated_row.name, best_match_idx))
            used_control_indices.add(best_match_idx)
    
    # Create matched dataset
    matched_treated_indices = [pair[0] for pair in matched_pairs]
    matched_control_indices = [pair[1] for pair in matched_pairs]
    
    matched_treated = treated.loc[matched_treated_indices].copy()
    matched_control = control.loc[matched_control_indices].copy()
    
    matched_data = pd.concat([matched_treated, matched_control], ignore_index=True)
    
    n_pairs = len(matched_pairs)
    pair_ids = list(range(n_pairs)) * 2
    matched_data['pair_id'] = pair_ids
    
    print(f"After matching: {n_pairs} pairs ({len(matched_data)} participants)")
    print(f"Matching rate: {n_pairs/len(treated):.1%} of treated participants matched")
    
    return matched_data, matched_treated, matched_control

def analyze_outcomes_simple(matched_data):
    """Analyze outcomes in matched cohort (simplified)"""
    print("Analyzing outcomes in matched cohort...")
    
    outcomes = ['Y_Hypertension', 'Y_Overweight', 'Y_Composite']
    outcome_names = ['Hypertension', 'Overweight', 'Composite']
    
    matched_results = {}
    
    for outcome, outcome_name in zip(outcomes, outcome_names):
        print(f"Analyzing {outcome_name} outcome...")
        
        treated = matched_data[matched_data['treatment'] == 1][outcome]
        control = matched_data[matched_data['treatment'] == 0][outcome]
        
        treated_rate = treated.mean()
        control_rate = control.mean()
        risk_diff = treated_rate - control_rate
        
        # Simple chi-square test
        treated_events = treated.sum()
        treated_total = len(treated)
        control_events = control.sum()
        control_total = len(control)
        
        # 2x2 contingency table
        table = np.array([
            [treated_events, treated_total - treated_events],
            [control_events, control_total - control_events]
        ])
        
        # Chi-square test
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(table)
        except:
            p_value = 1.0
        
        # Odds ratio
        if treated_events > 0 and control_events > 0 and treated_total > treated_events and control_total > control_events:
            odds_ratio = (treated_events / (treated_total - treated_events)) / (control_events / (control_total - control_events))
            
            # Simple CI calculation
            log_or_se = np.sqrt(1/treated_events + 1/(treated_total - treated_events) + 
                               1/control_events + 1/(control_total - control_events))
            or_ci_lower = np.exp(np.log(odds_ratio) - 1.96 * log_or_se)
            or_ci_upper = np.exp(np.log(odds_ratio) + 1.96 * log_or_se)
        else:
            odds_ratio = np.nan
            or_ci_lower = np.nan
            or_ci_upper = np.nan
        
        matched_results[outcome] = {
            'treated_rate': treated_rate,
            'control_rate': control_rate,
            'risk_difference': risk_diff,
            'odds_ratio': odds_ratio,
            'or_ci_lower': or_ci_lower,
            'or_ci_upper': or_ci_upper,
            'p_value': p_value,
            'n_pairs': len(matched_data) // 2
        }
    
    return matched_results

def create_simple_psm_plot(psm_data, matched_data, matched_results, output_dir):
    """Create simplified PSM visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Propensity score distributions
    treated_ps = psm_data[psm_data['treatment'] == 1]['propensity_score']
    control_ps = psm_data[psm_data['treatment'] == 0]['propensity_score']
    
    ax1.hist(treated_ps, bins=30, alpha=0.7, label='Sub-phenotype 1', color='red', density=True)
    ax1.hist(control_ps, bins=30, alpha=0.7, label='Sub-phenotype 0', color='blue', density=True)
    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Propensity Score Distribution (Before Matching)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Matched propensity scores
    matched_treated_ps = matched_data[matched_data['treatment'] == 1]['propensity_score']
    matched_control_ps = matched_data[matched_data['treatment'] == 0]['propensity_score']
    
    ax2.hist(matched_treated_ps, bins=20, alpha=0.7, label='Sub-phenotype 1', color='red', density=True)
    ax2.hist(matched_control_ps, bins=20, alpha=0.7, label='Sub-phenotype 0', color='blue', density=True)
    ax2.set_xlabel('Propensity Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Propensity Score Distribution (After Matching)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Outcome rates
    outcomes = list(matched_results.keys())
    outcome_names = [outcome.replace('Y_', '') for outcome in outcomes]
    treated_rates = [matched_results[outcome]['treated_rate'] * 100 for outcome in outcomes]
    control_rates = [matched_results[outcome]['control_rate'] * 100 for outcome in outcomes]
    
    x = np.arange(len(outcome_names))
    width = 0.35
    
    ax3.bar(x - width/2, treated_rates, width, label='Sub-phenotype 1', color='red', alpha=0.8)
    ax3.bar(x + width/2, control_rates, width, label='Sub-phenotype 0', color='blue', alpha=0.8)
    ax3.set_xlabel('Outcomes')
    ax3.set_ylabel('Outcome Rate (%)')
    ax3.set_title('Outcome Rates in Matched Cohort')
    ax3.set_xticks(x)
    ax3.set_xticklabels(outcome_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Results summary
    ax4.axis('off')
    
    n_pairs = matched_results[outcomes[0]]['n_pairs']
    significant_outcomes = [outcome.replace('Y_', '') for outcome, results in matched_results.items() 
                          if results['p_value'] < 0.05]
    
    summary_text = f"""
PROPENSITY SCORE MATCHING RESULTS

MATCHED COHORT:
├─ Number of matched pairs: {n_pairs}
├─ Total participants: {n_pairs * 2}
└─ Matching method: 1:1 nearest neighbor

CAUSAL EFFECTS:
"""
    
    for outcome in outcomes:
        outcome_name = outcome.replace('Y_', '')
        or_val = matched_results[outcome]['odds_ratio']
        p_val = matched_results[outcome]['p_value']
        
        if not np.isnan(or_val):
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            summary_text += f"├─ {outcome_name}: OR={or_val:.2f}, p={p_val:.3f} {significance}\n"
        else:
            summary_text += f"├─ {outcome_name}: Unable to calculate\n"
    
    summary_text += f"\nKEY FINDINGS:\n"
    if significant_outcomes:
        summary_text += f"✅ Significant causal effects: {', '.join(significant_outcomes)}\n"
        summary_text += f"✅ PSM confirms differential outcomes\n"
    else:
        summary_text += f"⚠️  No statistically significant causal effects\n"
        summary_text += f"💡 May indicate confounding in original analysis\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.3))
    
    plt.suptitle('Simplified Propensity Score Matching Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simplified_psm_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Simplified PSM plots saved")

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Simplified Propensity Score Matching Analysis ===")
    print("Running PSM analysis without problematic dependencies...")
    
    # Load data
    print("\n--- Loading and preparing data ---")
    base_df = load_and_map_data(DATA_FILES)
    optimal_gshc_df = create_optimal_gshc(base_df)
    print(f"Optimal GSHC size: {len(optimal_gshc_df)}")
    
    # Perform clustering
    print("\n--- Performing clustering analysis ---")
    clustered_data, labels = perform_clustering_for_psm(optimal_gshc_df)
    
    # Calculate propensity scores
    print("\n--- Calculating propensity scores ---")
    psm_data, ps_model, scaler, confounders = calculate_propensity_scores(clustered_data, labels)
    
    # Perform matching
    print("\n--- Performing propensity score matching ---")
    matched_data, matched_treated, matched_control = perform_simple_matching(psm_data)
    
    # Analyze outcomes
    print("\n--- Analyzing outcomes in matched cohort ---")
    matched_results = analyze_outcomes_simple(matched_data)
    
    # Generate visualization
    print("\n--- Generating PSM visualization ---")
    create_simple_psm_plot(psm_data, matched_data, matched_results, OUTPUT_DIR)
    
    # Save results
    psm_results = {
        'psm_data': psm_data,
        'matched_data': matched_data,
        'matched_results': matched_results,
        'confounders': confounders
    }
    
    results_file = os.path.join(OUTPUT_DIR, 'simplified_psm_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(psm_results, f)
    
    print(f"\nSimplified PSM results saved to: {results_file}")
    
    # Print summary
    print(f"\n📊 PSM Analysis Summary:")
    n_pairs = len(matched_data) // 2
    print(f"   📈 Matched Pairs: {n_pairs}")
    
    significant_outcomes = []
    for outcome, results in matched_results.items():
        if results['p_value'] < 0.05:
            significant_outcomes.append(outcome.replace('Y_', ''))
    
    if significant_outcomes:
        print(f"   ✅ Significant causal effects: {', '.join(significant_outcomes)}")
        print(f"   🎉 PSM confirms causal relationships!")
    else:
        print(f"   ⚠️  No statistically significant causal effects detected")
    
    print(f"\n✅ Simplified PSM analysis complete!")
    print(f"📁 Output: {OUTPUT_DIR}/simplified_psm_analysis.png")
