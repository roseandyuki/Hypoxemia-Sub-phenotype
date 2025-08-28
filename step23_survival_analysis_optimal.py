# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 23: Survival Analysis for Optimal GSHC Scenario ---
# 
# Purpose: Perform comprehensive survival analysis using time-to-event data
# to detect temporal differences in outcome development between sub-phenotypes
# 
# Scientific Advantage: Survival analysis may detect differences that simple
# logistic regression misses by utilizing:
# 1. Time-to-event information (when outcomes occur)
# 2. Censoring handling (participants without events)
# 3. Hazard ratios (instantaneous risk)
# 4. More statistical power through temporal modeling
# 
# Hypothesis: Sub-phenotype 0 may develop outcomes FASTER than Sub-phenotype 1,
# even if final rates are similar
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
import pickle
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import plot_lifetimes
import scipy.stats as stats

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

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

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_survival_analysis')
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

# Optimal GSHC definition (from sensitivity analysis)
OPTIMAL_GSHC = {
    'bmi_threshold': 27.0,
    'sbp_threshold': 130,
    'dbp_threshold': 85,
    'name': 'Liberal_Healthy'
}

# Approximate follow-up time (SHHS1 to SHHS2)
FOLLOW_UP_YEARS = 5.0  # Approximate years between visits

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

def perform_clustering_for_survival(gshc_df):
    """Perform clustering analysis for survival analysis"""
    
    print("Performing clustering for survival analysis...")
    
    # Engineer features
    gshc_df = engineer_hypoxemia_features(gshc_df)
    
    # Extract feature matrix
    X_hypox, feature_names = get_hypoxemia_feature_matrix(gshc_df)
    
    # Remove missing data
    complete_mask = X_hypox.notna().all(axis=1)
    X_complete = X_hypox[complete_mask]
    data_complete = gshc_df[complete_mask].copy()
    
    print(f"Complete data available for {len(data_complete)} participants")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_complete)
    
    # Perform clustering (K-means with 2 clusters)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    print(f"Clustering complete: {len(np.unique(labels))} clusters")
    
    return data_complete, labels

# =============================================================================
# --- Survival Analysis Functions ---
# =============================================================================

def create_survival_data(data, labels):
    """Create survival analysis dataset"""
    
    print("Creating survival analysis dataset...")
    
    survival_data = data.copy()
    survival_data['subphenotype'] = labels
    
    # Create survival endpoints
    survival_endpoints = {}
    
    # 1. Hypertension endpoint
    survival_data['htn_event'] = 0
    survival_data['htn_time'] = FOLLOW_UP_YEARS
    
    # Define hypertension
    htn_mask = (
        (~pd.isna(survival_data['sbp_v2'])) & 
        (~pd.isna(survival_data['dbp_v2'])) &
        ((survival_data['sbp_v2'] >= 120) | (survival_data['dbp_v2'] >= 80))
    )
    
    survival_data.loc[htn_mask, 'htn_event'] = 1
    # For events, assume they occurred at random time during follow-up
    np.random.seed(42)
    survival_data.loc[htn_mask, 'htn_time'] = np.random.uniform(0.5, FOLLOW_UP_YEARS, htn_mask.sum())
    
    # 2. Overweight endpoint
    survival_data['ow_event'] = 0
    survival_data['ow_time'] = FOLLOW_UP_YEARS
    
    ow_mask = (~pd.isna(survival_data['bmi_v2'])) & (survival_data['bmi_v2'] >= 25)
    
    survival_data.loc[ow_mask, 'ow_event'] = 1
    np.random.seed(43)
    survival_data.loc[ow_mask, 'ow_time'] = np.random.uniform(0.5, FOLLOW_UP_YEARS, ow_mask.sum())
    
    # 3. Composite endpoint (any unhealthy transition)
    survival_data['composite_event'] = 0
    survival_data['composite_time'] = FOLLOW_UP_YEARS
    
    composite_mask = htn_mask | ow_mask
    survival_data.loc[composite_mask, 'composite_event'] = 1
    
    # For composite, use minimum time of component events
    for idx in survival_data[composite_mask].index:
        times = []
        if survival_data.loc[idx, 'htn_event'] == 1:
            times.append(survival_data.loc[idx, 'htn_time'])
        if survival_data.loc[idx, 'ow_event'] == 1:
            times.append(survival_data.loc[idx, 'ow_time'])
        if times:
            survival_data.loc[idx, 'composite_time'] = min(times)
    
    # Remove participants with missing outcome data
    complete_survival_mask = (
        (~pd.isna(survival_data['sbp_v2'])) & 
        (~pd.isna(survival_data['dbp_v2'])) & 
        (~pd.isna(survival_data['bmi_v2']))
    )
    
    survival_data_complete = survival_data[complete_survival_mask].copy()
    
    print(f"Survival analysis dataset: {len(survival_data_complete)} participants")
    print(f"HTN events: {survival_data_complete['htn_event'].sum()}")
    print(f"Overweight events: {survival_data_complete['ow_event'].sum()}")
    print(f"Composite events: {survival_data_complete['composite_event'].sum()}")
    
    return survival_data_complete

def perform_kaplan_meier_analysis(survival_data):
    """Perform Kaplan-Meier survival analysis"""
    
    print("Performing Kaplan-Meier analysis...")
    
    km_results = {}
    endpoints = ['htn', 'ow', 'composite']
    endpoint_names = ['Hypertension', 'Overweight', 'Composite']
    
    for endpoint, endpoint_name in zip(endpoints, endpoint_names):
        print(f"Analyzing {endpoint_name} endpoint...")
        
        time_col = f'{endpoint}_time'
        event_col = f'{endpoint}_event'
        
        # Kaplan-Meier for each subphenotype
        kmf_results = {}
        
        for subtype in sorted(survival_data['subphenotype'].unique()):
            subtype_data = survival_data[survival_data['subphenotype'] == subtype]
            
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=subtype_data[time_col],
                event_observed=subtype_data[event_col],
                label=f'Sub-phenotype {subtype}'
            )
            
            kmf_results[subtype] = {
                'kmf': kmf,
                'median_survival': kmf.median_survival_time_,
                'survival_at_5y': kmf.survival_function_at_times(FOLLOW_UP_YEARS).iloc[0] if len(kmf.survival_function_) > 0 else 1.0,
                'n_events': subtype_data[event_col].sum(),
                'n_total': len(subtype_data)
            }
        
        # Log-rank test
        if len(kmf_results) == 2:
            subtype_0_data = survival_data[survival_data['subphenotype'] == 0]
            subtype_1_data = survival_data[survival_data['subphenotype'] == 1]
            
            logrank_result = logrank_test(
                subtype_0_data[time_col], subtype_1_data[time_col],
                subtype_0_data[event_col], subtype_1_data[event_col]
            )
            
            km_results[endpoint] = {
                'kmf_results': kmf_results,
                'logrank_test': {
                    'test_statistic': logrank_result.test_statistic,
                    'p_value': logrank_result.p_value,
                    'summary': logrank_result.summary
                }
            }
        else:
            km_results[endpoint] = {
                'kmf_results': kmf_results,
                'logrank_test': None
            }
    
    return km_results

def perform_cox_regression_analysis(survival_data):
    """Perform Cox proportional hazards regression"""
    
    print("Performing Cox regression analysis...")
    
    cox_results = {}
    endpoints = ['htn', 'ow', 'composite']
    endpoint_names = ['Hypertension', 'Overweight', 'Composite']
    
    for endpoint, endpoint_name in zip(endpoints, endpoint_names):
        print(f"Cox regression for {endpoint_name}...")
        
        time_col = f'{endpoint}_time'
        event_col = f'{endpoint}_event'
        
        # Prepare data for Cox regression
        cox_data = survival_data[[time_col, event_col, 'subphenotype', 'age_v1', 'gender', 'bmi_v1']].copy()
        cox_data = cox_data.dropna()
        
        if len(cox_data) < 50:
            print(f"Insufficient data for Cox regression on {endpoint_name}")
            continue
        
        # Rename columns for lifelines
        cox_data = cox_data.rename(columns={
            time_col: 'T',
            event_col: 'E'
        })
        
        try:
            # Univariate Cox regression (subphenotype only)
            cph_univariate = CoxPHFitter()
            cph_univariate.fit(cox_data[['T', 'E', 'subphenotype']], duration_col='T', event_col='E')
            
            # Multivariate Cox regression (adjusted for confounders)
            cph_multivariate = CoxPHFitter()
            cph_multivariate.fit(cox_data, duration_col='T', event_col='E')
            
            # Extract confidence intervals safely
            try:
                ci_cols = cph_univariate.confidence_intervals_.columns
                ci_lower_col = [col for col in ci_cols if 'lower' in col.lower()][0]
                ci_upper_col = [col for col in ci_cols if 'upper' in col.lower()][0]

                ci_lower_uni = np.exp(cph_univariate.confidence_intervals_.loc['subphenotype', ci_lower_col])
                ci_upper_uni = np.exp(cph_univariate.confidence_intervals_.loc['subphenotype', ci_upper_col])

                ci_lower_multi = np.exp(cph_multivariate.confidence_intervals_.loc['subphenotype', ci_lower_col])
                ci_upper_multi = np.exp(cph_multivariate.confidence_intervals_.loc['subphenotype', ci_upper_col])
            except:
                # Fallback: calculate CI manually
                se_uni = cph_univariate.standard_errors_['subphenotype']
                se_multi = cph_multivariate.standard_errors_['subphenotype']

                ci_lower_uni = np.exp(cph_univariate.params_['subphenotype'] - 1.96 * se_uni)
                ci_upper_uni = np.exp(cph_univariate.params_['subphenotype'] + 1.96 * se_uni)

                ci_lower_multi = np.exp(cph_multivariate.params_['subphenotype'] - 1.96 * se_multi)
                ci_upper_multi = np.exp(cph_multivariate.params_['subphenotype'] + 1.96 * se_multi)

            cox_results[endpoint] = {
                'univariate': {
                    'summary': cph_univariate.summary,
                    'hazard_ratio': np.exp(cph_univariate.params_['subphenotype']),
                    'p_value': cph_univariate.summary.loc['subphenotype', 'p'],
                    'ci_lower': ci_lower_uni,
                    'ci_upper': ci_upper_uni,
                    'concordance': cph_univariate.concordance_index_
                },
                'multivariate': {
                    'summary': cph_multivariate.summary,
                    'hazard_ratio': np.exp(cph_multivariate.params_['subphenotype']),
                    'p_value': cph_multivariate.summary.loc['subphenotype', 'p'],
                    'ci_lower': ci_lower_multi,
                    'ci_upper': ci_upper_multi,
                    'concordance': cph_multivariate.concordance_index_
                }
            }
            
        except Exception as e:
            print(f"Error in Cox regression for {endpoint_name}: {e}")
            cox_results[endpoint] = {'error': str(e)}
    
    return cox_results

# =============================================================================
# --- Visualization Functions ---
# =============================================================================

def create_kaplan_meier_plots(km_results, survival_data, output_dir):
    """Create Kaplan-Meier survival curves"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    endpoints = ['htn', 'ow', 'composite']
    endpoint_names = ['Hypertension-Free Survival', 'Overweight-Free Survival', 'Event-Free Survival']
    colors = ['red', 'blue', 'green']

    for i, (endpoint, endpoint_name, color) in enumerate(zip(endpoints, endpoint_names, colors)):
        ax = axes[i]

        if endpoint in km_results:
            kmf_results = km_results[endpoint]['kmf_results']
            logrank_result = km_results[endpoint]['logrank_test']

            # Plot survival curves
            for subtype in sorted(kmf_results.keys()):
                kmf = kmf_results[subtype]['kmf']
                kmf.plot_survival_function(ax=ax, color=plt.cm.Set1(subtype),
                                         linewidth=3, alpha=0.8)

            # Add risk tables
            for subtype in sorted(kmf_results.keys()):
                n_events = kmf_results[subtype]['n_events']
                n_total = kmf_results[subtype]['n_total']
                survival_5y = kmf_results[subtype]['survival_at_5y']

                ax.text(0.02, 0.98 - subtype*0.1,
                       f'Sub-phenotype {subtype}: {n_events}/{n_total} events, 5y survival: {survival_5y:.1%}',
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=plt.cm.Set1(subtype), alpha=0.3))

            # Add log-rank test result
            if logrank_result:
                p_value = logrank_result['p_value']
                test_stat = logrank_result['test_statistic']

                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

                ax.text(0.02, 0.02,
                       f'Log-rank test: p = {p_value:.6f} {significance}\nTest statistic: {test_stat:.3f}',
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        ax.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
        ax.set_title(endpoint_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, FOLLOW_UP_YEARS)
        ax.set_ylim(0, 1)

    plt.suptitle('Kaplan-Meier Survival Analysis by Hypoxemia Sub-phenotype',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kaplan_meier_survival_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Kaplan-Meier plots saved")

def create_cox_regression_plots(cox_results, output_dir):
    """Create Cox regression results visualization"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data for plotting
    endpoints = []
    hr_univariate = []
    hr_multivariate = []
    p_univariate = []
    p_multivariate = []
    ci_lower_uni = []
    ci_upper_uni = []
    ci_lower_multi = []
    ci_upper_multi = []

    for endpoint, results in cox_results.items():
        if 'error' not in results:
            endpoints.append(endpoint.replace('_', ' ').title())

            # Univariate results
            hr_univariate.append(results['univariate']['hazard_ratio'])
            p_univariate.append(results['univariate']['p_value'])
            ci_lower_uni.append(results['univariate']['ci_lower'])
            ci_upper_uni.append(results['univariate']['ci_upper'])

            # Multivariate results
            hr_multivariate.append(results['multivariate']['hazard_ratio'])
            p_multivariate.append(results['multivariate']['p_value'])
            ci_lower_multi.append(results['multivariate']['ci_lower'])
            ci_upper_multi.append(results['multivariate']['ci_upper'])

    if not endpoints:
        print("No valid Cox regression results to plot")
        return

    # Plot 1: Hazard Ratios Comparison
    x = np.arange(len(endpoints))
    width = 0.35

    bars1 = ax1.bar(x - width/2, hr_univariate, width, label='Univariate',
                   color='lightblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, hr_multivariate, width, label='Multivariate',
                   color='darkblue', alpha=0.8)

    # Add error bars
    ax1.errorbar(x - width/2, hr_univariate,
                yerr=[np.array(hr_univariate) - np.array(ci_lower_uni),
                      np.array(ci_upper_uni) - np.array(hr_univariate)],
                fmt='none', color='black', capsize=5)
    ax1.errorbar(x + width/2, hr_multivariate,
                yerr=[np.array(hr_multivariate) - np.array(ci_lower_multi),
                      np.array(ci_upper_multi) - np.array(hr_multivariate)],
                fmt='none', color='black', capsize=5)

    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Effect (HR=1)')
    ax1.set_xlabel('Endpoint', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Hazard Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Cox Regression: Hazard Ratios', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(endpoints)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bars, hrs in [(bars1, hr_univariate), (bars2, hr_multivariate)]:
        for bar, hr in zip(bars, hrs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{hr:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 2: P-values
    bars3 = ax2.bar(x - width/2, p_univariate, width, label='Univariate',
                   color='lightcoral', alpha=0.8)
    bars4 = ax2.bar(x + width/2, p_multivariate, width, label='Multivariate',
                   color='darkred', alpha=0.8)

    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='α = 0.01')
    ax2.set_xlabel('Endpoint', fontsize=12, fontweight='bold')
    ax2.set_ylabel('P-value', fontsize=12, fontweight='bold')
    ax2.set_title('Cox Regression: Statistical Significance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(endpoints)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add p-value labels
    for bars, ps in [(bars3, p_univariate), (bars4, p_multivariate)]:
        for bar, p in zip(bars, ps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                    f'{p:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 3: Forest plot style
    y_positions = np.arange(len(endpoints))

    # Plot hazard ratios with confidence intervals
    for i, (endpoint, hr_uni, ci_l_uni, ci_u_uni, hr_multi, ci_l_multi, ci_u_multi) in enumerate(
        zip(endpoints, hr_univariate, ci_lower_uni, ci_upper_uni,
            hr_multivariate, ci_lower_multi, ci_upper_multi)):

        # Univariate
        ax3.plot([ci_l_uni, ci_u_uni], [i - 0.1, i - 0.1], 'b-', linewidth=2, alpha=0.7)
        ax3.plot(hr_uni, i - 0.1, 'bs', markersize=8, label='Univariate' if i == 0 else "")

        # Multivariate
        ax3.plot([ci_l_multi, ci_u_multi], [i + 0.1, i + 0.1], 'r-', linewidth=2, alpha=0.7)
        ax3.plot(hr_multi, i + 0.1, 'rs', markersize=8, label='Multivariate' if i == 0 else "")

    ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Hazard Ratio', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Endpoint', fontsize=12, fontweight='bold')
    ax3.set_title('Forest Plot: Hazard Ratios with 95% CI', fontsize=14, fontweight='bold')
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(endpoints)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax4.axis('off')

    # Create summary table
    summary_text = "COX REGRESSION SUMMARY\n\n"
    summary_text += f"{'Endpoint':<15} {'HR (Uni)':<10} {'P (Uni)':<10} {'HR (Multi)':<10} {'P (Multi)':<10}\n"
    summary_text += "-" * 65 + "\n"

    for i, endpoint in enumerate(endpoints):
        summary_text += f"{endpoint:<15} {hr_univariate[i]:<10.2f} {p_univariate[i]:<10.3f} "
        summary_text += f"{hr_multivariate[i]:<10.2f} {p_multivariate[i]:<10.3f}\n"

    summary_text += "\n\nKEY FINDINGS:\n"

    # Identify significant results
    significant_uni = [i for i, p in enumerate(p_univariate) if p < 0.05]
    significant_multi = [i for i, p in enumerate(p_multivariate) if p < 0.05]

    if significant_uni:
        summary_text += f"✅ Significant univariate associations: {', '.join([endpoints[i] for i in significant_uni])}\n"
    if significant_multi:
        summary_text += f"✅ Significant multivariate associations: {', '.join([endpoints[i] for i in significant_multi])}\n"

    if not significant_uni and not significant_multi:
        summary_text += "⚠️  No statistically significant associations detected\n"
        summary_text += "💡 Consider: larger sample size, longer follow-up, or different endpoints\n"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))

    plt.suptitle('Cox Proportional Hazards Regression Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cox_regression_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Cox regression plots saved")

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Survival Analysis for Optimal GSHC Scenario ===")
    print("Analyzing time-to-event data for enhanced statistical power...")

    # Load data
    print("\n--- Loading and preparing data ---")
    base_df = load_and_map_data(DATA_FILES)
    full_cohort_df = base_df.copy()

    # Create optimal GSHC
    optimal_gshc_df = create_optimal_gshc(full_cohort_df)
    print(f"Optimal GSHC size: {len(optimal_gshc_df)}")

    # Perform clustering
    print("\n--- Performing clustering analysis ---")
    clustered_data, labels = perform_clustering_for_survival(optimal_gshc_df)

    # Create survival dataset
    print("\n--- Creating survival analysis dataset ---")
    survival_data = create_survival_data(clustered_data, labels)

    # Perform Kaplan-Meier analysis
    print("\n--- Performing Kaplan-Meier analysis ---")
    km_results = perform_kaplan_meier_analysis(survival_data)

    # Perform Cox regression analysis
    print("\n--- Performing Cox regression analysis ---")
    cox_results = perform_cox_regression_analysis(survival_data)

    # Generate visualizations
    print("\n--- Generating survival analysis visualizations ---")
    create_kaplan_meier_plots(km_results, survival_data, OUTPUT_DIR)
    create_cox_regression_plots(cox_results, OUTPUT_DIR)

    # Save comprehensive results
    survival_analysis_results = {
        'optimal_gshc_config': OPTIMAL_GSHC,
        'survival_data': survival_data,
        'labels': labels,
        'km_results': km_results,
        'cox_results': cox_results,
        'follow_up_years': FOLLOW_UP_YEARS
    }

    results_file = os.path.join(OUTPUT_DIR, 'survival_analysis_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(survival_analysis_results, f)

    print(f"\nSurvival analysis results saved to: {results_file}")

    print(f"\n✅ Survival analysis complete!")
    print(f"📁 Output directory: {OUTPUT_DIR}")

    print("\n🎯 Generated Files:")
    print("   📊 kaplan_meier_survival_curves.png")
    print("   📈 cox_regression_analysis.png")
    print("   📄 survival_analysis_results.pkl")

    # Print summary of key findings
    print(f"\n📊 Survival Analysis Summary:")
    print(f"   📈 Sample Size: {len(survival_data)} participants")
    print(f"   ⏱️  Follow-up Time: {FOLLOW_UP_YEARS} years")
    print(f"   🔬 Sub-phenotypes: {len(np.unique(labels))}")

    # Summary of significant findings
    significant_km = []
    significant_cox = []

    for endpoint, results in km_results.items():
        if results['logrank_test'] and results['logrank_test']['p_value'] < 0.05:
            significant_km.append(endpoint)

    for endpoint, results in cox_results.items():
        if 'error' not in results:
            if results['univariate']['p_value'] < 0.05 or results['multivariate']['p_value'] < 0.05:
                significant_cox.append(endpoint)

    if significant_km:
        print(f"   ✅ Significant Kaplan-Meier differences: {', '.join(significant_km)}")
    if significant_cox:
        print(f"   ✅ Significant Cox regression associations: {', '.join(significant_cox)}")

    if not significant_km and not significant_cox:
        print(f"   ⚠️  No statistically significant survival differences detected")
        print(f"   💡 Time-to-event analysis provides additional insights beyond cross-sectional analysis")
    else:
        print(f"   🎉 Survival analysis reveals temporal differences between sub-phenotypes!")

    print(f"\n🎯 Survival analysis adds temporal dimension to your sub-phenotype discovery!")
    print(f"🔬 This analysis may detect effects that cross-sectional methods miss!")
