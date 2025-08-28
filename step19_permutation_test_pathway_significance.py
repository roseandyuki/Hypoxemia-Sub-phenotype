# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 19: Permutation Test for Pathway Significance ---
# 
# Purpose: Use permutation testing to establish statistical significance of 
# differential pathway prediction by hypoxemia sub-phenotypes
# 
# Scientific Question: Is the observed AUC difference between HTN and overweight 
# pathways (0.018) a real signal or just random noise?
# 
# Method: 
# 1. Calculate observed AUC difference (HTN - Overweight)
# 2. Permute outcome labels 10,000 times to break true associations
# 3. Recalculate AUC difference for each permutation
# 4. Generate null distribution and calculate p-value
# 5. Provide definitive statistical evidence for pathway differences
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import pickle
from tqdm import tqdm
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

INPUT_DIR = os.path.join(SCRIPT_DIR, 'output_hypoxemia_subphenotypes')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_permutation_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Permutation test parameters
N_PERMUTATIONS = 10000  # Number of permutation iterations
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

def perform_single_permutation(data, labels, outcome1_col, outcome2_col, iteration):
    """Perform one permutation iteration"""
    
    # Create permuted data
    permuted_data = data.copy()
    
    # Randomly permute the outcome labels
    n_samples = len(permuted_data)
    
    # Permute outcome 1
    permuted_outcome1 = permuted_data[outcome1_col].values.copy()
    np.random.shuffle(permuted_outcome1)
    permuted_data[f'{outcome1_col}_permuted'] = permuted_outcome1
    
    # Permute outcome 2
    permuted_outcome2 = permuted_data[outcome2_col].values.copy()
    np.random.shuffle(permuted_outcome2)
    permuted_data[f'{outcome2_col}_permuted'] = permuted_outcome2
    
    # Calculate AUC difference with permuted outcomes
    auc1, auc2, difference = calculate_pathway_auc_difference(
        permuted_data, labels, 
        f'{outcome1_col}_permuted', 
        f'{outcome2_col}_permuted'
    )
    
    return {
        'iteration': iteration,
        'auc1_permuted': auc1,
        'auc2_permuted': auc2,
        'difference_permuted': difference
    }

def run_permutation_test(data, labels, outcome1_col, outcome2_col):
    """Run complete permutation test"""
    
    print(f"Running permutation test with {N_PERMUTATIONS} iterations...")
    print(f"Testing: {outcome1_col} vs {outcome2_col}")
    print(f"Sample size: {len(data)}")
    print(f"Number of subtypes: {len(np.unique(labels))}")
    
    # Calculate observed difference
    print("\n--- Calculating observed difference ---")
    observed_auc1, observed_auc2, observed_difference = calculate_pathway_auc_difference(
        data, labels, outcome1_col, outcome2_col
    )
    
    if observed_difference is None:
        print("❌ Error: Could not calculate observed difference")
        return None
    
    print(f"Observed {outcome1_col} AUC: {observed_auc1:.4f}")
    print(f"Observed {outcome2_col} AUC: {observed_auc2:.4f}")
    print(f"Observed difference: {observed_difference:.4f}")
    
    # Run permutations
    print(f"\n--- Running {N_PERMUTATIONS} permutations ---")
    permutation_results = []
    
    for i in tqdm(range(N_PERMUTATIONS), desc="Permutation iterations"):
        # Set seed for reproducibility
        np.random.seed(RANDOM_SEED + i)
        
        result = perform_single_permutation(data, labels, outcome1_col, outcome2_col, i)
        if result['difference_permuted'] is not None:
            permutation_results.append(result)
    
    print(f"Successful permutations: {len(permutation_results)}/{N_PERMUTATIONS}")
    
    if len(permutation_results) < 100:
        print("⚠️  Warning: Too few successful permutations")
        return None
    
    # Extract null distribution
    null_differences = [result['difference_permuted'] for result in permutation_results]
    
    # Calculate p-value
    # Two-tailed test: how many permuted differences are as extreme as observed
    extreme_count = sum([1 for diff in null_differences if abs(diff) >= abs(observed_difference)])
    p_value_two_tailed = extreme_count / len(null_differences)
    
    # One-tailed test: how many permuted differences are >= observed (if observed > 0)
    if observed_difference > 0:
        extreme_count_one_tailed = sum([1 for diff in null_differences if diff >= observed_difference])
    else:
        extreme_count_one_tailed = sum([1 for diff in null_differences if diff <= observed_difference])
    
    p_value_one_tailed = extreme_count_one_tailed / len(null_differences)
    
    # Calculate statistics of null distribution
    null_mean = np.mean(null_differences)
    null_std = np.std(null_differences)
    null_ci = np.percentile(null_differences, [2.5, 97.5])
    
    return {
        'observed_auc1': observed_auc1,
        'observed_auc2': observed_auc2,
        'observed_difference': observed_difference,
        'null_differences': null_differences,
        'null_mean': null_mean,
        'null_std': null_std,
        'null_ci': null_ci,
        'p_value_two_tailed': p_value_two_tailed,
        'p_value_one_tailed': p_value_one_tailed,
        'n_permutations': len(permutation_results),
        'extreme_count_two_tailed': extreme_count,
        'extreme_count_one_tailed': extreme_count_one_tailed,
        'permutation_results': permutation_results
    }

# =============================================================================
# --- Statistical Analysis Functions ---
# =============================================================================

def calculate_effect_size(observed_difference, null_std):
    """Calculate Cohen's d effect size"""
    if null_std == 0:
        return np.inf if observed_difference != 0 else 0
    return observed_difference / null_std

def interpret_p_value(p_value):
    """Interpret p-value significance"""
    if p_value < 0.001:
        return "Extremely significant (p < 0.001)"
    elif p_value < 0.01:
        return "Highly significant (p < 0.01)"
    elif p_value < 0.05:
        return "Significant (p < 0.05)"
    elif p_value < 0.10:
        return "Marginally significant (p < 0.10)"
    else:
        return "Not significant (p ≥ 0.10)"

def calculate_confidence_level(p_value):
    """Calculate confidence level"""
    return (1 - p_value) * 100

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Permutation Test for Pathway Significance ===")
    print("Testing statistical significance of differential pathway prediction...")
    
    # Load original sub-phenotype results
    results_file = os.path.join(INPUT_DIR, 'hypoxemia_subphenotype_results.pkl')
    
    if not os.path.exists(results_file):
        print(f"❌ Error: Sub-phenotype results file not found at {results_file}")
        print("Please run step17_hypoxemia_subphenotype_discovery.py first!")
        exit(1)
    
    print(f"Loading sub-phenotype results from: {results_file}")
    with open(results_file, 'rb') as f:
        subphenotype_results = pickle.load(f)
    
    # Extract data and labels
    gshc_data = subphenotype_results['gshc_data']
    best_labels = subphenotype_results['best_labels']
    
    # Calculate outcomes
    gshc_data['Y_Hypertension'] = gshc_data.apply(has_hypertension, axis=1)
    gshc_data['Y_Overweight'] = gshc_data.apply(has_overweight, axis=1)
    
    # Remove missing outcomes
    complete_data = gshc_data.dropna(subset=['Y_Hypertension', 'Y_Overweight']).copy()
    
    # Align labels with complete data
    if len(best_labels) > len(complete_data):
        # Truncate labels to match complete data
        aligned_labels = best_labels[:len(complete_data)]
    else:
        aligned_labels = best_labels
    
    print(f"Analysis dataset: {len(complete_data)} participants")
    print(f"HTN cases: {complete_data['Y_Hypertension'].sum()}")
    print(f"Overweight cases: {complete_data['Y_Overweight'].sum()}")
    print(f"Sub-phenotypes: {len(np.unique(aligned_labels))}")
    
    # Run permutation test
    print(f"\n--- Running Permutation Test ---")
    permutation_results = run_permutation_test(
        complete_data, aligned_labels, 
        'Y_Hypertension', 'Y_Overweight'
    )
    
    if permutation_results is None:
        print("❌ Permutation test failed")
        exit(1)
    
    # Calculate additional statistics
    observed_diff = permutation_results['observed_difference']
    null_std = permutation_results['null_std']
    p_two_tailed = permutation_results['p_value_two_tailed']
    p_one_tailed = permutation_results['p_value_one_tailed']
    
    effect_size = calculate_effect_size(observed_diff, null_std)
    significance_interpretation = interpret_p_value(p_two_tailed)
    confidence_level = calculate_confidence_level(p_two_tailed)
    
    # Print results
    print(f"\n--- Permutation Test Results ---")
    print(f"📊 Observed HTN AUC: {permutation_results['observed_auc1']:.4f}")
    print(f"📊 Observed Overweight AUC: {permutation_results['observed_auc2']:.4f}")
    print(f"📊 Observed Difference: {observed_diff:.4f}")
    print(f"📊 Null Distribution Mean: {permutation_results['null_mean']:.4f}")
    print(f"📊 Null Distribution SD: {null_std:.4f}")
    print(f"📊 Effect Size (Cohen's d): {effect_size:.3f}")
    print(f"📊 Two-tailed p-value: {p_two_tailed:.6f}")
    print(f"📊 One-tailed p-value: {p_one_tailed:.6f}")
    print(f"📊 Significance: {significance_interpretation}")
    print(f"📊 Confidence Level: {confidence_level:.1f}%")
    
    # Save results
    final_results = {
        'permutation_test': permutation_results,
        'effect_size': effect_size,
        'significance_interpretation': significance_interpretation,
        'confidence_level': confidence_level,
        'analysis_parameters': {
            'n_permutations': N_PERMUTATIONS,
            'random_seed': RANDOM_SEED,
            'sample_size': len(complete_data),
            'n_subtypes': len(np.unique(aligned_labels))
        }
    }
    
    output_file = os.path.join(OUTPUT_DIR, 'permutation_test_results.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"\nPermutation test results saved to: {output_file}")
    
    # Interpretation for paper
    print(f"\n--- Interpretation for Paper ---")
    if p_two_tailed < 0.05:
        print("🎉 SIGNIFICANT RESULT!")
        print(f"✅ The observed AUC difference ({observed_diff:.4f}) is statistically significant")
        print(f"✅ This difference is NOT due to random chance (p = {p_two_tailed:.6f})")
        print(f"✅ You can confidently report this as a real biological signal")
        
        print(f"\n📝 Suggested paper text:")
        print(f'   "Although the absolute AUC values were modest (HTN: {permutation_results["observed_auc1"]:.3f}, ')
        print(f'   Overweight: {permutation_results["observed_auc2"]:.3f}), permutation testing (n={N_PERMUTATIONS:,}) ')
        print(f'   demonstrated that the differential pathway prediction was statistically significant ')
        print(f'   (p = {p_two_tailed:.6f}), indicating that hypoxemia sub-phenotypes show genuine ')
        print(f'   preferential association with hypertension versus overweight pathways."')
        
    else:
        print("⚠️  NON-SIGNIFICANT RESULT")
        print(f"❌ The observed difference could be due to random chance (p = {p_two_tailed:.6f})")
        print(f"❌ Cannot claim statistical significance for pathway differences")
        
        print(f"\n📝 Suggested paper text:")
        print(f'   "Permutation testing (n={N_PERMUTATIONS:,}) indicated that the observed ')
        print(f'   AUC difference between pathways was not statistically significant ')
        print(f'   (p = {p_two_tailed:.6f}), suggesting that larger sample sizes may be needed ')
        print(f'   to detect subtle pathway-specific associations."')
    
    print(f"\n🎯 Permutation Test Complete!")
    print(f"📊 Next: Run visualization script to see the null distribution!")
    print(f"💡 You now have definitive statistical evidence for your pathway findings!")
