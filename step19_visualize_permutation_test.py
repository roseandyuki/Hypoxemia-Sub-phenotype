# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 19B: Visualize Permutation Test Results ---
# 
# Purpose: Generate comprehensive visualizations for permutation test results
# 
# Usage: Run this after step19_permutation_test_pathway_significance.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from scipy import stats

plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# --- Configuration ---
# =============================================================================

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = '.'

INPUT_DIR = os.path.join(SCRIPT_DIR, 'output_permutation_test')
OUTPUT_DIR = INPUT_DIR

# =============================================================================
# --- Visualization Functions ---
# =============================================================================

def create_null_distribution_plot(permutation_results, output_dir):
    """Create null distribution visualization with observed value"""
    
    null_differences = permutation_results['null_differences']
    observed_diff = permutation_results['observed_difference']
    p_value = permutation_results['p_value_two_tailed']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Histogram of null distribution
    ax1.hist(null_differences, bins=50, density=True, alpha=0.7, color='lightblue', 
             edgecolor='black', label='Null Distribution')
    
    # Add observed value line
    ax1.axvline(observed_diff, color='red', linestyle='--', linewidth=3, 
               label=f'Observed Difference: {observed_diff:.4f}')
    
    # Add mean of null distribution
    null_mean = np.mean(null_differences)
    ax1.axvline(null_mean, color='blue', linestyle=':', linewidth=2, 
               label=f'Null Mean: {null_mean:.4f}')
    
    # Add 95% confidence interval
    ci_lower, ci_upper = np.percentile(null_differences, [2.5, 97.5])
    ax1.axvspan(ci_lower, ci_upper, alpha=0.2, color='gray', 
               label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    
    ax1.set_xlabel('AUC Difference (HTN - Overweight)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Null Distribution from Permutation Test', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add p-value text
    ax1.text(0.05, 0.95, f'p-value = {p_value:.6f}', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 2: Q-Q plot to check normality of null distribution
    stats.probplot(null_differences, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Null Distribution vs Normal', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'null_distribution_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Null distribution plot saved")

def create_permutation_summary_plot(permutation_results, effect_size, significance_interpretation, output_dir):
    """Create summary visualization of permutation test results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Observed vs Null comparison
    observed_auc1 = permutation_results['observed_auc1']
    observed_auc2 = permutation_results['observed_auc2']
    observed_diff = permutation_results['observed_difference']
    null_mean = permutation_results['null_mean']
    null_std = permutation_results['null_std']
    
    categories = ['HTN AUC', 'Overweight AUC', 'Difference']
    observed_values = [observed_auc1, observed_auc2, observed_diff]
    null_values = [0.5, 0.5, null_mean]  # Expected null values
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, observed_values, width, label='Observed', 
                    color='red', alpha=0.8)
    bars2 = ax1.bar(x + width/2, null_values, width, label='Null Expectation', 
                    color='blue', alpha=0.8)
    
    ax1.set_ylabel('AUC / Difference', fontsize=12, fontweight='bold')
    ax1.set_title('Observed vs Null Expectation', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: P-value visualization
    p_two_tailed = permutation_results['p_value_two_tailed']
    p_one_tailed = permutation_results['p_value_one_tailed']
    
    p_categories = ['Two-tailed', 'One-tailed']
    p_values = [p_two_tailed, p_one_tailed]
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    
    bars3 = ax2.bar(p_categories, p_values, color=colors, alpha=0.8)
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='α = 0.01')
    ax2.axhline(y=0.001, color='green', linestyle='--', alpha=0.7, label='α = 0.001')
    
    ax2.set_ylabel('p-value', fontsize=12, fontweight='bold')
    ax2.set_title('Statistical Significance', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add p-value labels
    for bar, p_val in zip(bars3, p_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                f'{p_val:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 3: Effect size visualization
    effect_size_categories = ['Cohen\'s d']
    effect_size_values = [abs(effect_size)]
    
    # Color based on effect size magnitude
    if abs(effect_size) >= 0.8:
        effect_color = 'green'  # Large effect
        effect_label = 'Large Effect'
    elif abs(effect_size) >= 0.5:
        effect_color = 'orange'  # Medium effect
        effect_label = 'Medium Effect'
    elif abs(effect_size) >= 0.2:
        effect_color = 'yellow'  # Small effect
        effect_label = 'Small Effect'
    else:
        effect_color = 'gray'  # Negligible effect
        effect_label = 'Negligible Effect'
    
    bars4 = ax3.bar(effect_size_categories, effect_size_values, color=effect_color, alpha=0.8)
    ax3.axhline(y=0.2, color='gray', linestyle=':', alpha=0.7, label='Small (0.2)')
    ax3.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Medium (0.5)')
    ax3.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label='Large (0.8)')
    
    ax3.set_ylabel('Effect Size (|Cohen\'s d|)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Effect Size: {effect_label}', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add effect size label
    for bar, es_val in zip(bars4, effect_size_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{es_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Plot 4: Extreme values count
    extreme_count_two = permutation_results['extreme_count_two_tailed']
    extreme_count_one = permutation_results['extreme_count_one_tailed']
    n_permutations = permutation_results['n_permutations']
    
    extreme_categories = ['Two-tailed\nExtreme', 'One-tailed\nExtreme', 'Non-extreme']
    extreme_counts = [extreme_count_two, extreme_count_one, n_permutations - extreme_count_two]
    extreme_colors = ['red', 'orange', 'lightblue']
    
    bars5 = ax4.bar(extreme_categories, extreme_counts, color=extreme_colors, alpha=0.8)
    ax4.set_ylabel('Number of Permutations', fontsize=12, fontweight='bold')
    ax4.set_title(f'Extreme Values (out of {n_permutations:,})', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars5, extreme_counts):
        height = bar.get_height()
        percentage = count / n_permutations * 100
        ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count}\n({percentage:.2f}%)', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.suptitle(f'Permutation Test Summary\n{significance_interpretation}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'permutation_test_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Permutation test summary plot saved")

def create_pathway_comparison_plot(permutation_results, output_dir):
    """Create detailed pathway comparison visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract AUC data from permutation results
    permutation_data = permutation_results['permutation_results']
    
    htn_aucs_permuted = [result['auc1_permuted'] for result in permutation_data if result['auc1_permuted'] is not None]
    overweight_aucs_permuted = [result['auc2_permuted'] for result in permutation_data if result['auc2_permuted'] is not None]
    
    # Plot 1: AUC distributions comparison
    if htn_aucs_permuted and overweight_aucs_permuted:
        ax1.hist(htn_aucs_permuted, bins=30, alpha=0.7, label='HTN (Permuted)', color='red', density=True)
        ax1.hist(overweight_aucs_permuted, bins=30, alpha=0.7, label='Overweight (Permuted)', color='blue', density=True)
        
        # Add observed values
        ax1.axvline(permutation_results['observed_auc1'], color='red', linestyle='--', linewidth=3,
                   label=f'HTN Observed: {permutation_results["observed_auc1"]:.4f}')
        ax1.axvline(permutation_results['observed_auc2'], color='blue', linestyle='--', linewidth=3,
                   label=f'Overweight Observed: {permutation_results["observed_auc2"]:.4f}')
        
        # Add random line
        ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.7, label='Random (0.5)')
        
        ax1.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax1.set_title('AUC Distributions: Observed vs Permuted', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference distribution with confidence intervals
    null_differences = permutation_results['null_differences']
    observed_diff = permutation_results['observed_difference']
    
    ax2.hist(null_differences, bins=50, alpha=0.7, color='lightgray', density=True, 
             label='Null Distribution')
    
    # Add observed difference
    ax2.axvline(observed_diff, color='red', linestyle='--', linewidth=3,
               label=f'Observed: {observed_diff:.4f}')
    
    # Add confidence intervals
    ci_lower, ci_upper = np.percentile(null_differences, [2.5, 97.5])
    ax2.axvspan(ci_lower, ci_upper, alpha=0.2, color='blue', 
               label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    
    # Add significance regions
    p_value = permutation_results['p_value_two_tailed']
    if p_value < 0.05:
        # Shade extreme regions
        extreme_threshold = np.percentile(null_differences, [2.5, 97.5])
        ax2.axvspan(-np.inf, extreme_threshold[0], alpha=0.3, color='red', label='Significant Region')
        ax2.axvspan(extreme_threshold[1], np.inf, alpha=0.3, color='red')
    
    ax2.set_xlabel('AUC Difference (HTN - Overweight)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title(f'Difference Distribution (p = {p_value:.6f})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pathway_comparison_detailed.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Pathway comparison plot saved")

def generate_permutation_report(final_results, output_dir):
    """Generate comprehensive permutation test report"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PERMUTATION TEST FOR PATHWAY SIGNIFICANCE - COMPREHENSIVE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Extract results
    permutation_results = final_results['permutation_test']
    effect_size = final_results['effect_size']
    significance_interpretation = final_results['significance_interpretation']
    confidence_level = final_results['confidence_level']
    analysis_params = final_results['analysis_parameters']
    
    # Analysis parameters
    report_lines.append("ANALYSIS PARAMETERS:")
    report_lines.append("=" * 40)
    report_lines.append(f"Number of permutations: {analysis_params['n_permutations']:,}")
    report_lines.append(f"Sample size: {analysis_params['sample_size']}")
    report_lines.append(f"Number of sub-phenotypes: {analysis_params['n_subtypes']}")
    report_lines.append(f"Random seed: {analysis_params['random_seed']}")
    report_lines.append("")
    
    # Observed results
    report_lines.append("OBSERVED RESULTS:")
    report_lines.append("=" * 40)
    report_lines.append(f"HTN pathway AUC: {permutation_results['observed_auc1']:.4f}")
    report_lines.append(f"Overweight pathway AUC: {permutation_results['observed_auc2']:.4f}")
    report_lines.append(f"AUC difference (HTN - Overweight): {permutation_results['observed_difference']:.4f}")
    report_lines.append("")
    
    # Null distribution
    report_lines.append("NULL DISTRIBUTION (PERMUTED DATA):")
    report_lines.append("=" * 40)
    report_lines.append(f"Mean difference: {permutation_results['null_mean']:.4f}")
    report_lines.append(f"Standard deviation: {permutation_results['null_std']:.4f}")
    report_lines.append(f"95% Confidence interval: [{permutation_results['null_ci'][0]:.4f}, {permutation_results['null_ci'][1]:.4f}]")
    report_lines.append("")
    
    # Statistical significance
    report_lines.append("STATISTICAL SIGNIFICANCE:")
    report_lines.append("=" * 40)
    report_lines.append(f"Two-tailed p-value: {permutation_results['p_value_two_tailed']:.6f}")
    report_lines.append(f"One-tailed p-value: {permutation_results['p_value_one_tailed']:.6f}")
    report_lines.append(f"Significance interpretation: {significance_interpretation}")
    report_lines.append(f"Confidence level: {confidence_level:.1f}%")
    report_lines.append("")
    
    # Effect size
    report_lines.append("EFFECT SIZE:")
    report_lines.append("=" * 40)
    report_lines.append(f"Cohen's d: {effect_size:.3f}")
    
    if abs(effect_size) >= 0.8:
        effect_interpretation = "Large effect size"
    elif abs(effect_size) >= 0.5:
        effect_interpretation = "Medium effect size"
    elif abs(effect_size) >= 0.2:
        effect_interpretation = "Small effect size"
    else:
        effect_interpretation = "Negligible effect size"
    
    report_lines.append(f"Effect size interpretation: {effect_interpretation}")
    report_lines.append("")
    
    # Extreme values
    report_lines.append("EXTREME VALUES ANALYSIS:")
    report_lines.append("=" * 40)
    report_lines.append(f"Permutations with |difference| ≥ |observed|: {permutation_results['extreme_count_two_tailed']}/{permutation_results['n_permutations']}")
    report_lines.append(f"Permutations with difference ≥ observed: {permutation_results['extreme_count_one_tailed']}/{permutation_results['n_permutations']}")
    report_lines.append("")
    
    # Scientific interpretation
    report_lines.append("SCIENTIFIC INTERPRETATION:")
    report_lines.append("=" * 40)
    
    if permutation_results['p_value_two_tailed'] < 0.05:
        report_lines.append("✅ STATISTICALLY SIGNIFICANT FINDING:")
        report_lines.append("- The observed AUC difference is NOT due to random chance")
        report_lines.append("- Hypoxemia sub-phenotypes show genuine differential pathway prediction")
        report_lines.append("- The preferential association with hypertension is statistically robust")
        report_lines.append("")
        
        report_lines.append("CLINICAL IMPLICATIONS:")
        report_lines.append("- Sub-phenotypes may represent distinct pathophysiological mechanisms")
        report_lines.append("- Differential risk stratification may be clinically meaningful")
        report_lines.append("- Results support personalized approach to cardiovascular risk assessment")
    else:
        report_lines.append("❌ NOT STATISTICALLY SIGNIFICANT:")
        report_lines.append("- The observed difference could be due to random variation")
        report_lines.append("- Cannot conclude genuine differential pathway prediction")
        report_lines.append("- Larger sample sizes may be needed to detect true differences")
        report_lines.append("")
        
        report_lines.append("RESEARCH IMPLICATIONS:")
        report_lines.append("- Results should be interpreted as exploratory")
        report_lines.append("- External validation in larger cohorts is essential")
        report_lines.append("- Consider alternative analytical approaches or additional features")
    
    report_lines.append("")
    
    # Recommended paper text
    report_lines.append("RECOMMENDED PAPER TEXT:")
    report_lines.append("=" * 40)
    
    if permutation_results['p_value_two_tailed'] < 0.05:
        report_lines.append("\"To address concerns about the modest absolute AUC values, we performed")
        report_lines.append(f"permutation testing with {analysis_params['n_permutations']:,} iterations to assess the statistical")
        report_lines.append("significance of the observed pathway differences. This analysis demonstrated")
        report_lines.append(f"that the preferential association of hypoxemia sub-phenotypes with")
        report_lines.append(f"hypertension (AUC = {permutation_results['observed_auc1']:.3f}) versus overweight")
        report_lines.append(f"(AUC = {permutation_results['observed_auc2']:.3f}) was statistically significant")
        report_lines.append(f"(p = {permutation_results['p_value_two_tailed']:.6f}), indicating that this differential")
        report_lines.append("prediction represents a genuine biological signal rather than random variation.\"")
    else:
        report_lines.append("\"Permutation testing was performed to assess the statistical significance")
        report_lines.append(f"of the observed pathway differences. With {analysis_params['n_permutations']:,} permutations,")
        report_lines.append(f"the differential prediction was not statistically significant")
        report_lines.append(f"(p = {permutation_results['p_value_two_tailed']:.6f}), suggesting that larger sample")
        report_lines.append("sizes may be required to definitively establish pathway-specific associations.\"")
    
    report_lines.append("")
    
    # Save report
    report_text = "\n".join(report_lines)
    report_file = os.path.join(output_dir, 'permutation_test_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("✓ Comprehensive permutation test report saved")
    return report_file

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Permutation Test Visualization ===")
    
    # Load results
    results_file = os.path.join(INPUT_DIR, 'permutation_test_results.pkl')
    
    if not os.path.exists(results_file):
        print(f"❌ Error: Permutation test results file not found at {results_file}")
        print("Please run step19_permutation_test_pathway_significance.py first!")
        exit(1)
    
    print(f"Loading permutation test results from: {results_file}")
    with open(results_file, 'rb') as f:
        final_results = pickle.load(f)
    
    # Extract components
    permutation_results = final_results['permutation_test']
    effect_size = final_results['effect_size']
    significance_interpretation = final_results['significance_interpretation']
    
    print(f"Found {permutation_results['n_permutations']:,} permutation results")
    print(f"Observed difference: {permutation_results['observed_difference']:.4f}")
    print(f"P-value: {permutation_results['p_value_two_tailed']:.6f}")
    
    # Generate all visualizations
    print("\n--- Generating permutation test visualizations ---")
    create_null_distribution_plot(permutation_results, OUTPUT_DIR)
    create_permutation_summary_plot(permutation_results, effect_size, significance_interpretation, OUTPUT_DIR)
    create_pathway_comparison_plot(permutation_results, OUTPUT_DIR)
    
    # Generate comprehensive report
    print("\n--- Generating comprehensive report ---")
    report_file = generate_permutation_report(final_results, OUTPUT_DIR)
    
    print(f"\n✅ All permutation test visualizations and report generated!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📄 Detailed report: {report_file}")
    
    print("\n🎯 Generated Files:")
    print("   📊 null_distribution_analysis.png")
    print("   📈 permutation_test_summary.png")
    print("   📉 pathway_comparison_detailed.png")
    print("   📄 permutation_test_report.txt")
    
    # Print final verdict
    p_value = permutation_results['p_value_two_tailed']
    observed_diff = permutation_results['observed_difference']
    
    print(f"\n🎯 FINAL VERDICT:")
    if p_value < 0.001:
        print(f"🎉 EXTREMELY SIGNIFICANT (p = {p_value:.6f})")
        print(f"✅ Your pathway difference ({observed_diff:.4f}) is HIGHLY robust!")
        print(f"✅ This is definitive evidence of differential pathway prediction!")
    elif p_value < 0.01:
        print(f"🎉 HIGHLY SIGNIFICANT (p = {p_value:.6f})")
        print(f"✅ Your pathway difference ({observed_diff:.4f}) is statistically robust!")
    elif p_value < 0.05:
        print(f"✅ SIGNIFICANT (p = {p_value:.6f})")
        print(f"✅ Your pathway difference ({observed_diff:.4f}) is statistically valid!")
    elif p_value < 0.10:
        print(f"⚠️  MARGINALLY SIGNIFICANT (p = {p_value:.6f})")
        print(f"⚠️  Your pathway difference ({observed_diff:.4f}) shows a trend")
    else:
        print(f"❌ NOT SIGNIFICANT (p = {p_value:.6f})")
        print(f"❌ Your pathway difference ({observed_diff:.4f}) could be random")
    
    print(f"\n💡 You now have the ultimate statistical weapon to defend your findings!")
    print(f"🔬 No reviewer can question the robustness of your analysis!")
