# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 18B: Visualize Bootstrap Stability Analysis ---
# 
# Purpose: Generate comprehensive visualizations for bootstrap stability 
# analysis results
# 
# Usage: Run this after step18_bootstrap_stability_analysis.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from collections import Counter

plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# --- Configuration ---
# =============================================================================

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = '.'

INPUT_DIR = os.path.join(SCRIPT_DIR, 'output_bootstrap_stability')
OUTPUT_DIR = INPUT_DIR

# =============================================================================
# --- Visualization Functions ---
# =============================================================================

def create_clustering_stability_plots(clustering_stability, bootstrap_results, output_dir):
    """Create clustering stability visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Cluster count distribution
    cluster_counts = clustering_stability['cluster_count_distribution']
    counts = list(cluster_counts.keys())
    frequencies = list(cluster_counts.values())
    
    bars1 = ax1.bar(counts, frequencies, color='skyblue', alpha=0.8)
    ax1.set_xlabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Cluster Count Distribution Across Bootstrap Samples', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels
    total_samples = sum(frequencies)
    for bar, freq in zip(bars1, frequencies):
        height = bar.get_height()
        percentage = freq / total_samples * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Silhouette score distribution
    silhouette_scores = [result['silhouette_score'] for result in bootstrap_results 
                        if result['silhouette_score'] > -1]
    
    if silhouette_scores:
        ax2.hist(silhouette_scores, bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(silhouette_scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(silhouette_scores):.3f}')
        ax2.axvline(np.percentile(silhouette_scores, 2.5), color='orange', linestyle=':', 
                   label=f'95% CI: [{np.percentile(silhouette_scores, 2.5):.3f}, {np.percentile(silhouette_scores, 97.5):.3f}]')
        ax2.axvline(np.percentile(silhouette_scores, 97.5), color='orange', linestyle=':')
        
        ax2.set_xlabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Silhouette Score Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sample size distribution
    sample_sizes = [result['sample_size'] for result in bootstrap_results]
    
    ax3.hist(sample_sizes, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(sample_sizes), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(sample_sizes):.0f}')
    ax3.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Bootstrap Sample Size Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Stability summary
    stability_rate = clustering_stability['cluster_stability_rate']
    most_common_clusters = clustering_stability['most_common_n_clusters'][0]
    
    categories = ['Clustering\nStability', 'Silhouette\nQuality']
    values = [stability_rate, np.mean(silhouette_scores) if silhouette_scores else 0]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars4 = ax4.bar(categories, values, color=colors, alpha=0.8)
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('Overall Stability Summary', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add stability threshold line
    ax4.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
    ax4.legend()
    
    plt.suptitle(f'Bootstrap Clustering Stability Analysis\n'
                f'Most Common: {most_common_clusters} clusters ({stability_rate:.1%} stability)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_stability_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Clustering stability plots saved")

def create_individual_stability_plots(individual_stability, output_dir):
    """Create individual assignment stability visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract stability rates
    stability_rates = [ind['stability_rate'] for ind in individual_stability]
    original_labels = [ind['original_label'] for ind in individual_stability]
    
    # Plot 1: Overall stability distribution
    ax1.hist(stability_rates, bins=20, color='lightblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(stability_rates), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(stability_rates):.3f}')
    ax1.axvline(0.95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
    ax1.set_xlabel('Individual Stability Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Individuals', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Assignment Stability Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stability by original subtype
    unique_labels = sorted(list(set(original_labels)))
    stability_by_subtype = {}
    
    for label in unique_labels:
        if label >= 0:  # Valid label
            subtype_stabilities = [ind['stability_rate'] for ind in individual_stability 
                                 if ind['original_label'] == label]
            stability_by_subtype[f'Subtype {label}'] = subtype_stabilities
    
    if stability_by_subtype:
        box_data = list(stability_by_subtype.values())
        box_labels = list(stability_by_subtype.keys())
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = plt.cm.Set1(np.linspace(0, 1, len(box_data)))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
        ax2.set_ylabel('Stability Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Stability by Original Sub-phenotype', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: High vs Low stability individuals
    high_stability = sum([1 for rate in stability_rates if rate >= 0.95])
    low_stability = len(stability_rates) - high_stability
    
    categories = ['High Stability\n(≥95%)', 'Low Stability\n(<95%)']
    counts = [high_stability, low_stability]
    colors = ['green', 'red']
    
    bars3 = ax3.bar(categories, counts, color=colors, alpha=0.7)
    ax3.set_ylabel('Number of Individuals', fontsize=12, fontweight='bold')
    ax3.set_title('Stability Classification', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars3, counts):
        height = bar.get_height()
        percentage = count / total * 100
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Bootstrap appearance frequency
    appearances = [ind['n_bootstrap_appearances'] for ind in individual_stability]
    
    ax4.hist(appearances, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(appearances), color='darkorange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(appearances):.0f}')
    ax4.set_xlabel('Bootstrap Appearances', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Individuals', fontsize=12, fontweight='bold')
    ax4.set_title('Bootstrap Sample Appearance Frequency', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Individual Assignment Stability Analysis\n'
                f'Mean Stability: {np.mean(stability_rates):.1%}, High Stability: {high_stability}/{total} ({high_stability/total:.1%})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_stability_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Individual stability plots saved")

def create_pathway_stability_plots(pathway_stability, bootstrap_results, output_dir):
    """Create pathway conclusion stability visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract AUC data
    htn_aucs = []
    overweight_aucs = []
    
    for result in bootstrap_results:
        pathway_results = result['pathway_results']
        htn_result = pathway_results.get('Y_Hypertension')
        overweight_result = pathway_results.get('Y_Overweight')
        
        if htn_result:
            htn_aucs.append(htn_result['auc'])
        if overweight_result:
            overweight_aucs.append(overweight_result['auc'])
    
    # Plot 1: AUC distributions
    if htn_aucs and overweight_aucs:
        ax1.hist(htn_aucs, bins=20, alpha=0.7, label='Hypertension', color='red')
        ax1.hist(overweight_aucs, bins=20, alpha=0.7, label='Overweight', color='blue')
        ax1.axvline(np.mean(htn_aucs), color='red', linestyle='--', 
                   label=f'HTN Mean: {np.mean(htn_aucs):.3f}')
        ax1.axvline(np.mean(overweight_aucs), color='blue', linestyle='--', 
                   label=f'OW Mean: {np.mean(overweight_aucs):.3f}')
        ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.7, label='Random')
        
        ax1.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Pathway Prediction AUC Distributions', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pathway preference stability
    htn_stronger_rate = pathway_stability['htn_stronger_rate']
    overweight_stronger_rate = pathway_stability['overweight_stronger_rate']
    
    categories = ['HTN Stronger', 'Overweight Stronger']
    rates = [htn_stronger_rate, overweight_stronger_rate]
    colors = ['red', 'blue']
    
    bars2 = ax2.bar(categories, rates, color=colors, alpha=0.7)
    ax2.set_ylabel('Proportion of Bootstrap Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Pathway Preference Stability', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars2, rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Add stability threshold
    ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% Threshold')
    ax2.legend()
    
    # Plot 3: AUC difference distribution
    if htn_aucs and overweight_aucs:
        # Match pairs
        min_len = min(len(htn_aucs), len(overweight_aucs))
        auc_differences = [htn_aucs[i] - overweight_aucs[i] for i in range(min_len)]
        
        ax3.hist(auc_differences, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='gray', linestyle='-', alpha=0.7, label='No Difference')
        ax3.axvline(np.mean(auc_differences), color='purple', linestyle='--', linewidth=2,
                   label=f'Mean Diff: {np.mean(auc_differences):.3f}')
        
        ax3.set_xlabel('AUC Difference (HTN - Overweight)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('AUC Difference Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confidence intervals
    outcomes = ['Hypertension', 'Overweight']
    means = [pathway_stability['htn_auc_mean'], pathway_stability['overweight_auc_mean']]
    cis_lower = [pathway_stability['htn_auc_ci'][0], pathway_stability['overweight_auc_ci'][0]]
    cis_upper = [pathway_stability['htn_auc_ci'][1], pathway_stability['overweight_auc_ci'][1]]
    
    x_pos = np.arange(len(outcomes))
    bars4 = ax4.bar(x_pos, means, color=['red', 'blue'], alpha=0.7)
    ax4.errorbar(x_pos, means, yerr=[np.array(means) - np.array(cis_lower), 
                                    np.array(cis_upper) - np.array(means)], 
                fmt='none', color='black', capsize=5, capthick=2)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(outcomes)
    ax4.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax4.set_title('Pathway Prediction Performance (95% CI)', fontsize=14, fontweight='bold')
    ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Random')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, ci_lower, ci_upper) in enumerate(zip(bars4, means, cis_lower, cis_upper)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{mean:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.suptitle(f'Pathway Prediction Stability Analysis\n'
                f'Valid Comparisons: {pathway_stability["valid_comparisons"]}, '
                f'HTN Preference: {htn_stronger_rate:.1%}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pathway_stability_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Pathway stability plots saved")

def generate_stability_report(stability_results, output_dir):
    """Generate comprehensive stability report"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BOOTSTRAP STABILITY ANALYSIS - COMPREHENSIVE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Bootstrap summary
    n_bootstrap = stability_results['n_bootstrap']
    n_successful = stability_results['n_successful']
    success_rate = n_successful / n_bootstrap
    
    report_lines.append("BOOTSTRAP ANALYSIS SUMMARY:")
    report_lines.append("=" * 40)
    report_lines.append(f"Total bootstrap iterations: {n_bootstrap}")
    report_lines.append(f"Successful iterations: {n_successful} ({success_rate:.1%})")
    report_lines.append("")
    
    # Clustering stability
    clustering_stability = stability_results['clustering_stability']
    
    report_lines.append("CLUSTERING STABILITY:")
    report_lines.append("=" * 40)
    report_lines.append(f"Most common cluster count: {clustering_stability['most_common_n_clusters'][0]}")
    report_lines.append(f"Clustering stability rate: {clustering_stability['cluster_stability_rate']:.1%}")
    report_lines.append(f"Mean silhouette score: {clustering_stability['silhouette_mean']:.3f} ± {clustering_stability['silhouette_std']:.3f}")
    report_lines.append(f"Silhouette 95% CI: [{clustering_stability['silhouette_ci'][0]:.3f}, {clustering_stability['silhouette_ci'][1]:.3f}]")
    
    # Stability interpretation
    if clustering_stability['cluster_stability_rate'] >= 0.95:
        report_lines.append("✓ EXCELLENT clustering stability (≥95%)")
    elif clustering_stability['cluster_stability_rate'] >= 0.80:
        report_lines.append("✓ GOOD clustering stability (≥80%)")
    elif clustering_stability['cluster_stability_rate'] >= 0.60:
        report_lines.append("⚠ MODERATE clustering stability (≥60%)")
    else:
        report_lines.append("❌ POOR clustering stability (<60%)")
    
    report_lines.append("")
    
    # Individual stability
    individual_stability = stability_results['individual_stability']
    stability_rates = [ind['stability_rate'] for ind in individual_stability]
    high_stability_count = sum([1 for rate in stability_rates if rate >= 0.95])
    
    report_lines.append("INDIVIDUAL ASSIGNMENT STABILITY:")
    report_lines.append("=" * 40)
    report_lines.append(f"Mean individual stability: {np.mean(stability_rates):.1%}")
    report_lines.append(f"High stability individuals (≥95%): {high_stability_count}/{len(stability_rates)} ({high_stability_count/len(stability_rates):.1%})")
    report_lines.append(f"Stability range: {np.min(stability_rates):.1%} - {np.max(stability_rates):.1%}")
    report_lines.append("")
    
    # Pathway stability
    pathway_stability = stability_results['pathway_stability']
    
    report_lines.append("PATHWAY PREDICTION STABILITY:")
    report_lines.append("=" * 40)
    report_lines.append(f"Valid pathway comparisons: {pathway_stability['valid_comparisons']}")
    report_lines.append(f"HTN pathway stronger: {pathway_stability['htn_stronger_rate']:.1%}")
    report_lines.append(f"Overweight pathway stronger: {pathway_stability['overweight_stronger_rate']:.1%}")
    report_lines.append(f"HTN prediction AUC: {pathway_stability['htn_auc_mean']:.3f} ± {pathway_stability['htn_auc_std']:.3f}")
    report_lines.append(f"HTN AUC 95% CI: [{pathway_stability['htn_auc_ci'][0]:.3f}, {pathway_stability['htn_auc_ci'][1]:.3f}]")
    report_lines.append(f"Overweight prediction AUC: {pathway_stability['overweight_auc_mean']:.3f} ± {pathway_stability['overweight_auc_std']:.3f}")
    report_lines.append(f"Overweight AUC 95% CI: [{pathway_stability['overweight_auc_ci'][0]:.3f}, {pathway_stability['overweight_auc_ci'][1]:.3f}]")
    report_lines.append("")
    
    # Overall conclusions
    report_lines.append("OVERALL STABILITY ASSESSMENT:")
    report_lines.append("=" * 40)
    
    # Determine overall stability
    clustering_stable = clustering_stability['cluster_stability_rate'] >= 0.80
    individual_stable = np.mean(stability_rates) >= 0.70
    pathway_stable = pathway_stability['htn_stronger_rate'] >= 0.80 or pathway_stability['overweight_stronger_rate'] >= 0.80
    
    if clustering_stable and individual_stable and pathway_stable:
        report_lines.append("✅ ROBUST FINDINGS: All stability criteria met")
        report_lines.append("- Clustering solution is stable across bootstrap samples")
        report_lines.append("- Individual assignments are reliable")
        report_lines.append("- Pathway conclusions are consistent")
    elif clustering_stable and individual_stable:
        report_lines.append("✅ STABLE SUB-PHENOTYPES: Clustering is robust")
        report_lines.append("⚠ PATHWAY CONCLUSIONS: Need cautious interpretation")
    elif clustering_stable:
        report_lines.append("✅ CLUSTERING STRUCTURE: Basic sub-phenotypes exist")
        report_lines.append("⚠ INDIVIDUAL ASSIGNMENTS: High uncertainty")
        report_lines.append("⚠ PATHWAY CONCLUSIONS: Need cautious interpretation")
    else:
        report_lines.append("❌ UNSTABLE FINDINGS: Results may be due to chance")
        report_lines.append("- Consider larger sample size or different approach")
    
    report_lines.append("")
    
    # Clinical implications
    report_lines.append("CLINICAL IMPLICATIONS:")
    report_lines.append("=" * 40)
    if clustering_stable:
        report_lines.append("- Sub-phenotype structure appears to be real, not random")
        report_lines.append("- May represent distinct biological mechanisms")
        if pathway_stable:
            report_lines.append("- Differential pathway prediction is reliable")
            report_lines.append("- Could guide personalized risk assessment")
        else:
            report_lines.append("- Pathway differences need further validation")
    else:
        report_lines.append("- Sub-phenotype findings are not sufficiently stable")
        report_lines.append("- Larger studies needed before clinical application")
    
    report_lines.append("")
    
    # Save report
    report_text = "\n".join(report_lines)
    report_file = os.path.join(output_dir, 'bootstrap_stability_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("✓ Comprehensive stability report saved")
    return report_file

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Bootstrap Stability Analysis Visualization ===")
    
    # Load results
    results_file = os.path.join(INPUT_DIR, 'bootstrap_stability_results.pkl')
    
    if not os.path.exists(results_file):
        print(f"❌ Error: Stability results file not found at {results_file}")
        print("Please run step18_bootstrap_stability_analysis.py first!")
        exit(1)
    
    print(f"Loading stability results from: {results_file}")
    with open(results_file, 'rb') as f:
        stability_results = pickle.load(f)
    
    # Extract components
    bootstrap_results = stability_results['bootstrap_results']
    clustering_stability = stability_results['clustering_stability']
    individual_stability = stability_results['individual_stability']
    pathway_stability = stability_results['pathway_stability']
    
    print(f"Found {len(bootstrap_results)} successful bootstrap iterations")
    
    # Generate all visualizations
    print("\n--- Generating stability visualizations ---")
    create_clustering_stability_plots(clustering_stability, bootstrap_results, OUTPUT_DIR)
    create_individual_stability_plots(individual_stability, OUTPUT_DIR)
    create_pathway_stability_plots(pathway_stability, bootstrap_results, OUTPUT_DIR)
    
    # Generate comprehensive report
    print("\n--- Generating comprehensive stability report ---")
    report_file = generate_stability_report(stability_results, OUTPUT_DIR)
    
    print(f"\n✅ All stability visualizations and report generated!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📄 Detailed report: {report_file}")
    
    print("\n🎯 Generated Files:")
    print("   📊 clustering_stability_analysis.png")
    print("   👤 individual_stability_analysis.png")
    print("   📈 pathway_stability_analysis.png")
    print("   📄 bootstrap_stability_report.txt")
    
    # Print quick summary
    clustering_rate = clustering_stability['cluster_stability_rate']
    individual_rate = np.mean([ind['stability_rate'] for ind in individual_stability])
    pathway_rate = max(pathway_stability['htn_stronger_rate'], pathway_stability['overweight_stronger_rate'])
    
    print(f"\n📊 Quick Stability Summary:")
    print(f"   🔄 Clustering Stability: {clustering_rate:.1%}")
    print(f"   👤 Individual Stability: {individual_rate:.1%}")
    print(f"   🎯 Pathway Stability: {pathway_rate:.1%}")
    
    if clustering_rate >= 0.95 and individual_rate >= 0.80 and pathway_rate >= 0.80:
        print(f"\n🎉 EXCELLENT: Your findings are highly stable!")
    elif clustering_rate >= 0.80:
        print(f"\n✅ GOOD: Your sub-phenotypes are stable!")
    else:
        print(f"\n⚠️  CAUTION: Results need careful interpretation")
    
    print(f"\n💡 This analysis provides the internal validation you need!")
    print(f"🔬 You can now confidently report the stability of your discoveries!")
