# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 17B: Visualize Hypoxemia Sub-phenotypes ---
# 
# Purpose: Generate comprehensive visualizations for hypoxemia sub-phenotype 
# discovery results
# 
# Usage: Run this after step17_hypoxemia_subphenotype_discovery.py
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

INPUT_DIR = os.path.join(SCRIPT_DIR, 'output_hypoxemia_subphenotypes')
OUTPUT_DIR = INPUT_DIR

# =============================================================================
# --- Visualization Functions ---
# =============================================================================

def create_clustering_overview(clustering_results, output_dir):
    """Create overview of all clustering results"""
    
    # Extract quality metrics
    configs = []
    silhouette_scores = []
    calinski_scores = []
    n_clusters = []
    
    for config_name, results in clustering_results.items():
        if 'error' not in results:
            configs.append(config_name)
            silhouette_scores.append(results['quality']['silhouette'])
            calinski_scores.append(results['quality']['calinski_harabasz'])
            n_clusters.append(results['quality']['n_clusters'])
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Silhouette scores
    bars1 = ax1.bar(configs, silhouette_scores, color='skyblue', alpha=0.8)
    ax1.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax1.set_title('Clustering Quality: Silhouette Score', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars1, silhouette_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Calinski-Harabasz scores
    bars2 = ax2.bar(configs, calinski_scores, color='lightcoral', alpha=0.8)
    ax2.set_ylabel('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
    ax2.set_title('Clustering Quality: Calinski-Harabasz Score', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars2, calinski_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_quality_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Clustering quality comparison saved")

def create_dimensionality_reduction_plots(dim_reductions, best_labels, output_dir):
    """Create dimensionality reduction visualizations"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['PCA', 't-SNE', 'UMAP']
    colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(best_labels))))
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        data = dim_reductions[method]['data']
        
        # Plot each cluster
        for cluster_id in np.unique(best_labels):
            mask = best_labels == cluster_id
            ax.scatter(data[mask, 0], data[mask, 1], 
                      c=[colors[cluster_id]], 
                      label=f'Subtype {cluster_id}',
                      alpha=0.7, s=50)
        
        ax.set_title(f'{method} Visualization', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method} Component 1', fontsize=12)
        ax.set_ylabel(f'{method} Component 2', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add explained variance for PCA
        if method == 'PCA':
            var_explained = dim_reductions[method]['explained_variance']
            ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimensionality_reduction_plots.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Dimensionality reduction plots saved")

def create_subphenotype_characteristics(subtype_analysis, output_dir):
    """Create sub-phenotype characteristic comparison"""
    
    # Extract data for plotting
    subtypes = list(subtype_analysis.keys())
    characteristics = ['min_spo2', 'avg_spo2', 'rdi', 'age', 'bmi', 'sbp']
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, char in enumerate(characteristics):
        ax = axes[idx]
        
        values = []
        labels = []
        
        for subtype in subtypes:
            if char in subtype_analysis[subtype]['baseline_characteristics']:
                values.append(subtype_analysis[subtype]['baseline_characteristics'][char])
                labels.append(subtype.replace('_', ' '))
        
        bars = ax.bar(labels, values, color=plt.cm.Set1(np.linspace(0, 1, len(labels))), alpha=0.8)
        
        # Customize plot
        ax.set_title(f'{char.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Sub-phenotype Baseline Characteristics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subphenotype_characteristics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Sub-phenotype characteristics plot saved")

def create_pathway_analysis_plots(subtype_analysis, pathway_results, output_dir):
    """Create pathway analysis visualizations"""
    
    # Extract outcome rates
    subtypes = list(subtype_analysis.keys())
    outcomes = ['htn_rate', 'overweight_rate', 'composite_rate']
    outcome_labels = ['Hypertension', 'Overweight', 'Composite']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Outcome rates by subtype
    x = np.arange(len(subtypes))
    width = 0.25
    
    for idx, (outcome, label) in enumerate(zip(outcomes, outcome_labels)):
        rates = [subtype_analysis[subtype][outcome] for subtype in subtypes]
        bars = ax1.bar(x + idx*width, rates, width, label=label, alpha=0.8)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Sub-phenotype', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Outcome Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Outcome Rates by Hypoxemia Sub-phenotype', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([s.replace('_', ' ') for s in subtypes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pathway prediction performance
    if pathway_results:
        outcomes_pred = []
        aucs = []
        auc_stds = []
        
        for outcome, results in pathway_results.items():
            if 'error' not in results:
                outcomes_pred.append(outcome.replace('Y_', ''))
                aucs.append(results['auc_mean'])
                auc_stds.append(results['auc_std'])
        
        if outcomes_pred:
            bars = ax2.bar(outcomes_pred, aucs, yerr=auc_stds, capsize=5, 
                          color='lightgreen', alpha=0.8)
            
            ax2.set_ylabel('Cross-Validation AUC', fontsize=12, fontweight='bold')
            ax2.set_title('Sub-phenotype Pathway Prediction', fontsize=14, fontweight='bold')
            ax2.set_ylim(0.4, 1.0)
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add value labels
            for bar, auc, std in zip(bars, aucs, auc_stds):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                        f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pathway_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Pathway analysis plots saved")

def create_hypoxemia_distribution_plots(gshc_data, best_labels, feature_names, output_dir):
    """Create hypoxemia feature distribution plots by subtype"""
    
    # Select key hypoxemia features for visualization
    key_features = ['min_spo2', 'avg_spo2', 'spo2_range', 'hypoxemia_burden', 'rdi']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(best_labels))))
    
    for idx, feature in enumerate(key_features):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Create violin plots for each subtype
        data_by_subtype = []
        subtype_labels = []
        
        for subtype in np.unique(best_labels):
            mask = best_labels == subtype
            if feature in gshc_data.columns:
                feature_data = gshc_data[mask][feature].dropna()
                if len(feature_data) > 0:
                    data_by_subtype.append(feature_data)
                    subtype_labels.append(f'Subtype {subtype}')
        
        if data_by_subtype:
            parts = ax.violinplot(data_by_subtype, positions=range(len(data_by_subtype)))
            
            # Color the violin plots
            for pc, color in zip(parts['bodies'], colors[:len(data_by_subtype)]):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(subtype_labels)))
            ax.set_xticklabels(subtype_labels)
            ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{feature.replace("_", " ").title()} Distribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(key_features), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Hypoxemia Feature Distributions by Sub-phenotype', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypoxemia_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Hypoxemia distribution plots saved")

def generate_subphenotype_report(results_data, output_dir):
    """Generate comprehensive sub-phenotype report"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HYPOXEMIA SUB-PHENOTYPE DISCOVERY - COMPREHENSIVE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Clustering summary
    report_lines.append("CLUSTERING ANALYSIS SUMMARY:")
    report_lines.append("=" * 40)
    
    best_config = results_data['best_config']
    clustering_results = results_data['clustering_results']
    
    if best_config and best_config in clustering_results:
        best_result = clustering_results[best_config]
        report_lines.append(f"Best clustering method: {best_config}")
        report_lines.append(f"Silhouette score: {best_result['quality']['silhouette']:.3f}")
        report_lines.append(f"Number of clusters: {best_result['quality']['n_clusters']}")
        report_lines.append("")
    
    # Sub-phenotype characteristics
    subtype_analysis = results_data['subtype_analysis']
    
    report_lines.append("SUB-PHENOTYPE CHARACTERISTICS:")
    report_lines.append("=" * 40)
    
    for subtype_name, analysis in subtype_analysis.items():
        report_lines.append(f"{subtype_name.upper()}:")
        report_lines.append(f"  - Sample size: {analysis['n_participants']}")
        report_lines.append(f"  - Hypertension rate: {analysis['htn_rate']:.1%}")
        report_lines.append(f"  - Overweight rate: {analysis['overweight_rate']:.1%}")
        report_lines.append(f"  - Composite outcome rate: {analysis['composite_rate']:.1%}")
        report_lines.append(f"  - Mean age: {analysis['baseline_characteristics']['age']:.1f} years")
        report_lines.append(f"  - Mean BMI: {analysis['baseline_characteristics']['bmi']:.1f} kg/m²")
        report_lines.append(f"  - Mean min SpO2: {analysis['baseline_characteristics']['min_spo2']:.1f}%")
        report_lines.append(f"  - Mean avg SpO2: {analysis['baseline_characteristics']['avg_spo2']:.1f}%")
        report_lines.append(f"  - Mean RDI: {analysis['baseline_characteristics']['rdi']:.1f} events/hr")
        report_lines.append("")
    
    # Pathway prediction results
    pathway_results = results_data['pathway_results']
    
    if pathway_results:
        report_lines.append("PATHWAY PREDICTION ANALYSIS:")
        report_lines.append("=" * 40)
        
        for outcome, results in pathway_results.items():
            if 'error' not in results:
                report_lines.append(f"{outcome.replace('Y_', '').upper()}:")
                report_lines.append(f"  - Prediction AUC: {results['auc_mean']:.3f} ± {results['auc_std']:.3f}")
                report_lines.append(f"  - Positive cases: {results['n_positive']}/{results['n_total']} ({results['n_positive']/results['n_total']:.1%})")
                
                # Show subtype coefficients
                report_lines.append("  - Subtype associations:")
                for subtype, coef in results['coefficients'].items():
                    direction = "↑" if coef > 0 else "↓"
                    report_lines.append(f"    {subtype}: {coef:.3f} {direction}")
                report_lines.append("")
    
    # Clinical implications
    report_lines.append("CLINICAL IMPLICATIONS:")
    report_lines.append("=" * 40)
    report_lines.append("- Distinct hypoxemia patterns may exist within healthy populations")
    report_lines.append("- Different sub-phenotypes may predict different disease pathways")
    report_lines.append("- Personalized risk stratification based on hypoxemia patterns")
    report_lines.append("- Potential for targeted interventions based on sub-phenotype")
    report_lines.append("")
    
    # Save report
    report_text = "\n".join(report_lines)
    report_file = os.path.join(output_dir, 'hypoxemia_subphenotype_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("✓ Comprehensive sub-phenotype report saved")
    return report_file

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Hypoxemia Sub-phenotype Visualization ===")
    
    # Load results
    results_file = os.path.join(INPUT_DIR, 'hypoxemia_subphenotype_results.pkl')
    
    if not os.path.exists(results_file):
        print(f"❌ Error: Results file not found at {results_file}")
        print("Please run step17_hypoxemia_subphenotype_discovery.py first!")
        exit(1)
    
    print(f"Loading results from: {results_file}")
    with open(results_file, 'rb') as f:
        results_data = pickle.load(f)
    
    # Extract key components
    clustering_results = results_data['clustering_results']
    best_config = results_data['best_config']
    best_labels = results_data['best_labels']
    dim_reductions = results_data['dim_reductions']
    subtype_analysis = results_data['subtype_analysis']
    pathway_results = results_data['pathway_results']
    gshc_data = results_data['gshc_data']
    feature_names = results_data['feature_names']
    
    print(f"Found {len(np.unique(best_labels))} sub-phenotypes")
    
    # Generate all visualizations
    print("\n--- Generating visualizations ---")
    create_clustering_overview(clustering_results, OUTPUT_DIR)
    create_dimensionality_reduction_plots(dim_reductions, best_labels, OUTPUT_DIR)
    create_subphenotype_characteristics(subtype_analysis, OUTPUT_DIR)
    create_pathway_analysis_plots(subtype_analysis, pathway_results, OUTPUT_DIR)
    create_hypoxemia_distribution_plots(gshc_data, best_labels, feature_names, OUTPUT_DIR)
    
    # Generate comprehensive report
    print("\n--- Generating comprehensive report ---")
    report_file = generate_subphenotype_report(results_data, OUTPUT_DIR)
    
    print(f"\n✅ All visualizations and report generated!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📄 Detailed report: {report_file}")
    
    print("\n🎯 Generated Files:")
    print("   📊 clustering_quality_comparison.png")
    print("   🎨 dimensionality_reduction_plots.png")
    print("   📋 subphenotype_characteristics.png")
    print("   📈 pathway_analysis.png")
    print("   📉 hypoxemia_distributions.png")
    print("   📄 hypoxemia_subphenotype_report.txt")
    
    print("\n🔬 Key Findings to Look For:")
    print("   - Are there distinct clusters in the dimensionality reduction plots?")
    print("   - Do sub-phenotypes show different baseline characteristics?")
    print("   - Do sub-phenotypes differentially predict HTN vs overweight?")
    print("   - What are the hypoxemia patterns that define each sub-phenotype?")

    print("\n💡 Next Steps:")
    print("   - Run step18_bootstrap_stability_analysis.py for internal validation")
    print("   - Use bootstrap resampling to test robustness of findings")
    print("   - Generate confidence intervals for all key metrics")
