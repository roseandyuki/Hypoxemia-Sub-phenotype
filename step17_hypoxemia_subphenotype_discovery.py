# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 17: Hypoxemia Sub-phenotype Discovery ---
# 
# Purpose: Discover data-driven hypoxemia sub-phenotypes in the GSHC and 
# test their differential prediction of disease pathways
# 
# Scientific Question: Are there distinct nocturnal hypoxemia patterns in 
# the GSHC that predict different disease pathways (hypertension vs overweight)?
# 
# Clinical Significance: Could identify different mechanisms of sleep-related 
# cardiovascular and metabolic risk, leading to personalized interventions
# 
# Method: 
# 1. Engineer comprehensive hypoxemia features from SHHS data
# 2. Apply unsupervised clustering to identify sub-phenotypes
# 3. Validate clusters using multiple algorithms and stability analysis
# 4. Test differential pathway prediction (HTN vs overweight)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import umap
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pickle
from tqdm import tqdm

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

# Hypoxemia-related features to extract/engineer
HYPOXEMIA_FEATURES = [
    'min_spo2',      # Minimum SpO2 (existing)
    'avg_spo2',      # Average SpO2 (existing)
    'spo2_range',    # Range of SpO2 (engineered)
    'spo2_variability',  # Coefficient of variation (engineered)
    'hypoxemia_burden',  # Composite score (engineered)
    'desaturation_severity'  # Severity index (engineered)
]

# Clustering algorithms to test
CLUSTERING_CONFIGS = {
    'KMeans_2': {'algorithm': KMeans, 'params': {'n_clusters': 2, 'random_state': 42}},
    'KMeans_3': {'algorithm': KMeans, 'params': {'n_clusters': 3, 'random_state': 42}},
    'GMM_2': {'algorithm': GaussianMixture, 'params': {'n_components': 2, 'random_state': 42}},
    'GMM_3': {'algorithm': GaussianMixture, 'params': {'n_components': 3, 'random_state': 42}},
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

def create_gshc(df):
    """Create Gold-Standard Healthy Cohort"""
    gshc_criteria = (
        (df['bmi_v1'] < 25) & 
        (df['sbp_v1'] < 120) & 
        (df['dbp_v1'] < 80)
    )
    return df[gshc_criteria].copy()

# =============================================================================
# --- Feature Engineering Functions ---
# =============================================================================

def engineer_hypoxemia_features(df):
    """Engineer comprehensive hypoxemia features"""
    
    df_features = df.copy()
    
    # Basic range and variability
    df_features['spo2_range'] = df_features['avg_spo2'] - df_features['min_spo2']
    
    # Coefficient of variation (proxy for variability)
    # Since we don't have individual SpO2 measurements, use range/mean as proxy
    df_features['spo2_variability'] = df_features['spo2_range'] / df_features['avg_spo2']
    
    # Hypoxemia burden (composite score)
    # Higher burden = lower min_spo2 and higher variability
    df_features['hypoxemia_burden'] = (100 - df_features['min_spo2']) * (1 + df_features['spo2_variability'])
    
    # Desaturation severity index
    # Combines minimum SpO2 with respiratory disturbance
    df_features['desaturation_severity'] = (100 - df_features['min_spo2']) * np.log1p(df_features['rdi'])
    
    # Relative hypoxemia (compared to average)
    df_features['relative_hypoxemia'] = (df_features['avg_spo2'] - df_features['min_spo2']) / df_features['avg_spo2']
    
    return df_features

def get_hypoxemia_feature_matrix(df):
    """Extract hypoxemia feature matrix for clustering"""
    
    # Define all hypoxemia-related features
    hypox_features = [
        'min_spo2', 'avg_spo2', 'spo2_range', 'spo2_variability',
        'hypoxemia_burden', 'desaturation_severity', 'relative_hypoxemia'
    ]
    
    # Add RDI as it's closely related to hypoxemia
    hypox_features.append('rdi')
    
    # Extract feature matrix
    feature_matrix = df[hypox_features].copy()
    
    return feature_matrix, hypox_features

# =============================================================================
# --- Clustering Functions ---
# =============================================================================

def apply_clustering_algorithm(X, config_name, config):
    """Apply a single clustering algorithm"""
    
    algorithm_class = config['algorithm']
    params = config['params']
    
    # Fit the algorithm
    if algorithm_class == GaussianMixture:
        model = algorithm_class(**params)
        model.fit(X)
        labels = model.predict(X)
        # For GMM, also get probabilities
        probabilities = model.predict_proba(X)
        return labels, model, probabilities
    else:
        model = algorithm_class(**params)
        labels = model.fit_predict(X)
        return labels, model, None

def evaluate_clustering_quality(X, labels):
    """Evaluate clustering quality using multiple metrics"""
    
    if len(np.unique(labels)) < 2:
        return {'silhouette': -1, 'calinski_harabasz': 0, 'n_clusters': len(np.unique(labels))}
    
    try:
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        
        return {
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'n_clusters': len(np.unique(labels))
        }
    except:
        return {'silhouette': -1, 'calinski_harabasz': 0, 'n_clusters': len(np.unique(labels))}

def perform_comprehensive_clustering(X, feature_names):
    """Perform clustering with multiple algorithms"""
    
    results = {}
    
    print("Testing clustering algorithms...")
    for config_name, config in CLUSTERING_CONFIGS.items():
        print(f"  - {config_name}")
        
        try:
            if len(config['params']) > 2 and 'probabilities' in str(config):
                labels, model, probabilities = apply_clustering_algorithm(X, config_name, config)
                results[config_name] = {
                    'labels': labels,
                    'model': model,
                    'probabilities': probabilities,
                    'quality': evaluate_clustering_quality(X, labels)
                }
            else:
                labels, model, probabilities = apply_clustering_algorithm(X, config_name, config)
                results[config_name] = {
                    'labels': labels,
                    'model': model,
                    'probabilities': probabilities,
                    'quality': evaluate_clustering_quality(X, labels)
                }
        except Exception as e:
            print(f"    Error with {config_name}: {e}")
            results[config_name] = {
                'labels': None,
                'model': None,
                'probabilities': None,
                'quality': {'silhouette': -1, 'calinski_harabasz': 0, 'n_clusters': 0},
                'error': str(e)
            }
    
    return results

# =============================================================================
# --- Dimensionality Reduction and Visualization ---
# =============================================================================

def create_dimensionality_reduction(X, labels_dict):
    """Create multiple dimensionality reduction representations"""
    
    print("Creating dimensionality reductions...")
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
    X_tsne = tsne.fit_transform(X)
    
    # UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X)//3))
    X_umap = umap_reducer.fit_transform(X)
    
    return {
        'PCA': {'data': X_pca, 'explained_variance': pca.explained_variance_ratio_},
        't-SNE': {'data': X_tsne},
        'UMAP': {'data': X_umap}
    }

# =============================================================================
# --- Pathway Analysis Functions ---
# =============================================================================

def analyze_subphenotype_pathways(df, best_labels, output_dir):
    """Analyze differential pathway prediction by sub-phenotypes"""
    
    print("\nAnalyzing sub-phenotype pathway predictions...")
    
    # Add cluster labels to dataframe
    df_analysis = df.copy()
    df_analysis['hypoxemia_subtype'] = best_labels
    
    # Calculate outcomes
    df_analysis['Y_Hypertension'] = df_analysis.apply(has_hypertension, axis=1)
    df_analysis['Y_Overweight'] = df_analysis.apply(has_overweight, axis=1)
    df_analysis['Y_Composite'] = df_analysis.apply(has_transitioned, axis=1)
    
    # Remove missing outcomes
    df_analysis = df_analysis.dropna(subset=['Y_Hypertension', 'Y_Overweight', 'Y_Composite'])
    
    # Analyze each subtype
    subtype_analysis = {}
    
    for subtype in np.unique(best_labels):
        subtype_data = df_analysis[df_analysis['hypoxemia_subtype'] == subtype]
        
        if len(subtype_data) < 10:  # Skip if too few samples
            continue
            
        subtype_analysis[f'Subtype_{subtype}'] = {
            'n_participants': len(subtype_data),
            'htn_rate': subtype_data['Y_Hypertension'].mean(),
            'overweight_rate': subtype_data['Y_Overweight'].mean(),
            'composite_rate': subtype_data['Y_Composite'].mean(),
            'baseline_characteristics': {
                'age': subtype_data['age_v1'].mean(),
                'bmi': subtype_data['bmi_v1'].mean(),
                'sbp': subtype_data['sbp_v1'].mean(),
                'min_spo2': subtype_data['min_spo2'].mean(),
                'avg_spo2': subtype_data['avg_spo2'].mean(),
                'rdi': subtype_data['rdi'].mean()
            }
        }
    
    # Test differential pathway prediction
    pathway_results = test_differential_pathway_prediction(df_analysis)
    
    return subtype_analysis, pathway_results

def test_differential_pathway_prediction(df):
    """Test if subtypes differentially predict HTN vs overweight"""
    
    results = {}
    
    # Create dummy variables for subtypes
    subtypes = pd.get_dummies(df['hypoxemia_subtype'], prefix='subtype')
    
    # Test each outcome
    for outcome in ['Y_Hypertension', 'Y_Overweight', 'Y_Composite']:
        y = df[outcome]
        
        if y.sum() < 10:  # Skip if too few positive cases
            continue
            
        try:
            # Fit logistic regression
            lr = LogisticRegression(random_state=42, class_weight='balanced')
            cv_scores = cross_val_score(lr, subtypes, y, cv=5, scoring='roc_auc')
            
            # Fit full model for coefficients
            lr.fit(subtypes, y)
            
            results[outcome] = {
                'auc_mean': np.mean(cv_scores),
                'auc_std': np.std(cv_scores),
                'coefficients': dict(zip(subtypes.columns, lr.coef_[0])),
                'n_positive': int(y.sum()),
                'n_total': len(y)
            }
            
        except Exception as e:
            results[outcome] = {'error': str(e)}
    
    return results

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Hypoxemia Sub-phenotype Discovery ===")
    print("Discovering data-driven hypoxemia patterns in the GSHC...")
    
    # Setup output directory
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_hypoxemia_subphenotypes')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved to: {OUTPUT_DIR}")
    
    # Load data
    print("\n--- Loading and preparing data ---")
    base_df = load_and_map_data(DATA_FILES)
    base_df['Y_Transition'] = base_df.apply(has_transitioned, axis=1)
    full_cohort_df = base_df.dropna(subset=['Y_Transition']).copy()
    
    # Create GSHC
    gshc_df = create_gshc(full_cohort_df)
    print(f"GSHC size: {len(gshc_df)}")
    
    # Engineer hypoxemia features
    print("\n--- Engineering hypoxemia features ---")
    gshc_df = engineer_hypoxemia_features(gshc_df)
    
    # Extract hypoxemia feature matrix
    X_hypox, feature_names = get_hypoxemia_feature_matrix(gshc_df)
    
    # Remove rows with missing hypoxemia data
    complete_mask = X_hypox.notna().all(axis=1)
    X_hypox_complete = X_hypox[complete_mask]
    gshc_complete = gshc_df[complete_mask].copy()
    
    print(f"Complete hypoxemia data available for {len(X_hypox_complete)} participants")
    print(f"Hypoxemia features: {feature_names}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hypox_complete)
    
    # Perform clustering
    print(f"\n--- Performing clustering analysis ---")
    clustering_results = perform_comprehensive_clustering(X_scaled, feature_names)
    
    # Find best clustering solution
    best_config = None
    best_score = -1
    
    for config_name, results in clustering_results.items():
        if results['quality']['silhouette'] > best_score:
            best_score = results['quality']['silhouette']
            best_config = config_name
    
    print(f"\nBest clustering solution: {best_config}")
    print(f"Silhouette score: {best_score:.3f}")
    
    if best_config and clustering_results[best_config]['labels'] is not None:
        best_labels = clustering_results[best_config]['labels']
        
        # Create dimensionality reductions
        dim_reductions = create_dimensionality_reduction(X_scaled, {best_config: best_labels})
        
        # Analyze sub-phenotype pathways
        subtype_analysis, pathway_results = analyze_subphenotype_pathways(gshc_complete, best_labels, OUTPUT_DIR)
        
        # Save results
        results_file = os.path.join(OUTPUT_DIR, 'hypoxemia_subphenotype_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump({
                'clustering_results': clustering_results,
                'best_config': best_config,
                'best_labels': best_labels,
                'feature_names': feature_names,
                'dim_reductions': dim_reductions,
                'subtype_analysis': subtype_analysis,
                'pathway_results': pathway_results,
                'gshc_data': gshc_complete
            }, f)
        
        print(f"\nResults saved to: {results_file}")
        
        # Print summary
        print(f"\n--- Sub-phenotype Summary ---")
        for subtype_name, analysis in subtype_analysis.items():
            print(f"{subtype_name}: n={analysis['n_participants']}")
            print(f"  HTN rate: {analysis['htn_rate']:.1%}")
            print(f"  Overweight rate: {analysis['overweight_rate']:.1%}")
            print(f"  Min SpO2: {analysis['baseline_characteristics']['min_spo2']:.1f}%")
            print(f"  RDI: {analysis['baseline_characteristics']['rdi']:.1f}")
        
        print("\n🎯 Hypoxemia Sub-phenotype Discovery Complete!")
        print("📊 Next: Run visualization script to see the sub-phenotypes!")
        
    else:
        print("❌ No valid clustering solution found")
        print("This might indicate:")
        print("  - Insufficient sample size")
        print("  - No clear sub-phenotypes in the data")
        print("  - Need for different clustering approaches")
