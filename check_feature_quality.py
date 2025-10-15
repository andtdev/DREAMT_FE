"""
Script to check feature quality in the dataset
"""
import pandas as pd
import numpy as np
import glob

# Load all feature files
feature_files = glob.glob('dataset_sample/features_df/*.csv')
print(f"Found {len(feature_files)} feature files\n")

# Load first file to get feature names
df = pd.read_csv(feature_files[0])
features = [col for col in df.columns if col not in ['Sleep_Stage', 'timestamp_start', 'sid', 'Obstructive_Apnea', 'Central_Apnea', 'Hypopnea', 'Multiple_Events', 'artifact']]

print(f"Total features: {len(features)}\n")

# Check for issues
issues = {
    'constant_features': [],
    'high_nan_features': [],
    'infinite_features': [],
    'high_correlation_pairs': []
}

# Load and concatenate all data
all_data = []
for f in feature_files:
    df = pd.read_csv(f)
    all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)
print(f"Total samples: {len(full_df)}\n")

print("=" * 60)
print("FEATURE QUALITY REPORT")
print("=" * 60)

# 1. Check for constant features
print("\n1. CONSTANT FEATURES (std = 0):")
for feat in features:
    if feat in full_df.columns:
        if full_df[feat].std() == 0 or pd.isna(full_df[feat].std()):
            issues['constant_features'].append(feat)
            print(f"   - {feat}")

if not issues['constant_features']:
    print("   ✓ No constant features found")

# 2. Check for high NaN rates
print("\n2. FEATURES WITH >20% MISSING VALUES:")
nan_threshold = 0.2
for feat in features:
    if feat in full_df.columns:
        nan_rate = full_df[feat].isna().sum() / len(full_df)
        if nan_rate > nan_threshold:
            issues['high_nan_features'].append((feat, nan_rate))
            print(f"   - {feat}: {nan_rate*100:.1f}% missing")

if not issues['high_nan_features']:
    print("   ✓ No features with >20% missing values")

# 3. Check for infinite values
print("\n3. FEATURES WITH INFINITE VALUES:")
numeric_cols = full_df[features].select_dtypes(include=[np.number]).columns
for feat in numeric_cols:
    inf_count = np.isinf(full_df[feat]).sum()
    if inf_count > 0:
        issues['infinite_features'].append((feat, inf_count))
        print(f"   - {feat}: {inf_count} infinite values ({inf_count/len(full_df)*100:.2f}%)")

if not issues['infinite_features']:
    print("   ✓ No infinite values found")

# 4. Check for highly correlated features
print("\n4. HIGHLY CORRELATED FEATURE PAIRS (|corr| > 0.95):")
numeric_df = full_df[features].select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()

# Find pairs with correlation > 0.95 (excluding self-correlation)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            high_corr_pairs.append((feat1, feat2, corr_val))
            if len(high_corr_pairs) <= 20:  # Limit output
                print(f"   - {feat1} <-> {feat2}: {corr_val:.3f}")

if high_corr_pairs:
    print(f"\n   Total: {len(high_corr_pairs)} highly correlated pairs")
    issues['high_correlation_pairs'] = high_corr_pairs
else:
    print("   ✓ No highly correlated feature pairs found")

# 5. Summary statistics
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total features analyzed: {len(features)}")
print(f"Constant features: {len(issues['constant_features'])}")
print(f"High NaN features (>20%): {len(issues['high_nan_features'])}")
print(f"Features with infinite values: {len(issues['infinite_features'])}")
print(f"Highly correlated pairs (>0.95): {len(issues['high_correlation_pairs'])}")

# Calculate how many "good" features remain
problematic_features = set(issues['constant_features'])
problematic_features.update([f[0] for f in issues['high_nan_features']])
problematic_features.update([f[0] for f in issues['infinite_features']])

print(f"\nProblematic features to consider removing: {len(problematic_features)}")
print(f"Clean features remaining: {len(features) - len(problematic_features)}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
if problematic_features:
    print("1. Remove constant and high-NaN features")
    print("2. Replace infinite values with NaN, then impute")
    print("3. Consider removing one feature from each highly correlated pair")
else:
    print("✓ Feature quality looks good!")
    print("  The issue is likely model-related, not data quality")
    print("  → Try feature selection based on importance")
    print("  → Try class-specific models")
    print("  → Consider that 5-class may be fundamentally hard")

print("\n" + "=" * 60)

