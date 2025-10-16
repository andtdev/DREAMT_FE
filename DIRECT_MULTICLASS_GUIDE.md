# Direct Multiclass Classification Guide

## Quick Start

```bash
python experiments_multiclass_direct.py
```

## What This Script Does

Instead of the hierarchical approach (3 separate models), this uses a **single multiclass LightGBM model** that predicts all 5 sleep stages simultaneously.

## Key Improvements

### 1. **No Error Cascading** ✓
- Hierarchical: Errors multiply through levels (43.7% test accuracy)
- Direct: All classes predicted together (~70-75% expected)

### 2. **Better Feature Sharing** ✓
- Model learns relationships between ALL sleep stages at once
- Features that distinguish Wake from Sleep also help distinguish N1 from N2

### 3. **Simpler Architecture** ✓
- 1 model instead of 3
- Fewer hyperparameters to tune
- More reliable and robust

### 4. **Moderate Focal Weights** ✓
- Gamma = 1.2 (instead of 1.5)
- Less aggressive, more balanced
- Prevents over-prediction of minority classes

### 5. **Balanced Regularization** ✓
- Allows more model capacity (max_depth=3-6, num_leaves=20-60)
- Not too restrictive
- Can learn complex patterns without overfitting

### 6. **More Optimization** ✓
- 150 trials (vs 100 × 3 = 300 for hierarchical)
- Faster overall since only one model
- Better hyperparameters

## Output Files

Results saved to timestamped directory: `./results/direct_multiclass_YYYY_MM_DD_HH_MM/`

```
├── results.csv                          # Performance metrics
├── method_info.txt                      # Method description and parameters
├── validation_confusion_matrix.png      # Validation confusion matrix
├── confusion_matrix_train.png           # Train confusion matrix
├── confusion_matrix_validation.png      # Validation confusion matrix
├── confusion_matrix_test.png            # Test confusion matrix
├── shap_summary.png                     # Overall feature importance (bar plot)
├── shap_summary_class_Wake.png         # SHAP for Wake class
├── shap_summary_class_REM.png          # SHAP for REM class
├── shap_summary_class_N1.png           # SHAP for N1 class
├── shap_summary_class_N2.png           # SHAP for N2 class
└── shap_summary_class_N3.png           # SHAP for N3 class
```

## Expected Performance

Based on the original paper and proper regularization:

| Metric | Expected Value |
|--------|---------------|
| Train Accuracy | ~75-80% |
| Validation Accuracy | ~70-75% |
| Test Accuracy | ~70-75% |
| Cohen's Kappa | ~0.60-0.65 |

### Per-Class Performance

| Class | Expected F1 |
|-------|------------|
| Wake | ~0.80-0.85 |
| REM | ~0.65-0.70 |
| N1 | ~0.50-0.60 |
| N2 | ~0.75-0.80 |
| N3 | ~0.60-0.70 |

## Comparison: Hierarchical vs Direct

| Metric | Hierarchical (Previous) | Direct (New) |
|--------|------------------------|--------------|
| **Test Accuracy** | 43.7% ❌ | ~70-75% ✓ |
| **Training Time** | ~40 min (3 models) | ~25 min (1 model) |
| **Architecture** | 3 binary/multiclass | 1 multiclass |
| **Error Cascading** | Yes (fatal flaw) | No ✓ |
| **Interpretability** | Good (per level) | Good (per class) |
| **Reliability** | Poor (cascade errors) | High ✓ |

## Configuration Options

### Adjust Focal Weight Gamma

Edit the main() function:

```python
classifier = DirectMulticlassClassifier(
    use_focal_weights=True,
    gamma=1.2  # Try 1.0 for less aggressive, 1.5 for more aggressive
)
```

### Disable Focal Weights

```python
classifier = DirectMulticlassClassifier(
    use_focal_weights=False  # Uses simple 'balanced' weighting
)
```

### Adjust Number of Trials

Edit the fit() method:

```python
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=200,  # Increase for more thorough search
    trials=trials
)
```

## Troubleshooting

### If accuracy is still low (<60%):
1. Check data quality - maybe increase threshold
2. Try gamma=1.0 (less aggressive weights)
3. Increase max_evals to 200 for better optimization

### If overfitting (train >> validation):
1. Increase regularization (reg_alpha, reg_lambda ranges)
2. Reduce max_depth to 2-5
3. Reduce num_leaves to 15-40

### If underfitting (both train and validation low):
1. Decrease regularization
2. Increase max_depth to 4-8
3. Increase num_leaves to 30-80
4. Increase n_estimators

## Why This Should Work

1. **Proven Approach**: The original notebook used direct multiclass successfully
2. **No Cascade Errors**: Single model = no error propagation
3. **Better Optimization**: More trials for one model vs spread across three
4. **Moderate Weights**: gamma=1.2 gives ~3-4x weight to minorities (vs 8x before)
5. **Balanced Regularization**: Strong enough to prevent overfitting, flexible enough to learn

## Next Steps After Running

1. Check `results.csv` for performance metrics
2. Review confusion matrices - should be much more balanced
3. Examine SHAP plots to understand feature importance per class
4. Compare with hierarchical results in `HIERARCHICAL_ISSUES_AND_SOLUTION.md`

---

**Expected Runtime**: ~20-30 minutes on standard hardware

**Memory Usage**: Similar to hierarchical (no data resampling)

**Success Criteria**: Test accuracy > 65%, validation-test gap < 10%

