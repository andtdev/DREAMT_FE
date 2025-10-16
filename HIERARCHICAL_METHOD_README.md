# Hierarchical Sleep Stage Classification

## Overview

This document describes the new hierarchical classification approach implemented in `experiments_multiclass.py`. This script replaces the Jupyter notebook workflow with a more maintainable Python script.

## What's New

### 1. Script-Based Workflow
- Converted from Jupyter notebook (`experiments_multiclass.ipynb`) to Python script (`experiments_multiclass.py`)
- Easier to version control, modify, and integrate into pipelines
- More reproducible and trackable

### 2. Hierarchical Classification Architecture

The new approach uses a **three-level hierarchy** instead of direct multiclass classification:

#### Level 1: Wake vs Sleep
- **Binary Classification**: Wake (0) vs All Sleep Stages (1, 2, 3, 4)
- **Purpose**: Distinguish wake from sleep states
- **Key Features**: Activity-based features (ACC), heart rate, temperature

#### Level 2: REM vs Non-REM
- **Binary Classification**: REM (1) vs Non-REM (2, 3, 4)
- **Applied to**: Only samples classified as "Sleep" from Level 1
- **Purpose**: Distinguish REM from non-REM sleep
- **Key Features**: Heart rate variability, movement patterns

#### Level 3: Non-REM Stage Classification
- **Multiclass Classification**: N1 (2) vs N2 (3) vs N3 (4)
- **Applied to**: Only samples classified as "Non-REM" from Level 2
- **Purpose**: Distinguish between light (N1), medium (N2), and deep (N3) sleep
- **Key Features**: Deep sleep indicators (HRV, temperature, EDA)

### 3. Advantages of Hierarchical Approach

1. **Better Class Imbalance Handling**: Each level handles its own class balance
2. **Feature Customization**: Different features can be optimized for each classification task
3. **Interpretability**: Clear decision path through the hierarchy
4. **Improved Performance**: Each classifier focuses on distinguishing similar patterns

### 4. Method Documentation

The script automatically generates a `method_info.txt` file containing:
- Method description and rationale
- Hyperparameters for each level
- Feature selection strategy
- Timestamp and configuration

## Usage

### Basic Usage

```bash
python experiments_multiclass.py
```

### What the Script Does

1. **Data Preparation**: Loads and preprocesses feature data
2. **Data Splitting**: Creates train/validation/test splits
3. **Hierarchical Training**: 
   - Trains Level 1 model (Wake vs Sleep)
   - Trains Level 2 model (REM vs Non-REM)
   - Trains Level 3 model (N1 vs N2 vs N3)
4. **Evaluation**: Tests on all datasets
5. **Results Output**: 
   - CSV file with metrics: `./results/hierarchical/hierarchical_results.csv`
   - Method info: `./results/hierarchical/method_info.txt`

### Output Files

```
./results/hierarchical/
├── hierarchical_results.csv              # Performance metrics
├── method_info.txt                       # Method description and parameters
├── level1_confusion_matrix.png           # Level 1 confusion matrix (Wake vs Sleep)
├── level1_shap.png                       # Level 1 feature importance (bar plot)
├── level1_shap_beeswarm.png             # Level 1 feature importance (detailed)
├── level2_confusion_matrix.png           # Level 2 confusion matrix (REM vs Non-REM)
├── level2_shap.png                       # Level 2 feature importance (bar plot)
├── level2_shap_beeswarm.png             # Level 2 feature importance (detailed)
├── level3_confusion_matrix.png           # Level 3 confusion matrix (N1 vs N2 vs N3)
├── level3_shap.png                       # Level 3 feature importance (bar plot)
├── level3_shap_beeswarm.png             # Level 3 feature importance (detailed)
├── final_confusion_matrix_train.png      # Final results on train set
├── final_confusion_matrix_validation.png # Final results on validation set
└── final_confusion_matrix_test.png       # Final results on test set
```

## Key Differences from Original Notebook

| Aspect | Original Notebook | New Script |
|--------|------------------|------------|
| **Format** | Jupyter notebook | Python script |
| **Classification** | Direct multiclass | Hierarchical (3 levels) |
| **Models Used** | LightGBM + LSTM + GPBoost | LightGBM only (others kept as imports) |
| **Feature Selection** | Global features | Can customize per level |
| **Documentation** | Manual | Automatic method_info.txt |
| **Class Imbalance** | SMOTE + class weights | Handled at each hierarchy level |

## Modifications and Extensions

### Customizing Features per Level

To use different features at each level, modify the `select_features_for_level()` method in the `HierarchicalSleepClassifier` class:

```python
def select_features_for_level(self, all_features, level):
    if level == 1:
        # Wake vs Sleep: prioritize activity features
        return [f for f in all_features if 'ACC' in f or 'HR' in f]
    elif level == 2:
        # REM vs Non-REM: prioritize HRV features
        return [f for f in all_features if 'HRV' in f or 'derivative' in f]
    elif level == 3:
        # N1 vs N2 vs N3: prioritize deep sleep features
        return [f for f in all_features if 'TEMP' in f or 'EDA' in f]
```

### Adjusting Hyperparameter Search

Modify the `space` dictionary in `train_lgb_binary()` or `train_lgb_multiclass()` methods to adjust hyperparameter ranges.

### Adding LSTM Post-Processing

To add LSTM post-processing (currently excluded), you can:
1. Use `classifier.predict_proba()` to get probability sequences
2. Apply the LSTM from `models.py` as a post-processing step

## Visualizations

The script automatically generates visualizations for each level and the final results:

### Confusion Matrices
- **Level-specific matrices**: Show performance at each hierarchical level
  - Level 1: Wake vs Sleep classification
  - Level 2: REM vs Non-REM classification (within sleep)
  - Level 3: N1 vs N2 vs N3 classification (within non-REM)
- **Final matrices**: Show overall 5-class performance on train/validation/test sets

### SHAP (Feature Importance) Plots
For each level, two types of SHAP plots are generated:

1. **Bar Plot** (`levelX_shap.png`): Shows the top 20 most important features
   - Easy to read ranking of feature importance
   - Good for quick identification of key features

2. **Beeswarm Plot** (`levelX_shap_beeswarm.png`): Shows detailed feature impacts
   - Each dot is a sample
   - Color indicates feature value (red=high, blue=low)
   - X-axis shows impact on prediction
   - Reveals feature interactions and patterns

## Performance Metrics

The script reports:
- **Accuracy**: Overall classification accuracy
- **F1 (weighted)**: Weighted F1 score across all classes
- **F1 (macro)**: Unweighted average F1 score
- **Cohen's Kappa**: Agreement metric accounting for chance
- **Per-Class F1**: Individual F1 scores for Wake, REM, N1, N2, N3

## Future Enhancements

1. ~~**Feature Importance Analysis**: Add feature importance visualization per level~~ ✓ Implemented
2. **Cross-Validation**: Implement subject-wise cross-validation
3. **LSTM Integration**: Add optional LSTM post-processing
4. **Uncertainty Estimation**: Add prediction confidence metrics
5. **Model Persistence**: Save/load trained models
6. **Feature Selection**: Automatically select different feature subsets per level based on importance

## Questions and Troubleshooting

### Common Issues

**Q: Script is slow during training**
- A: Reduce `max_evals` in hyperparameter optimization (default: 50)

**Q: Class imbalance warnings**
- A: The hierarchical approach handles this at each level automatically

**Q: Memory errors**
- A: Reduce batch sizes or use feature selection to reduce dimensionality

## References

- Original multiclass notebook: `experiments_multiclass.ipynb`
- Utility functions: `utils.py`
- Data loading: `datasets.py`
- Model definitions: `models.py`

---

**Created**: 2025-10-15  
**Author**: Hierarchical Classification Script  
**Version**: 1.0

