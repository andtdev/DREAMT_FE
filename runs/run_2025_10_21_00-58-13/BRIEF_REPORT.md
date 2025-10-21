# Run Report: 2025-10-21 00:58:13

## Configuration Changes
- **Regularization**: Increased to MODERATE (reg_alpha: 5-50, reg_lambda: 0.5-3.0)
- **SMOTE**: Reduced to 50% oversampling (from 100% full balancing)
- **Class Weights**: R=12.36x, N1=13.35x, W=0.59x, N2+N3=0.60x (4x boost on minorities)

## Overall Performance

### LightGBM Test Results
- **Accuracy**: 52.58% (⚠️ poor)
- **F1 Macro**: 47.42%
- **Train/Test Gap**: 11.36% (improved - no severe overfitting)

### LSTM Test Results (After Temporal Modeling)
- **Accuracy**: 60.22% (+7.6% improvement)
- **F1 Macro**: 52.30% (+4.9% improvement)
- **Cohen's Kappa**: 0.4327

## Per-Class Performance (LSTM Final)

| Class | F1 Score | AUROC | Issues |
|-------|----------|-------|--------|
| **W (Wake)** | 0.8084 | 0.9408 | ✓ Good |
| **R (REM)** | **0.3688** | 0.7903 | ❌ **VERY BAD** |
| **N1** | 0.2578 | 0.7440 | ❌ Poor |
| **N2+N3** | 0.6572 | 0.8546 | ✓ Acceptable |

## REM Phase Analysis - Critical Issue

### REM Confusion Matrix (LSTM Test)
Out of **1,674 REM samples**:
- **508 (30.3%)** correctly classified as REM ✓
- **585 (34.9%)** misclassified as N2+N3 ❌
- **507 (30.3%)** misclassified as N1 ❌
- **74 (4.4%)** misclassified as W

**Problem**: REM is essentially being classified randomly across N1, R, and N2+N3. The model cannot distinguish REM from other sleep stages.

## Root Causes of Poor REM Performance

1. **Class Weight Too Aggressive**: 12.36x multiplier may be causing model instability
2. **Feature Discrimination**: Current features may not capture distinctive REM characteristics:
   - Rapid eye movements
   - Muscle atonia
   - High-frequency EEG patterns
   - Irregular heart rate/breathing

3. **SMOTE Synthetic Samples**: Even at 50%, synthetic REM samples may not capture true REM physiology

4. **Regularization**: Moderate regularization (reg_alpha=5, reg_lambda=1.89) still preventing enough splits

## Key Observations

✅ **Good**:
- No severe overfitting (train 63.9%, test 52.6%)
- Wake detection excellent (F1=0.81)
- LSTM provides meaningful improvement (+7.6% accuracy)

❌ **Bad**:
- REM detection failing (F1=0.37)
- N1 detection still poor (F1=0.26)
- High confusion between minority classes (R, N1, N2+N3)
- Still getting "No further splits" warnings despite reduced regularization

## Recommendations

1. **Reduce class weights** to 2x-3x (currently 4x is too aggressive)
2. **Further reduce regularization** or try different approach (remove SMOTE entirely, use only class weights)
3. **Add REM-specific features**:
   - Eye movement metrics (if available)
   - EEG frequency band ratios (theta/alpha)
   - Heart rate variability during REM
4. **Consider separate REM vs non-REM binary classifier** as first stage
5. **Analyze feature importance for REM** class to identify discriminative features

## Next Iteration Plan

### Priority 1: Reduce Class Weight Aggressiveness
**Change in `new_experiments.py`:**
```python
# Line ~210-220: Change scaling_factors from 4.0x to 2.0x
scaling_factors = {
    0: 1.0,    # W (majority) - no boost
    1: 2.0,    # R (minority) - reduced from 4.0 to 2.0
    2: 2.0,    # N1 (minority) - reduced from 4.0 to 2.0
    3: 1.0     # N2+N3 (majority) - no boost
}
```

### Priority 2: Remove SMOTE Completely
**Change in `new_experiments.py` (line ~227-238):**
```python
# Comment out SMOTE - rely purely on class weights
print("\nSkipping SMOTE - using class weights only to prevent synthetic sample issues...")
X_train_resampled = X_train
y_train_resampled = y_train
```

### Priority 3: Further Reduce Regularization
**Change in `models.py` (line ~288-289):**
```python
"reg_alpha": hp.quniform("reg_alpha", 0, 10, 1),      # Reduced from 5-50
"reg_lambda": hp.uniform("reg_lambda", 0.1, 1.0),     # Reduced from 0.5-3.0
```

### Alternative Approach (If Above Fails)
Consider trying **3-class classification** instead:
1. Merge R + N1 into single "Light Sleep" class
2. Classes: W, Light (R+N1), Deep (N2+N3)
3. This reduces inter-class confusion between REM and N1

## Comparison to Previous Runs

| Run | Test Acc | REM F1 | Train/Test Gap |
|-----|----------|--------|----------------|
| Oct 20 (22:44) | 59.6% | 0.349 | Moderate |
| Oct 21 (00:34) | 59.8% | 0.312 | **SEVERE (24%)** |
| **Oct 21 (00:58)** | **52.6%** | **0.369** | **11.4% (Good)** |

**Verdict**: Fixed overfitting but overall performance degraded. REM remains the critical bottleneck.

