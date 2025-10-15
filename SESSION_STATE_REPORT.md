# Session State Report: Multiclass Sleep Staging
**Date**: 2025-10-15  
**Status**: Ready to Resume  
**Current Performance**: ~41% macro recall (LSTM), ~32% (LightGBM)

---

## 🎯 Current State Summary

### **What We're Trying to Do:**
Convert binary sleep classification (Wake vs Non-Wake) to full 5-class sleep stage classification:
- **W** (Wake)
- **R** (REM Sleep)
- **N1** (Light Sleep Stage 1)
- **N2** (Light Sleep Stage 2)
- **N3** (Deep Sleep / Slow-Wave Sleep)

### **Current Results:**
```
LSTM Test Confusion Matrix (Latest Run):
       W     R    N1    N2    N3  (Predicted)
W   [1523  131  551   207   35]  Recall: 62.2% ✅
R   [  29  526  356   686   77]  Recall: 31.4% ❌
N1  [ 289  135  558   358   55]  Recall: 40.0% ❌
N2  [ 550  684 1514  3196 1084]  Recall: 45.5% ❌
N3  [   1   18    7   200   79]  Recall: 25.9% ❌

Macro Average Recall: ~41%

LightGBM Test Performance:
- Recall: 32.3%
- Precision: 55.3%
- F1 Score: 34.1%
- AUROC: 70.8%
```

---

## 📁 Files Modified (Ready to Commit)

### **Core Code Files:**

1. **`datasets.py`** - Data loading and preprocessing
   - ✅ Removed N1/N2/N3 collapsing to "N"
   - ✅ 5-class mapping: `{"W": 0, "R": 1, "N1": 2, "N2": 3, "N3": 4}`
   - ✅ Removed SMOTE resampling (was causing overfitting)
   - ✅ `resample_data()` now returns original data unchanged

2. **`models.py`** - Model definitions and training
   - ✅ LightGBM converted to multiclass (`objective="multiclass"`, `num_class=5`)
   - ✅ LSTM converted to 5 output classes
   - ✅ BiLSTMPModel and LSTMPModel updated with `num_layers=2` for proper dropout
   - ✅ Class weights implemented: `(total/count)^1.6` for R and N3, `^1.2` for others
   - ✅ Focal loss with `gamma=1.5` for LSTM
   - ✅ 100 hyperopt trials (increased from 50)
   - ✅ Early stopping, regularization (subsampling, feature sampling, L1/L2)
   - ✅ Validation N3 recall tracking post-training

3. **`utils.py`** - Utilities and metrics
   - ✅ FocalLoss class updated for multiclass with alpha (class weights) and gamma
   - ✅ `compute_probabilities()` updated for 5 probability columns
   - ✅ `calculate_metrics()` rewritten for multiclass (weighted, one-vs-rest AUROC)
   - ✅ `plot_cm()` updated for 5x5 confusion matrices

4. **`experiments_multiclass.ipynb`** - Main notebook
   - ✅ All cells updated for 5-class classification
   - ✅ Contains latest results with current hyperparameters

### **Documentation Files (NEW):**

5. **`MULTICLASS_SLEEP_STAGING_REPORT.md`**
   - Comprehensive report of all changes made
   - 7 phases of optimization documented
   - Next steps and recommendations

6. **`CRITICAL_UPDATE_FEATURES_ALREADY_PRESENT.md`**
   - Key finding: Dataset already has 139 comprehensive features
   - HRV time-domain, frequency-domain, entropy, fractal dimensions
   - Revised recommendations based on feature availability

7. **`SESSION_STATE_REPORT.md`** (this file)
   - Current state snapshot for restart

8. **`check_feature_quality.py`**
   - Script to audit feature quality (NaN, inf, correlation)
   - Ready to run when needed

---

## 🔑 Key Findings

### **1. Features Are Comprehensive (Critical Discovery)**
The dataset already contains **139 features** including:
- ✅ All HRV time-domain metrics (SDNN, RMSSD, pNN50, etc.)
- ✅ All HRV frequency-domain (LF, HF, VHF, LF/HF ratio)
- ✅ Advanced metrics: entropy (ApEn, SampEn), fractal dimensions (HFD, KFD), DFA
- ✅ Multi-modal: accelerometer (3-axis), temperature, skin conductance, circadian rhythm

**Implication**: Poor performance (41%) is NOT due to missing features.

### **2. Class Imbalance is Severe**
```
Test Set Distribution:
- W:  2,447 (19.1%)
- R:  1,674 (13.1%) ← Minority
- N1: 1,395 (10.9%) ← Minority  
- N2: 7,028 (54.9%) ← Dominant majority
- N3:   305 (2.4%)  ← Extremely rare
Total: 12,849
```

### **3. Major Confusion Patterns**
- **N3 → N2**: 65.6% of N3 misclassified as N2
- **R → N2**: 41.0% of R misclassified as N2
- **N2 → N3**: 15.4% of N2 misclassified as N3 (over-predicting N3)

### **4. Model Configuration (Current)**

**LightGBM Hyperparameters:**
```python
{
    'max_depth': 2-4,
    'reg_alpha': 50-300,
    'reg_lambda': 5-15,
    'num_leaves': 10-40,
    'n_estimators': 50-200,
    'learning_rate': 0.01-0.1,
    'min_child_samples': 50-200,
    'colsample_bytree': 0.6-1.0,
    'subsample': 0.6-0.9,
    'min_data_in_leaf': 20-100,
    'boosting_type': 'gbdt'  # DART removed due to early stopping incompatibility
}

Class Weights (Focal-Loss-Inspired):
{
    0: (43761/10698)^1.2,  # W: ~4.9
    1: (43761/4666)^1.6,   # R: ~29.4 (boosted to protect R)
    2: (43761/4320)^1.2,   # N1: ~13.2
    3: (43761/22209)^1.2,  # N2: ~2.3
    4: (43761/1868)^1.6,   # N3: ~47.1 (balanced with R)
}

Optimization: weighted F1 score
Early Stopping: 10 rounds on validation set
Hyperopt Trials: 100
```

**LSTM Configuration:**
```python
Architecture: 2-layer BiLSTM
- Input size: 7 (5 LightGBM probabilities + 2 features)
- Hidden size: 64
- Output size: 5
- Num layers: 2 (enables dropout)
- Dropout: 0.3
- Bidirectional: True

Loss: FocalLoss(alpha=class_weights, gamma=1.5)
- Gamma reduced from 2.0 to 1.5 for better R/N3 balance

Class Weights:
- Same as LightGBM: R and N3 boosted to ^1.6

Optimizer: Adam (lr=0.001, weight_decay=1e-5)
Scheduler: ReduceLROnPlateau(factor=0.5, patience=3)
Early Stopping: patience=5 on validation loss
Gradient Clipping: max_norm=1.0
Epochs: 300 (with early stopping)
```

---

## 📊 What Changed (Chronological)

### **Phase 1: Multiclass Conversion**
- Changed from binary to 5-class
- Updated all metrics and plotting

### **Phase 2: Initial Class Weighting**
- Added `class_weight='balanced'`
- Added weighted CrossEntropyLoss

### **Phase 3: SMOTE Removal**
- Identified SMOTE causing severe overfitting
- Removed synthetic sample generation
- Strengthened regularization instead

### **Phase 4: Focal Loss (First Attempt)**
- Implemented FocalLoss with gamma=2.0
- Used aggressive N3 weighting (^1.8)
- Added feature/row subsampling
- Tried DART boosting

### **Phase 5: DART Removal & Weight Adjustment**
- Removed DART (incompatible with early stopping)
- Changed to weighted F1 optimization
- Reduced weights to ^1.2

### **Phase 6: Aggressive N3 Weighting**
- Increased N3 to ^1.8 (66.4x weight)
- Increased LSTM gamma to 2.0
- **Result**: N3 improved but R collapsed to 26%

### **Phase 7: R & N3 Rebalancing (CURRENT)**
- Both R and N3 set to ^1.6 (~29.4x and ~47.1x)
- LSTM gamma reduced to 1.5
- Added 2-layer LSTM architecture (dropout now works)
- **Result**: Current state (R: 31.4%, N3: 25.9%)

---

## 🚀 Next Steps (When You Resume)

### **Priority 1: Feature Analysis** ⭐ RECOMMENDED
The dataset has 139 features but performance is only 41%. Check if most features are being ignored:

1. **Analyze SHAP plot** (already in notebook Cell 12):
   ```python
   # Look at the SHAP feature importance plot
   # Questions to answer:
   # - Which features have the highest importance?
   # - Are most of the 139 features near zero importance?
   # - Do different classes rely on different features?
   ```

2. **If many features have low importance**, try feature selection:
   ```python
   # Keep only top 30-50 features
   feature_importance = final_lgb_model.feature_importances_
   top_indices = np.argsort(feature_importance)[-50:]  # Top 50
   X_train_filtered = X_train[:, top_indices]
   
   # Retrain with reduced feature set
   # May improve by 10-20% by removing noise
   ```

3. **Check for data quality issues**:
   - Run `check_feature_quality.py` (already created)
   - Look for NaN, inf, constant features
   - Check for highly correlated features (>0.95)

### **Priority 2: Class-Specific Models**
Since overall performance is poor despite good features, try specialized models:

```python
# Train 5 binary "one-vs-rest" classifiers
models = {}
for class_id in range(5):
    y_binary = (y_train == class_id).astype(int)
    models[class_id] = train_binary_model(X_train, y_binary)

# Ensemble predictions
def predict_ensemble(X):
    probs = []
    for class_id in range(5):
        prob = models[class_id].predict_proba(X)[:, 1]
        probs.append(prob)
    probs = np.array(probs).T
    # Normalize to sum to 1
    probs = probs / probs.sum(axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
```

### **Priority 3: Reality Check**
If improvements from above are < 10%:
- **Accept that 5-class staging with wearables has a ~60% ceiling**
- Professional PSG uses EEG + EOG + EMG
- Consumer wearables typically achieve 60-70% vs PSG
- Your 41% → target 55-60% is realistic

### **Optional: Hierarchical Classification**
If needed, try breaking the problem down:
```python
# Stage 1: Wake vs Sleep (likely easy - use movement)
# Stage 2: If Sleep → REM vs NREM (use HR variability)
# Stage 3: If NREM → N1 vs N2 vs N3 (use HRV complexity)
```

---

## 🐛 Known Issues

1. **GPBoost not working** - needs multiclass implementation (deferred)
2. **Dropout warning in LSTM** - now fixed with `num_layers=2`
3. **Performance plateau** - fundamental limitation vs need for better features/models (TBD)

---

## 💾 How to Resume

### **1. Start Jupyter Notebook**
```bash
cd /home/admin/DREAMT_FE
jupyter notebook
# Or your preferred method to start
```

### **2. Open `experiments_multiclass.ipynb`**
- Cell 10: Shows latest LightGBM results
- Cell 12: SHAP feature importance plot ⭐ **START HERE**
- Cell 16: Shows latest LSTM results

### **3. Analyze SHAP Plot (Cell 12)**
Look at which features matter most:
- Are most features near zero importance?
- Which features are actually used?
- This will guide next steps

### **4. Depending on SHAP Results:**

**If most features have low importance:**
→ Implement feature selection (Priority 1.2)

**If features are well-distributed:**
→ Try class-specific models (Priority 2)

**If still poor after both:**
→ Likely at fundamental ceiling for wearables (~60%)

---

## 📝 Code Locations

### **Where to find key functions:**

**Data Loading:**
- `datasets.py`: `data_preparation()`, `split_data()`, `train_test_split()`, `resample_data()`

**Models:**
- `models.py`: `LightGBM_engine()`, `LSTM_engine()`, `LSTM_dataloader()`
- `models.py`: Lines 38-60: `BiLSTMPModel` class
- `models.py`: Lines 63-79: `LSTMPModel` class

**Metrics & Utils:**
- `utils.py`: `calculate_metrics()`, `plot_cm()`, `compute_probabilities()`
- `utils.py`: Lines 189-239: `FocalLoss` class

**Hyperparameters:**
- `models.py`: Lines 100-112: LightGBM hyperparameter space
- `models.py`: Lines 120-126: LightGBM class weights
- `models.py`: Lines 499-506: LSTM class weights
- `models.py`: Line 513: Focal loss gamma

---

## 🎬 Quick Commands

### **To restart from scratch:**
```bash
# Restart Jupyter kernel
# Re-run all cells in experiments_multiclass.ipynb
```

### **To check feature quality:**
```bash
cd /home/admin/DREAMT_FE
# Will need to run in Jupyter environment with pandas available
# Or modify check_feature_quality.py to run in notebook
```

### **To modify hyperparameters:**
Edit `models.py`:
- Lines 120-126: LightGBM class weights
- Line 513: LSTM focal loss gamma
- Lines 100-112: LightGBM search space

---

## 📈 Performance Expectations

### **Current Performance:**
- Macro Recall: 41%
- Best class (W): 62%
- Worst class (N3): 26%

### **Realistic Targets with Feature Selection:**
- Macro Recall: 50-55% (optimistic)
- Best class (W): 70-75%
- Worst class (N3): 35-45%

### **Absolute Ceiling (Wearables without EEG):**
- Macro Recall: ~60% (best case)
- Consumer wearables vs PSG: 60-70% agreement
- 5-class requires brain waves (EEG) for reliable N2/N3/REM distinction

---

## ✅ Git Commit Checklist

Files to commit:
- [x] `datasets.py` - Multiclass + SMOTE removal
- [x] `models.py` - LightGBM + LSTM multiclass with focal loss
- [x] `utils.py` - FocalLoss + multiclass metrics
- [x] `experiments_multiclass.ipynb` - Latest results
- [x] `MULTICLASS_SLEEP_STAGING_REPORT.md` - Full documentation
- [x] `CRITICAL_UPDATE_FEATURES_ALREADY_PRESENT.md` - Feature analysis
- [x] `SESSION_STATE_REPORT.md` - This file
- [x] `check_feature_quality.py` - Feature audit script

Suggested commit message:
```
Multiclass sleep staging implementation with focal loss

- Converted from binary to 5-class (W, R, N1, N2, N3)
- Implemented focal loss with class-specific weights
- Added regularization, early stopping, 2-layer LSTM
- Current performance: 41% macro recall (LSTM)
- Identified 139 comprehensive features already present
- Next: Feature selection and class-specific models

Phase 7 complete - ready for feature importance analysis
```

---

## 🎯 Key Takeaways

1. **✅ Multiclass conversion complete and working**
2. **✅ Comprehensive features already present (139 total)**
3. **❌ Performance still poor (41% macro recall)**
4. **🔍 Root cause unclear: model issue vs fundamental ceiling**
5. **🚀 Next step: Analyze SHAP to see if feature selection helps**
6. **⚠️ May be at physiological limit (~60% ceiling without EEG)**

---

## 📞 Contact Points

If you need to reference decisions:
- Why SMOTE removed: See Phase 3 notes
- Why gamma=1.5: See Phase 7 notes (balance R and N3)
- Why ^1.6 for R/N3: See Phase 7 notes (prevent R collapse)
- Why 2-layer LSTM: See Phase 7 notes (enable dropout)

---

**STATUS**: Ready to resume tomorrow
**NEXT ACTION**: Analyze SHAP plot in Cell 12
**ESTIMATED TIME**: 1-2 hours for feature analysis + retraining

---

*Session saved: 2025-10-15*  
*Resume point: Notebook Cell 12 - SHAP Analysis*

