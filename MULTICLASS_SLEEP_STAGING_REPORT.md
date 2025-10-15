# Comprehensive Report: 5-Class Sleep Stage Classification
## Using Wearable Sensor Data (LightGBM + LSTM)

---

## Executive Summary

**Objective**: Convert binary sleep classification (Wake vs Non-Wake) to full 5-class sleep stage classification: Wake (W), REM (R), N1, N2, N3.

**Current Status**: ❌ **Poor Performance** - Significant issues persist across all classes, particularly minority classes (R, N1, N3).

**Best Results Achieved**:
- LightGBM Test: **32.3% Recall** (macro)
- LSTM Test: **41.0% Recall** (macro, estimated)
- **Major Issue**: Extreme class imbalance leading to poor minority class detection

---

## 1. Data Distribution (Test Set)

| Class | Count | Percentage | Category |
|-------|-------|------------|----------|
| **W** (Wake) | 2,447 | 19.1% | Moderate |
| **R** (REM) | 1,674 | 13.1% | **Minority** |
| **N1** | 1,395 | 10.9% | **Minority** |
| **N2** | 7,028 | **54.9%** | **Majority** |
| **N3** | 305 | **2.4%** | **Extreme Minority** |
| **Total** | 12,849 | 100% | |

**Key Observation**: Severe class imbalance with N2 dominating (55%) and N3 being extremely rare (2.4%).

---

## 2. Changes Implemented (Chronological)

### **Phase 1: Initial Multiclass Conversion**
**Changes**:
- Updated `datasets.py`: Removed N1/N2/N3 collapsing to "N", created 5-class mapping
- Updated `models.py`: Changed LightGBM from `objective="binary"` to `objective="multiclass"`, `num_class=5`
- Updated `models.py`: Changed LSTM output from 2 to 5 classes
- Updated `utils.py`: Modified metrics, probability computation, and plotting for 5 classes

**Results**: 
- ❌ Initial run showed catastrophic N3 performance (15% recall)
- ❌ Heavy bias towards N2 predictions

---

### **Phase 2: Class Imbalance Handling**
**Changes**:
- Added `class_weight='balanced'` to LightGBM
- Added weighted `CrossEntropyLoss` to LSTM
- Implemented early stopping for LightGBM

**Results**:
- ⚠️ Marginal improvement but still poor
- ❌ SMOTE identified as causing severe overfitting

---

### **Phase 3: SMOTE Removal & Regularization**
**Changes**:
- **Removed SMOTE** from `resample_data()` in `datasets.py`
- **Strengthened LightGBM regularization**:
  - `max_depth`: 2-4 (reduced from 2-6)
  - `reg_alpha`: 50-300 (increased from 0-180)
  - `reg_lambda`: 5-15 (increased from 0.2-5)
  - `num_leaves`: 10-40 (reduced from 20-100)
  - `min_child_samples`: 50-200 (newly added)
- **Added early stopping**: 10 rounds
- **Increased hyperopt trials**: 50 → 100

**Results**:
- ✅ Reduced overfitting significantly
- ⚠️ Performance still below acceptable levels

---

### **Phase 4: Focal Loss Implementation**
**Changes**:
- **Updated `FocalLoss` class** in `utils.py` for multiclass
- **LightGBM**: Custom class weights using `(total/count)^1.5`:
  - W: ~2.0, R: ~5.4, N1: ~5.9, N2: ~0.3, N3: ~14.2
- **LSTM**: Applied `FocalLoss` with `gamma=2.0` and same class weights
- **Added feature/row subsampling** to LightGBM:
  - `colsample_bytree`: 0.6-1.0
  - `subsample`: 0.6-0.9
  - `min_data_in_leaf`: 20-100
- **Added DART boosting** option

**Results**:
- ❌ Performance **decreased** due to DART incompatibility with early stopping
- ❌ Macro F1 dropped to **0.356** on validation

---

### **Phase 5: DART Removal & Weight Adjustment**
**Changes**:
- **Removed DART** (incompatible with early stopping)
- **Reverted to weighted F1** (from macro F1)
- **Adjusted class weights** to `(total/count)^1.2` (from 1.5):
  - W: ~4.9, R: ~12.1, N1: ~13.2, N2: ~2.3, N3: ~32.4

**Results**:
- ✅ Training stabilized
- ⚠️ N3 improved to **50% recall** but...
- ❌ **R collapsed to 26% recall** (catastrophic)

---

### **Phase 6: Aggressive N3 Weighting**
**Changes**:
- **Increased N3 weight** to `(total/count)^1.8`:
  - W: ~4.9, R: ~12.1, N1: ~13.2, N2: ~2.3, **N3: ~66.4**
- **LSTM gamma increased** from 1.0 → 2.0
- **Added N3 recall tracking** (post-training metric)

**Results**:
- ✅ N3 improved to **50% recall**
- ❌ **R degraded further to 26% recall**
- ❌ Trade-off: Model sacrificed R to save N3

---

### **Phase 7: R & N3 Rebalancing (Current)**
**Changes**:
- **Rebalanced class weights** `^1.6` for both R and N3:
  - W: ~4.9, **R: ~29.4**, N1: ~13.2, N2: ~2.3, **N3: ~47.1**
- **LSTM gamma reduced** from 2.0 → 1.5
- **Fixed LSTM architecture**: Added `num_layers=2` to enable dropout
- **LSTM now**: 2-layer BiLSTM with dropout=0.3 (previously failed with 1 layer)

**Current Results** (Test Set):
```
LightGBM Test Performance:
- Recall: 32.3% (macro)
- Precision: 55.3%
- F1 Score: 34.1%
- AUROC: 70.8%

LSTM Test Confusion Matrix:
       W     R    N1    N2    N3  (Predicted)
W   [1523  131  551   207   35]  Recall: 62.2% ✅
R   [  29  526  356   686   77]  Recall: 31.4% ❌
N1  [ 289  135  558   358   55]  Recall: 40.0% ❌
N2  [ 550  684 1514  3196 1084]  Recall: 45.5% ❌
N3  [   1   18    7   200   79]  Recall: 25.9% ❌
```

**Class-Specific Analysis**:

| Class | True Count | Correctly Classified | Recall | Major Confusion |
|-------|-----------|---------------------|--------|-----------------|
| **W** | 2,447 | 1,523 | **62.2%** ✅ | 551 → N1 (22.5%) |
| **R** | 1,674 | 526 | **31.4%** ❌ | 686 → N2 (41.0%) |
| **N1** | 1,395 | 558 | **40.0%** ⚠️ | 358 → N2 (25.7%) |
| **N2** | 7,028 | 3,196 | **45.5%** ⚠️ | 1,514 → N1 (21.5%), 1,084 → N3 (15.4%) |
| **N3** | 305 | 79 | **25.9%** ❌ | 200 → N2 (65.6%) |
| **Macro Avg** | - | - | **41.0%** | - |

---

## 3. Current Issues & Root Causes

### **Issue 1: Fundamental Feature Limitation**
**Problem**: Wearable sensor features (accelerometer + HRV) may lack discriminative power for 5-class sleep staging.

**Evidence**:
- Even with extreme class weighting, performance is poor
- LightGBM warnings: "No further splits with positive gain" (tree can't learn)
- Professional sleep staging requires EEG, EOG, EMG (not just actigraphy + heart rate)

**Impact**: May represent a ceiling on achievable performance with current features.

---

### **Issue 2: Severe Class Imbalance**
**Problem**: N3 (2.4%) and R (13.1%) are extremely underrepresented.

**Evidence**:
- Despite 66.4x weight on N3, only 25.9% recall
- Despite 29.4x weight on R, only 31.4% recall
- Model still biased towards N2 (majority class)

**Impact**: Even aggressive weighting cannot overcome imbalance.

---

### **Issue 3: N2-N3-R Confusion Triangle**
**Problem**: Model consistently confuses N2 ↔ N3 ↔ R.

**Evidence**:
- 200/305 N3 → N2 (65.6%)
- 686/1674 R → N2 (41.0%)
- 1,084/7,028 N2 → N3 (15.4%)

**Root Cause**: These stages are physiologically similar without EEG:
- N2 and N3 both involve slow-wave sleep (difference is % of delta waves)
- REM has similar heart rate to light sleep
- Accelerometer cannot distinguish muscle atonia (REM) from stillness (deep sleep)

---

### **Issue 4: LSTM Not Adding Value**
**Problem**: LSTM doesn't significantly improve over LightGBM.

**Evidence**:
- LightGBM: 32.3% macro recall
- LSTM: 41.0% macro recall (modest improvement)
- LSTM training unstable (high loss, slow convergence)

**Root Cause**: 
- LSTM relies on temporal patterns from LightGBM probabilities
- If LightGBM probabilities are poor, LSTM has bad input
- 2-layer LSTM may be overfitting on limited data

---

## 4. Recommended Next Steps

### **Option A: Feature Engineering (Most Promising)**

**Rationale**: Current features may be insufficient for 5-class staging.

**Actions**:
1. **Add spectral features from HRV**:
   - VLF, LF, HF power bands
   - LF/HF ratio (sympathetic/parasympathetic balance)
   - Sample entropy

2. **Add time-domain HRV features**:
   - RMSSD, SDNN, pNN50
   - Heart rate variability indices

3. **Add movement quality features**:
   - Not just magnitude, but patterns:
     - Jerk (acceleration derivative)
     - Frequency of micro-movements
     - Longest stillness duration

4. **Add circadian rhythm features**:
   - Time since sleep onset
   - Time of night
   - Distance from estimated circadian nadir

5. **Add inter-epoch features**:
   - Rolling statistics (mean, std over past N epochs)
   - Change detection (sudden HR changes)
   - Stability metrics

**Expected Impact**: ✅ Could significantly improve R and N3 detection if physiological differences can be captured.

---

### **Option B: Simplify to 3-Class Problem**

**Rationale**: N2-N3 distinction may be impossible without EEG.

**Actions**:
1. **Merge N2 + N3 → "Deep Sleep"**
2. **Keep W, R, N1 separate**
3. **3-class problem**: Wake, REM, Light Sleep (N1), Deep Sleep (N2+N3)

**Pros**:
- ✅ More realistic for wearable devices
- ✅ Clinically useful (deep sleep duration is key metric)
- ✅ Reduces class imbalance (Deep Sleep ~55%)

**Cons**:
- ❌ Less granular
- ❌ User explicitly said "without condensing together any of the classes"

**Expected Impact**: ⚠️ Would improve metrics but violates requirement.

---

### **Option C: Ensemble with Class-Specific Models**

**Rationale**: One model trying to do everything may be suboptimal.

**Actions**:
1. **Train 5 binary classifiers**:
   - W vs all
   - R vs all
   - N1 vs all
   - N2 vs all
   - N3 vs all

2. **Ensemble predictions**:
   - Get probabilities from all 5 models
   - Normalize to sum to 1
   - Take argmax

3. **Tune per-class thresholds**:
   - Different threshold for R (boost recall)
   - Different threshold for N3 (boost recall)

**Expected Impact**: ✅ May improve minority class detection by avoiding multi-class compromises.

---

### **Option D: Data Augmentation**

**Rationale**: Extremely limited N3 data (only 305 samples).

**Actions**:
1. **Synthetic N3 generation**:
   ```python
   def augment_n3(X_n3, y_n3, multiplier=5):
       # Interpolate between N3 samples
       synthetic = []
       for i in range(len(X_n3) * multiplier):
           idx1, idx2 = np.random.choice(len(X_n3), 2)
           alpha = np.random.uniform(0.3, 0.7)
           synthetic_sample = alpha * X_n3[idx1] + (1-alpha) * X_n3[idx2]
           # Add Gaussian noise
           noise = np.random.normal(0, 0.05, synthetic_sample.shape)
           synthetic.append(synthetic_sample + noise)
       return np.array(synthetic)
   ```

2. **Temporal jittering**: Shift epochs by ±5 seconds

3. **Noise injection**: Add small perturbations to features

**Expected Impact**: ⚠️ May help N3 but risk overfitting on synthetic patterns.

---

### **Option E: Two-Stage Hierarchical Classification**

**Rationale**: Group similar classes first, then refine.

**Actions**:
**Stage 1**: Wake vs Sleep (binary)
**Stage 2a**: If Sleep → REM vs NREM
**Stage 2b**: If NREM → N1 vs N2 vs N3

**Pros**:
- ✅ Each model has simpler decision boundary
- ✅ Wake detection likely easier (high movement)
- ✅ Can tune each stage independently

**Cons**:
- ❌ Errors propagate (if Stage 1 wrong, Stage 2 guaranteed wrong)
- ❌ More complex pipeline

**Expected Impact**: ✅ May improve by breaking complex problem into simpler sub-problems.

---

### **Option F: Transfer Learning from EEG Data**

**Rationale**: Learn sleep stage patterns from gold-standard EEG data, adapt to wearables.

**Actions**:
1. **Pre-train on EEG dataset** (e.g., Sleep-EDF, SHHS):
   - Learn general sleep stage patterns
   - Extract feature representations

2. **Fine-tune on wearable data**:
   - Transfer learned patterns
   - Adapt to different sensor modality

**Pros**:
- ✅ Leverages knowledge from high-quality sleep staging
- ✅ May improve generalization

**Cons**:
- ❌ Requires EEG dataset with aligned wearable data
- ❌ Modality gap (EEG ≠ actigraphy/HRV)

**Expected Impact**: ⚠️ Uncertain, depends on transferability across modalities.

---

## 5. Recommended Priority Order

### **Tier 1: High Impact, Feasible (Do First)**

1. **✅ Option A: Feature Engineering**
   - Add HRV spectral features
   - Add movement quality features
   - Add circadian/temporal features
   - **Expected improvement**: +10-20% in R and N3 recall

2. **✅ Option C: Ensemble with Class-Specific Models**
   - Train 5 binary classifiers
   - Ensemble predictions
   - Tune per-class thresholds
   - **Expected improvement**: +5-15% in minority classes

### **Tier 2: Medium Impact, More Complex**

3. **⚠️ Option E: Hierarchical Classification**
   - Two-stage approach
   - **Expected improvement**: +5-10% overall

4. **⚠️ Option D: Data Augmentation**
   - Synthetic N3 generation
   - **Expected improvement**: +5-10% in N3 only, risk of overfitting

### **Tier 3: Research/Experimental**

5. **🔬 Option F: Transfer Learning**
   - Requires significant research
   - **Expected improvement**: Unknown

6. **❌ Option B: 3-Class Simplification**
   - Violates user requirement
   - Only if all other options fail

---

## 6. Realistic Performance Expectations

Given the constraints (no EEG, severe imbalance, feature limitations):

**Achievable with Feature Engineering + Ensemble**:
| Class | Current Recall | Target Recall | Is Realistic? |
|-------|---------------|---------------|---------------|
| W | 62% | 75-80% | ✅ Yes |
| R | 31% | 50-60% | ✅ Possible |
| N1 | 40% | 55-65% | ✅ Possible |
| N2 | 46% | 60-70% | ✅ Possible |
| N3 | 26% | 40-50% | ⚠️ Challenging |
| **Macro Avg** | **41%** | **56-65%** | ✅ Achievable |

**Best-Case Scenario** (with all improvements):
- Macro recall: **~60%**
- AUROC: **~80%**
- Cohen's Kappa: **~0.50** (moderate agreement)

**Reality Check**: 
- Professional sleep staging requires PSG (EEG + EOG + EMG)
- Consumer wearables typically achieve 60-70% accuracy vs PSG
- Your performance target should align with this benchmark

---

## 7. Conclusion

**Summary**:
- ✅ Successfully converted to 5-class multiclass
- ✅ Implemented focal loss, class weighting, regularization
- ✅ Removed overfitting (SMOTE removal)
- ❌ **Performance still poor** (~41% macro recall)

**Root Cause**: 
- **Feature limitation** is the primary bottleneck
- **Extreme class imbalance** cannot be fully overcome with weighting alone
- **Physiological similarity** of N2/N3/R without EEG makes distinction very difficult

**Next Steps**:
1. **Implement Option A** (Feature Engineering) - highest priority
2. **Implement Option C** (Ensemble) - complement to feature engineering
3. **Reassess** after these improvements
4. **Consider hierarchical approach** (Option E) if still insufficient

**Final Note**: If performance remains below 55% macro recall after feature engineering, the problem may be fundamentally limited by sensor modality (actigraphy + HRV insufficient for 5-class staging), and simplification to 3-4 classes may be necessary for clinical usefulness.

---

## 8. Technical Debt & Code Quality

**Current Code State**:
- ✅ Clean modularization (`datasets.py`, `models.py`, `utils.py`)
- ✅ Proper multiclass support throughout pipeline
- ✅ Focal loss correctly implemented
- ✅ Dropout fixed in 2-layer LSTM

**Remaining Issues**:
- ⚠️ Hardcoded class weights (should be computed dynamically)
- ⚠️ No validation-based early stopping for LSTM
- ⚠️ GPBoost still not working (needs multiclass implementation)

---

*Report Generated: 2025-10-15*  
*Current Status: Phase 7 - R & N3 Rebalancing*  
*Next Action: Implement Feature Engineering (Option A)*

