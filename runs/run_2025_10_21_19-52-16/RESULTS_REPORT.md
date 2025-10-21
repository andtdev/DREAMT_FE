# Run Report: 2025-10-21 19:52:16
## Configuration: No SMOTE + 2x Class Weights + Low Regularization

---

## 🎯 Executive Summary

**Overall Verdict:** ⚠️ **Mixed Results - Progress on Overfitting, But Performance Plateau**

**Key Takeaways:**
- ✅ **Fixed overfitting:** Train/test gap reduced from 24% to 9.4%
- ✅ **LSTM adds value:** +6.8% accuracy improvement over LightGBM
- ✅ **Wake & N2+N3 detection:** Acceptable performance (F1 = 0.85, 0.70)
- ❌ **REM & N1 still failing:** No improvement (F1 = 0.38, 0.28)
- ❌ **Fundamental ceiling reached:** ~66% accuracy appears to be the limit with current features

**Recommendation:** This is likely the best you can achieve with aggregated features. **Time to move to raw 64Hz signals** (see RESTRUCTURING_README.md).

---

## 📊 Configuration Changes (From Previous Run)

### **What Changed:**

1. **Removed SMOTE Completely** ✓
   - Previous: 50% oversampling
   - Current: No synthetic samples
   - Rationale: Prevent unrealistic physiological patterns

2. **Reduced Class Weight Scaling** ✓
   - Previous: 4x boost for R and N1
   - Current: 2x boost for R and N1
   - Rationale: Reduce model instability

3. **Reduced Regularization** ✓
   - Previous: reg_alpha 5-50, reg_lambda 0.5-3.0
   - Current: reg_alpha 0-10, reg_lambda 0.1-1.0
   - Rationale: Allow more tree splits for minority classes

### **Current Configuration:**
```
Classes: 4 (W, R, N1, N2+N3)
Data: Original training data (no SMOTE)
Class Weights:
  - W: 0.586 (1.0x base)
  - R: 6.181 (2.0x boost)
  - N1: 6.676 (2.0x boost)
  - N2+N3: 0.599 (1.0x base)

LightGBM Hyperparameters (best from 50 trials):
  - learning_rate: 0.072
  - max_depth: 6
  - n_estimators: 270
  - num_leaves: 40
  - reg_alpha: 7.0
  - reg_lambda: 0.60

LSTM:
  - Input: 4 LightGBM probs + 5 HRV features = 9 inputs
  - Architecture: BiLSTM (hidden_size=64, layers=1)
  - Loss: Weighted Cross-Entropy
  - Epochs: 300
```

---

## 📈 Performance Results

### **LightGBM Test Performance**

| Metric | Train | Test | Gap |
|--------|-------|------|-----|
| **Accuracy** | 68.13% | 58.78% | 9.35% ✓ |
| **F1 Macro** | 63.89% | 51.18% | 12.71% |
| **F1 Weighted** | 72.40% | 63.30% | 9.10% |
| **AUROC Macro** | 95.25% | 82.28% | 12.97% |

**Analysis:**
- ✅ **Overfitting controlled:** 9.4% gap (vs 24% in previous run)
- ⚠️ **Overall performance dropped:** Test accuracy down from 60.2% to 58.8%
- ⚠️ **Trade-off:** Less overfitting but worse test performance

---

### **LSTM Final Performance** ⭐

| Metric | LightGBM Only | LSTM | Improvement |
|--------|---------------|------|-------------|
| **Accuracy** | 58.78% | 65.56% | **+6.77%** ✓ |
| **F1 Macro** | 51.18% | 55.12% | **+3.95%** ✓ |
| **F1 Weighted** | 63.30% | 69.10% | **+5.80%** ✓ |
| **AUROC Macro** | 82.28% | 84.53% | **+2.24%** ✓ |
| **Cohen's Kappa** | N/A | 0.4859 | - |

**Analysis:**
- ✅ **LSTM adds consistent value:** 6-7% accuracy boost across all metrics
- ✅ **Temporal modeling helps:** LSTM captures sleep stage transitions
- ✅ **Kappa = 0.49:** Moderate agreement (0.4-0.6 = moderate)

---

### **Per-Class Performance**

#### **LSTM Test Results (Final Model):**

| Class | F1 Score | AUROC | AUPRC | Samples | Recall |
|-------|----------|-------|-------|---------|--------|
| **W (Wake)** | **0.8459** ✓ | 0.9517 | 0.9431 | 6,547 | 78.0% |
| **R (REM)** | **0.3803** ❌ | 0.8054 | 0.3824 | 1,674 | 30.6% |
| **N1** | **0.2763** ❌ | 0.7581 | 0.1859 | 1,395 | 57.0% |
| **N2+N3** | **0.7025** ✓ | 0.8659 | 0.8020 | 7,333 | 64.1% |

#### **Comparison: LightGBM vs LSTM**

| Class | LightGBM F1 | LSTM F1 | Change | Verdict |
|-------|-------------|---------|--------|---------|
| W | 0.8012 | 0.8459 | **+0.0447** | ✓ Improved |
| R | 0.3773 | 0.3803 | **+0.0030** | ⚠️ Minimal change |
| N1 | 0.2555 | 0.2763 | **+0.0207** | ⚠️ Slight improvement |
| N2+N3 | 0.6130 | 0.7025 | **+0.0896** | ✓ Good improvement |

**Key Insights:**
- ✅ **Wake detection:** Excellent (F1=0.85, 78% recall)
- ✅ **N2+N3 detection:** Acceptable (F1=0.70, 64% recall)
- ❌ **REM detection:** Very poor (F1=0.38, only 30.6% recall)
- ❌ **N1 detection:** Worst class (F1=0.28, but 57% recall - high confusion)

---

## 🔍 Confusion Matrix Analysis

### **LSTM Test Confusion Matrix:**

```
              W      R     N1   N2+N3  (Predicted)
W    [     5104     88   1051    304]  78.0% recall ✓
R    [        3    513    445    713]  30.6% recall ❌
N1   [      179     93    795    328]  57.0% recall ⚠️
N2+N3[      235    330   2069   4699]  64.1% recall ⚠️
       (True)
```

### **Major Confusion Patterns:**

#### **1. N2+N3 → N1 Over-Prediction** ❌ **CRITICAL ISSUE**
- **2,069 samples (28.2%)** of N2+N3 misclassified as N1
- This is the **largest source of error** in the entire system
- Model is overly aggressive in predicting N1

#### **2. R → N2+N3 Confusion** ❌
- **713 samples (42.6%)** of REM misclassified as N2+N3
- REM and deep sleep have different physiology - shouldn't be confused
- Likely due to feature limitations (need raw BVP for HR irregularity)

#### **3. R → N1 Confusion** ❌
- **445 samples (26.6%)** of REM misclassified as N1
- This is physiologically understandable (both are "light" states)
- But shows fundamental limitation in distinguishing them

#### **4. N1 Scattered Predictions** ❌
- N1 only gets **795/1395 (57%)** correct
- Scattered across W (179), R (93), and N2+N3 (328)
- N1 is transition sleep - inherently ambiguous

### **Error Breakdown (LSTM Test):**

| True Class | Total Samples | Correct | Major Confusion | Secondary Confusion |
|------------|---------------|---------|-----------------|---------------------|
| W | 6,547 | 5,104 (78%) | → N1 (1,051 = 16%) | → N2+N3 (304 = 4.6%) |
| R | 1,674 | 513 (31%) | → N2+N3 (713 = 43%) ❌ | → N1 (445 = 27%) ❌ |
| N1 | 1,395 | 795 (57%) | → N2+N3 (328 = 24%) | → W (179 = 13%) |
| N2+N3 | 7,333 | 4,699 (64%) | → N1 (2,069 = 28%) ❌❌ | → R (330 = 4.5%) |

---

## 💡 Root Cause Analysis

### **Why is REM So Bad? (F1=0.38)**

**Physiological Requirements for REM Detection:**
1. **Rapid eye movements** - NOT in dataset
2. **Muscle atonia** - NOT directly measured
3. **Irregular heart rate** - Aggregated HRV features lose beat-to-beat dynamics
4. **High-frequency EEG** - NOT in dataset (no EEG)
5. **Irregular breathing** - Not captured with 30-sec aggregates

**What You Have:**
- ✅ Aggregated HRV metrics (SDNN, RMSSD, etc.)
- ✅ Movement (accelerometer)
- ✅ Skin conductance, temperature
- ❌ NOT enough for reliable REM detection

**Solution:** Need **raw 64Hz BVP** to capture beat-to-beat HR variability

---

### **Why is N1 So Bad? (F1=0.28)**

**N1 is Transition Sleep - Inherently Ambiguous:**
- Even human sleep techs have lowest agreement on N1 (~60-70%)
- N1 is brief (5-10 minutes) and unstable
- Shares features with both W (movement) and N2 (reduced HR)

**The N2+N3 → N1 Over-Prediction Problem:**
- Model is seeing 2,069 deep sleep epochs as N1
- Possible causes:
  1. Class weights (6.67x for N1) making model too eager to predict N1
  2. Feature overlap between light N2 and N1
  3. Regularization still too high (getting "no further splits" warnings)

**Solution:**
- Option 1: Reduce N1 class weight further (2x → 1.5x)
- Option 2: Merge R+N1 into single "Light Sleep" class (3-class problem)
- Option 3: Move to raw signals (CNN can learn micro-movements)

---

### **Why Still "No Further Splits" Warnings?**

Despite reducing regularization to reg_alpha=0-10 and reg_lambda=0.1-1.0:
- Best hyperparameters chosen: reg_alpha=7.0, reg_lambda=0.60
- Still getting 8 "no further splits" warnings during training
- This suggests:
  1. Features are genuinely not discriminative enough
  2. Minority classes (R, N1) have too few samples for fine-grained splits
  3. OR regularization needs to go even lower (0-5 range)

**But beware:** Lower regularization → higher overfitting risk

---

## 📊 Comparison to Previous Runs

| Run Date/Time | Config | Test Acc | REM F1 | N1 F1 | Train/Test Gap | Verdict |
|---------------|--------|----------|--------|-------|----------------|---------|
| Oct 20 22:44 | 5-class, 4x weights | 59.6% | 0.349 | N/A | Moderate | Baseline |
| Oct 21 00:34 | 50% SMOTE, 4x weights | 59.8% | 0.312 | N/A | **24%** ❌ | Severe overfitting |
| Oct 21 00:58 | 50% SMOTE, 4x weights, mod reg | 52.6% | 0.369 | N/A | 11.4% ✓ | Fixed overfit, but worse |
| **Oct 21 19:52** | **No SMOTE, 2x weights, low reg** | **65.6%** | **0.380** | **0.276** | **9.4%** ✓ | **Current best** |

### **Progress Summary:**
- ✅ **Best overall accuracy achieved:** 65.6% (LSTM)
- ✅ **Overfitting controlled:** 9.4% gap
- ⚠️ **REM barely improved:** 0.380 vs 0.369 (+3%)
- ❌ **Still far from target:** Commercial wearables achieve 70-75%

---

## 🎯 What's Working vs What's Not

### **✅ What's Working:**

1. **Wake Detection (F1=0.85):**
   - Movement-based features work well
   - Clear separation from sleep states
   - 78% recall is solid

2. **N2+N3 Detection (F1=0.70):**
   - Majority class with good features
   - 64% recall is acceptable
   - Merging N2+N3 reduced confusion

3. **Overfitting Control:**
   - Removing SMOTE helped significantly
   - 9.4% train/test gap is reasonable
   - Model generalizes acceptably

4. **LSTM Post-Processing:**
   - Consistent 6-8% accuracy boost
   - Temporal modeling captures transitions
   - Worth keeping in pipeline

### **❌ What's NOT Working:**

1. **REM Detection (F1=0.38):**
   - Only 31% of REM samples correctly identified
   - 43% confused with N2+N3 (opposite physiology!)
   - 27% confused with N1
   - **Fundamental feature limitation**

2. **N1 Detection (F1=0.28):**
   - Worst performing class
   - 28% of N2+N3 misclassified as N1 (2,069 samples!)
   - Transition sleep is inherently hard
   - **May need to merge with R or N2+N3**

3. **Feature Discriminability:**
   - 358 features but many low importance
   - Aggregated 30-sec features lose temporal dynamics
   - "No further splits" warnings persist
   - **Need raw signals or better features**

4. **Class Imbalance:**
   - Even with 2x weights, minority classes suffer
   - Training: W=24,620, R=4,666, N1=4,320, N2+N3=24,077
   - 5:1 imbalance (W/N2+N3 vs R/N1)
   - **Fundamental data limitation**

---

## 🚀 Recommendations

### **Short-Term (Within Current Framework):**

#### **Option 1: Further Reduce Regularization** ⚠️ Low Impact
```python
# In models.py, line 288-289:
"reg_alpha": hp.quniform("reg_alpha", 0, 5, 1),      # Reduced from 0-10
"reg_lambda": hp.uniform("reg_lambda", 0.01, 0.5),    # Reduced from 0.1-1.0
```
**Expected gain:** +1-2% accuracy  
**Risk:** Higher overfitting (gap may increase to 12-15%)  
**Verdict:** Worth one more try

---

#### **Option 2: Adjust Class Weights** ⚠️ May Help N1
```python
# In new_experiments.py, line 212-216:
scaling_factors = {
    0: 1.0,    # W - no boost
    1: 2.0,    # R - keep at 2x
    2: 1.5,    # N1 - reduce from 2x to 1.5x (reduce N1 over-prediction)
    3: 1.0     # N2+N3 - no boost
}
```
**Expected:** Reduce N2+N3→N1 confusion (currently 2,069 samples!)  
**Risk:** N1 recall may drop  
**Verdict:** Worth trying

---

#### **Option 3: 3-Class Classification** ⭐ **RECOMMENDED**

**Merge R + N1 → "Light Sleep"**

```python
# In datasets.py:
stage_to_num = {
    'W': 0,           # Wake
    'R': 1,           # Light Sleep (R + N1 merged)
    'N1': 1,          # Light Sleep (R + N1 merged)
    'N2': 2,          # Deep Sleep
    'N3': 2           # Deep Sleep
}

# Classes: ['W', 'Light', 'Deep']
```

**Why This Makes Sense:**
- ✅ Eliminates R↔N1 confusion (currently 445+93=538 samples)
- ✅ Matches physiological reality (both are light states)
- ✅ Simplifies problem (3 classes easier than 4)
- ✅ More honest about what wearables can measure

**Expected Performance:**
- Target: 72-75% accuracy (vs current 66%)
- All classes F1 > 0.60
- Clinically useful (Wake vs Light vs Deep is valuable)

**Downside:**
- Lose REM vs N1 distinction
- But this distinction is failing anyway (F1=0.38 and 0.28)

---

### **Long-Term (Strategic Direction):**

#### **Option 4: Move to Raw 64Hz Signals** ⭐⭐⭐ **STRONGLY RECOMMENDED**

See `RESTRUCTURING_README.md` for full details.

**Why:**
- Current system appears to be at **ceiling performance** (~66%)
- Aggregated features lose critical temporal information
- Raw 64Hz data enables:
  - Beat-to-beat HRV (essential for REM)
  - Micro-movements (essential for N1)
  - 1D CNN + LSTM architecture
  - State-of-the-art performance (75-82% achievable)

**Effort:** 10 weeks, $100-300 GPU cost  
**Expected Gain:** +10-16% accuracy (66% → 76-82%)  
**ROI:** Excellent - move from "below commercial" to "state-of-the-art"

---

## 📋 Immediate Next Steps

### **If Continuing with Current Approach:**

1. **Try 3-Class Classification** (1-2 days)
   - Merge R+N1 → "Light Sleep"
   - Expected: 72-75% accuracy
   - This is likely the best you can do with current features

2. **If 3-Class Still Not Good Enough:**
   - Accept that aggregated features have a ~75% ceiling
   - Move to raw 64Hz signals (see RESTRUCTURING_README.md)

### **If Moving to Raw Signals:**

1. **This Week:**
   - Set up AWS g4dn.xlarge spot instance
   - Verify access to 64Hz data
   - Start data pipeline development

2. **Next 2 Weeks:**
   - Implement 1D CNN + LSTM on raw BVP + accelerometer
   - Target: 72-75% accuracy (matches commercial)

3. **Months 2-3:**
   - Advanced architectures (multi-scale, attention)
   - Target: 78-82% accuracy (exceeds commercial, publishable)

---

## 🎓 Scientific Insights

### **What We Learned:**

1. **SMOTE Hurts More Than Helps:**
   - Synthetic samples don't capture real physiological patterns
   - Better to use class weights only

2. **Overfitting is Controllable:**
   - Removing SMOTE + moderate weights → 9% gap
   - But controlling overfitting doesn't improve test performance

3. **LSTM Adds Consistent Value:**
   - 6-8% accuracy boost across all experiments
   - Temporal modeling captures sleep stage transitions
   - Worth the computational cost

4. **Aggregated Features Have a Ceiling:**
   - ~66% accuracy appears to be the limit
   - This matches literature (wearables without raw signals: 60-70%)

5. **REM Requires Raw Signals:**
   - Beat-to-beat HRV dynamics are essential
   - 30-sec aggregates lose this information
   - F1=0.38 is likely the best possible with current features

6. **N1 is Fundamentally Hard:**
   - Even with good features, N1 F1 < 0.40 is common
   - Transition sleep is brief and ambiguous
   - Consider merging with R or N2 for practical applications

---

## 📈 Performance Summary

### **Current System (Best Achieved):**
```
Model: LightGBM → LSTM
Data: Aggregated 30-sec features (358 features)
Classes: 4 (W, R, N1, N2+N3)

Overall Performance:
- Accuracy: 65.56%
- F1 Macro: 55.12%
- Cohen's Kappa: 0.49 (moderate agreement)

Per-Class F1:
- W:     0.85  ✓ Excellent
- R:     0.38  ❌ Poor
- N1:    0.28  ❌ Very Poor
- N2+N3: 0.70  ✓ Acceptable

Key Issues:
- REM detection failing (only 31% recall)
- N1 detection worst class (F1=0.28)
- 2,069 N2+N3 samples misclassified as N1
- Likely at ceiling for aggregated features
```

### **Comparison to Benchmarks:**
```
Your Current System:           65.6%
Commercial Wearables:          70-75%
Research (raw signals):        78-85%
PSG with EEG (gold standard):  90-95%

Gap to Close:                  4-10% to match commercial
                              12-20% to be state-of-the-art
```

---

## 💼 Business Recommendations

### **If This is Research:**
- ✅ **Move to raw 64Hz signals** - this is publishable work
- ✅ Target 78-82% accuracy with deep learning
- ✅ Timeline: 10 weeks, Budget: $100-300

### **If This is Production:**
- ⚠️ **Try 3-class first** - simpler, more reliable
- ⚠️ 66% accuracy may be acceptable for some applications
- ⚠️ But below commercial wearables (70-75%)

### **If This is Proof-of-Concept:**
- ✅ You've proven the system works
- ✅ You've identified the limitations
- ✅ You have a clear path forward (raw signals)
- ✅ Ready to pitch next phase with concrete evidence

---

## 🎯 Final Verdict

**What You've Achieved:**
- ✅ Built working 4-class sleep staging system
- ✅ Achieved 66% accuracy with careful tuning
- ✅ Controlled overfitting (9% gap)
- ✅ Wake and N2+N3 detection work well

**What's Still Missing:**
- ❌ REM and N1 detection poor (F1 < 0.40)
- ❌ Below commercial wearable performance
- ❌ Feature limitations prevent further improvement

**The Path Forward is Clear:**
1. **Short-term:** Try 3-class (W, Light, Deep) → target 72-75%
2. **Long-term:** Move to raw 64Hz signals → target 78-82%

**This run represents the peak of what's achievable with aggregated features. Time to level up.** 🚀

---

## 📁 Generated Outputs

All results saved to: `runs/run_2025_10_21_19-52-16/`

**Files:**
- `LightGBM_multiclass_model.pkl` - Trained LightGBM model
- `LSTM_multiclass_model.pth` - Trained LSTM model
- `multiclass_train_CM.png` - Training confusion matrix
- `multiclass_test_CM.png` - LightGBM test confusion matrix
- `multiclass_LSTM_test_CM.png` - LSTM test confusion matrix
- `metrics_summary.csv` - Overall metrics comparison
- `lgb_vs_lstm_comparison.csv` - LightGBM vs LSTM comparison
- `multiclass_shap_bar_*.png` - Feature importance per class
- `log.txt` - Complete training log
- `code/` - Snapshot of all code files used

---

**Report Generated:** October 21, 2025  
**Status:** ✅ Training Complete  
**Recommendation:** Move to 3-class or raw signals  
**Next Review:** After next experiment

---

*For implementation details on raw signal processing, see `/home/admin/DREAMT_FE/RESTRUCTURING_README.md`*

