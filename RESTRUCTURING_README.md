# Sleep Staging System Restructuring Plan
**Transition from Aggregated Features to Raw 64Hz Signal Processing**

---

## 📋 Table of Contents
1. [Current State](#current-state)
2. [The Opportunity: 64Hz Raw Data](#the-opportunity-64hz-raw-data)
3. [Recommended Architectures](#recommended-architectures)
4. [GPU Infrastructure Setup](#gpu-infrastructure-setup)
5. [Cost Analysis](#cost-analysis)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Expected Performance Improvements](#expected-performance-improvements)

---

## 🎯 Current State

### **Current System:**
- **Architecture:** Aggregated features → LightGBM → LSTM
- **Features:** 358 hand-engineered features per 30-second window
- **Performance:** 58-60% test accuracy
- **Key Issues:**
  - N1 detection poor (F1=0.26)
  - REM detection struggling (F1=0.38)
  - N1/R/N2+N3 heavily confused
  - Information loss from feature aggregation

### **Current Results (4-Class: W, R, N1, N2+N3):**
```
Test Accuracy: 58.78%
F1 Macro: 51.18%

Per-Class F1 Scores:
- W (Wake):     0.80  ✓ Good
- R (REM):      0.38  ❌ Poor
- N1:           0.26  ❌ Very Poor
- N2+N3:        0.61  ⚠️ Acceptable
```

### **Fundamental Limitation:**
Without raw signals, we can't capture:
- Beat-to-beat heart rate variability (HRV dynamics)
- Micro-movements and position changes
- Rapid physiological transitions
- High-frequency patterns in BVP/EDA

---

## 🚀 The Opportunity: 64Hz Raw Data

### **Available Dataset:**
- **Participants:** 100 subjects
- **Duration:** ~8 hours per subject = 800 hours total
- **Sampling Rate:** 64Hz
- **Signals:**
  - **BVP (Blood Volume Pulse)**: PPG signal - heart rate dynamics
  - **3-axis Accelerometer**: Movement and posture
  - **EDA (Electrodermal Activity)**: Skin conductance, arousal
  - **Temperature**: Body temperature, circadian rhythm

### **Data Volume:**
- 100 participants × 8 hours × 3600 sec/hour = **2.88 million seconds**
- @ 64Hz = **184 million data points per signal**
- Total: **~740 million samples** across 4 signal types
- **This is a research-grade dataset for deep learning**

### **What This Enables:**
✅ End-to-end deep learning on raw signals  
✅ Automatic feature learning (no manual engineering)  
✅ Capture beat-to-beat HRV patterns  
✅ Detect micro-arousals and position changes  
✅ Learn temporal dependencies across multiple time scales  
✅ State-of-the-art performance (75-85% accuracy achievable)  

---

## 🏗️ Recommended Architectures

### **Architecture 1: 1D CNN + LSTM** ⭐⭐⭐ **RECOMMENDED START**

**Why:** Simple, proven, achievable in 2-4 weeks

```
Input: 30-second windows @ 64Hz
├── BVP: [1920 samples]
├── Accel X/Y/Z: [1920 × 3 samples]
├── EDA: [1920 samples]
└── Temp: [1920 samples]
Total: ~12,000 values per window

↓ 1D Convolutional Layers (learn local patterns)
├── Conv1D(64 filters, kernel=7, stride=2)    # Captures 3-4 heartbeats
├── BatchNorm + ReLU + MaxPool(2)
├── Conv1D(128 filters, kernel=5, stride=2)   # Captures breathing cycles
├── BatchNorm + ReLU + MaxPool(2)
└── Conv1D(256 filters, kernel=3, stride=2)   # High-level features
    └── Output: [batch, 256, 120]

↓ LSTM Layers (temporal dependencies)
├── BiLSTM(256 units, 2 layers)               # Learn sleep stage transitions
└── Output: [batch, 512]

↓ Attention Layer (optional)
└── Learn which parts of 30-sec window matter most

↓ Classification Head
├── Dropout(0.5)
├── Dense(128) + ReLU
├── Dropout(0.3)
└── Dense(4) + Softmax
    └── Output: [W, R, N1, N2+N3] probabilities
```

**Expected Performance:**
- **Target Accuracy:** 72-75%
- **REM F1:** 0.60-0.65 (vs current 0.38)
- **N1 F1:** 0.40-0.50 (vs current 0.26)
- **Training Time:** 1-2 hours on g4dn.xlarge

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn

class SimpleSleepNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Multi-channel input: [batch, 6, 1920]
        # 6 channels: BVP, Accel_X, Accel_Y, Accel_Z, EDA, Temp
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: [batch, 6, 1920]
        x = self.conv_layers(x)  # → [batch, 256, 30]
        
        x = x.permute(0, 2, 1)  # → [batch, 30, 256]
        x, _ = self.lstm(x)      # → [batch, 30, 512]
        x = x[:, -1, :]          # Take last timestep
        
        return self.fc(x)        # → [batch, num_classes]
```

---

### **Architecture 2: Multi-Scale CNN (DeepSleepNet Style)** ⭐⭐

**Why:** Captures patterns at multiple time scales simultaneously

```
Input: Raw signals [batch, 6, 1920]

Path A: Small-scale (heartbeat level)
├── Conv1D(kernel=3-5) on BVP
└── Captures: Individual heartbeats, arrhythmias

Path B: Medium-scale (breathing level)
├── Conv1D(kernel=20-50) on BVP + Accel
└── Captures: Respiratory patterns, chest movement

Path C: Large-scale (movement patterns)
├── Conv1D(kernel=100-200) on Accel
└── Captures: Body movements, position changes

→ Concatenate all paths
→ BiLSTM for temporal modeling
→ Classification
```

**Expected Performance:**
- **Target Accuracy:** 78-82%
- **Training Time:** 2-3 hours on g4dn.xlarge
- **Complexity:** Medium (2-3 weeks to implement)

---

### **Architecture 3: ResNet1D + Attention** ⭐

**Why:** State-of-the-art, but more complex

```
Input → ResNet1D Blocks (skip connections) → Self-Attention → LSTM → Output
```

**Expected Performance:**
- **Target Accuracy:** 80-85%
- **Training Time:** 3-4 hours on g5.xlarge
- **Complexity:** High (4-6 weeks to implement)

---

### **Architecture 4: Hierarchical Classification** ⭐⭐⭐

**Why:** Matches sleep physiology, often more interpretable

```
Stage 1: Wake vs Sleep
├── Uses: Accelerometer (movement)
└── Expected: ~92% accuracy

Stage 2: REM vs NREM (if Sleep)
├── Uses: BVP (HRV irregularity), EDA
└── Expected: ~78% accuracy

Stage 3: N1 vs N2+N3 (if NREM)
├── Uses: BVP (HRV complexity), Accel (micro-movements)
└── Expected: ~72% accuracy

Combined Pipeline Accuracy: ~75-80%
```

**Pros:**
- ✅ Easier to debug (know which stage fails)
- ✅ Can optimize each stage independently
- ✅ Matches physiological differences
- ✅ Less confusion between R and N1

**Cons:**
- ⚠️ Error propagation (Stage 1 wrong → all wrong)
- ⚠️ More complex pipeline

---

## 💻 GPU Infrastructure Setup

### **Recommended Instance: g4dn.xlarge** ⭐⭐⭐

**Specifications:**
- **GPU:** NVIDIA T4 (16GB VRAM, 8.1 TFLOPS FP32, 65 TFLOPS FP16)
- **CPU:** 4 vCPUs (Intel Cascade Lake)
- **RAM:** 16GB
- **Storage:** 125GB NVMe SSD
- **Cost:** $0.526/hour on-demand, $0.158/hour spot (70% discount)

**Why This Instance?**
✅ Perfect for your workload (16GB VRAM sufficient)  
✅ Best $/performance ratio  
✅ 10x faster than CPU (1.5 hrs vs 15+ hrs training)  
✅ FP16 mixed precision support (2-3x additional speedup)  
✅ Excellent spot instance availability  

### **Alternative: g5.xlarge** (If Budget Allows)

**Specifications:**
- **GPU:** NVIDIA A10G (24GB VRAM, 125 TFLOPS FP16)
- **Cost:** $1.006/hour on-demand, $0.302/hour spot
- **Performance:** 2x faster than T4

**Use When:**
- Need faster iteration (30-45 min vs 1-2 hour training)
- Larger batch sizes (128 vs 64)
- Complex architectures (multi-scale, attention)

### **Comparison Table:**

| Instance | GPU | VRAM | On-Demand | Spot | Training Time* | Monthly Cost** |
|----------|-----|------|-----------|------|----------------|----------------|
| **g4dn.xlarge** ⭐ | T4 | 16GB | $0.526/hr | $0.158/hr | 1.5 hrs | $25-50 |
| **g5.xlarge** | A10G | 24GB | $1.006/hr | $0.302/hr | 0.75 hrs | $48-100 |
| **p3.2xlarge** ❌ | V100 | 16GB | $3.06/hr | $0.918/hr | 0.5 hrs | $150-300 |
| **CPU (m5.2xlarge)** ❌ | None | 0 | $0.384/hr | $0.115/hr | 15+ hrs | N/A |

*For 1D CNN + BiLSTM, 50 epochs on 100 participants  
**For 160 hours/month usage (8hr/day, 5 days/week)

---

## 💰 Cost Analysis

### **Development Phase (Month 1-2)**
**Activity:** Architecture development, initial experiments  
**Usage:** 4 hours/day, 5 days/week = 80 hours/month

**With g4dn.xlarge:**
- On-Demand: 80 hrs × $0.526 = **$42/month**
- Spot: 80 hrs × $0.158 = **$13/month** ⭐

### **Research Phase (Month 3-4)**
**Activity:** Hyperparameter search, architecture comparison  
**Usage:** 8 hours/day, 5 days/week = 160 hours/month

**With g4dn.xlarge:**
- On-Demand: 160 hrs × $0.526 = **$84/month**
- Spot: 160 hrs × $0.158 = **$25/month** ⭐

**With g5.xlarge:**
- On-Demand: 160 hrs × $1.006 = **$161/month**
- Spot: 160 hrs × $0.302 = **$48/month**

### **Production Training (Month 5+)**
**Activity:** Final model training, evaluation  
**Usage:** 2-3 runs/week = 40 hours/month

**With g4dn.xlarge spot:**
- 40 hrs × $0.158 = **$6/month** ⭐ Very cheap!

### **Total Project Cost Estimate (6 months):**
- **Conservative (spot):** $13 + $25 + $25 + $25 + $6 + $6 = **~$100**
- **Realistic (mixed):** $25 + $50 + $50 + $40 + $20 + $20 = **~$205**
- **Premium (g5.xlarge on-demand):** **~$800**

### **Cost per Experiment:**
- **g4dn.xlarge spot:** 1.5 hrs × $0.158 = **$0.24**
- **g4dn.xlarge on-demand:** 1.5 hrs × $0.526 = **$0.79**
- **g5.xlarge spot:** 0.75 hrs × $0.302 = **$0.23**

**You can run 100+ experiments for <$50!**

---

## 🛠️ Implementation Roadmap

### **Phase 1: Infrastructure Setup (Week 1)**

**Goals:**
- [ ] Set up AWS account and billing alerts
- [ ] Launch g4dn.xlarge spot instance
- [ ] Install PyTorch with CUDA support
- [ ] Verify GPU functionality
- [ ] Set up S3 bucket for data storage

**Deliverables:**
- Working GPU instance
- Data upload pipeline
- Basic PyTorch GPU test

**Time:** 1 day  
**Cost:** ~$1

---

### **Phase 2: Data Pipeline (Week 1-2)**

**Goals:**
- [ ] Load 64Hz raw signal data
- [ ] Implement 30-second windowing
- [ ] Create PyTorch Dataset class
- [ ] Implement DataLoader with batching
- [ ] Train/val/test split (70/10/20)
- [ ] Data augmentation (optional: jitter, scaling, time-shift)

**Code Structure:**
```python
class SleepDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, subject_ids, transform=None):
        """
        Load 64Hz signals: BVP, Accel, EDA, Temp
        Create 30-second windows with labels
        """
        self.signals = []  # [N_samples, 6, 1920]
        self.labels = []   # [N_samples]
        
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]
```

**Deliverables:**
- DataLoader that yields [batch, 6, 1920] tensors
- Verified data shapes and distributions
- Balanced train/val/test splits

**Time:** 3-5 days  
**Cost:** $2-5

---

### **Phase 3: Baseline Model (Week 2-3)**

**Goals:**
- [ ] Implement SimpleSleepNet (1D CNN + LSTM)
- [ ] Set up training loop with mixed precision
- [ ] Implement early stopping
- [ ] Add tensorboard logging
- [ ] Train first model

**Model Architecture:**
```python
SimpleSleepNet:
  - 3 Conv1D layers (64, 128, 256 filters)
  - 2-layer BiLSTM (256 units)
  - Classification head
  - Total params: ~2M
```

**Training Configuration:**
```python
optimizer = torch.optim.Adam(lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(patience=5)
loss = nn.CrossEntropyLoss(weight=class_weights)
epochs = 50
batch_size = 64
mixed_precision = True  # 2-3x speedup
```

**Success Criteria:**
- Model trains without errors
- Validation accuracy > 65% (improvement over 60% baseline)
- No severe overfitting (train/val gap < 15%)

**Time:** 5-7 days  
**Cost:** $10-20

---

### **Phase 4: Optimization & Tuning (Week 3-5)**

**Goals:**
- [ ] Hyperparameter search (learning rate, dropout, architecture)
- [ ] Class weight tuning for R and N1
- [ ] Add attention mechanism
- [ ] Try different optimizers (AdamW, RAdam)
- [ ] Implement gradient clipping
- [ ] Data augmentation experiments

**Hyperparameter Search Space:**
```python
search_space = {
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'hidden_size': [128, 256, 512],
    'num_lstm_layers': [1, 2, 3],
    'dropout': [0.3, 0.5, 0.7],
    'weight_decay': [1e-5, 1e-4, 1e-3],
    'batch_size': [32, 64, 128],
}
```

**Success Criteria:**
- Accuracy > 72%
- REM F1 > 0.55
- N1 F1 > 0.40

**Time:** 2-3 weeks  
**Cost:** $30-60

---

### **Phase 5: Advanced Architectures (Week 6-8)**

**Goals:**
- [ ] Implement multi-scale CNN (DeepSleepNet)
- [ ] Add self-attention layers
- [ ] Try hierarchical classification
- [ ] Ensemble multiple models
- [ ] Compare all approaches

**Architectures to Test:**
1. Multi-scale CNN + LSTM
2. ResNet1D + Attention
3. Hierarchical (W vs Sleep → REM vs NREM → N1 vs N2+N3)
4. Ensemble (combine top 3 models)

**Success Criteria:**
- Best model accuracy > 75%
- REM F1 > 0.60
- N1 F1 > 0.45
- Publishable results

**Time:** 2-3 weeks  
**Cost:** $40-80

---

### **Phase 6: Evaluation & Documentation (Week 9-10)**

**Goals:**
- [ ] Comprehensive evaluation on test set
- [ ] Per-subject analysis
- [ ] Confusion matrix analysis
- [ ] Feature importance (attention weights)
- [ ] Compare to baseline
- [ ] Write technical report

**Deliverables:**
- Final model weights
- Evaluation notebook
- Comparison with current system
- Technical documentation
- Performance report

**Time:** 1-2 weeks  
**Cost:** $10-20

---

### **Total Timeline: 10 weeks**
### **Total Budget: $100-200 (spot) or $300-500 (on-demand)**

---

## 📊 Expected Performance Improvements

### **Current System (Aggregated Features):**
```
Overall Accuracy: 58.78%
F1 Macro: 51.18%

Per-Class Performance:
├── W (Wake):      F1=0.80, Recall=73.1%  ✓ Good
├── R (REM):       F1=0.38, Recall=46.5%  ❌ Poor
├── N1:            F1=0.26, Recall=54.5%  ❌ Very Poor
└── N2+N3:         F1=0.61, Recall=49.6%  ⚠️ Acceptable

Key Issues:
- Heavy R ↔ N1 confusion (25% each direction)
- N2+N3 → N1 over-prediction (30%)
- Fundamental information loss from aggregation
```

### **Expected: Simple 1D CNN + LSTM**
```
Overall Accuracy: 72-75%  (+14-16%)
F1 Macro: 65-68%

Per-Class Performance:
├── W (Wake):      F1=0.85, Recall=82%  ✓ Better
├── R (REM):       F1=0.60, Recall=65%  ✓ Much better (+58% improvement)
├── N1:            F1=0.42, Recall=48%  ✓ Better (+62% improvement)
└── N2+N3:         F1=0.72, Recall=70%  ✓ Better

Improvements:
- Better REM detection (BVP captures HR irregularity)
- Better N1 detection (accel captures micro-movements)
- Less confusion overall
```

### **Expected: Advanced (Multi-Scale CNN + Attention)**
```
Overall Accuracy: 78-82%  (+20-23%)
F1 Macro: 72-76%

Per-Class Performance:
├── W (Wake):      F1=0.90, Recall=88%  ✓ Excellent
├── R (REM):       F1=0.68, Recall=72%  ✓ Good (+79% improvement)
├── N1:            F1=0.50, Recall=58%  ✓ Acceptable (+92% improvement)
└── N2+N3:         F1=0.78, Recall=78%  ✓ Good

Breakthrough:
- REM vs N1 distinction much clearer
- Captures multi-timescale patterns
- Near state-of-the-art for wearables
```

### **Comparison to Commercial Wearables:**
```
Fitbit / Apple Watch / Oura Ring: 65-72% vs PSG
Your Current System:               59%
Target (Simple CNN):               72-75%  (matches commercial!)
Target (Advanced):                 78-82%  (exceeds commercial!)
```

### **Comparison to Research State-of-the-Art:**
```
Best published (wearables only, no EEG): 85%
With EEG (PSG gold standard):            90-95%
Your realistic target:                   78-82%
```

**This would be publication-worthy research!**

---

## 🚀 Quick Start Guide

### **Step 1: Launch AWS Instance**

```bash
# Choose instance
Instance type: g4dn.xlarge
AMI: Deep Learning AMI (Ubuntu 20.04) - ami-xxxxx
Storage: 100GB EBS gp3
Region: us-east-1 (best spot availability)

# For spot instance
Pricing: Request spot, max price $0.20/hr
```

### **Step 2: Connect & Setup**

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Check GPU
nvidia-smi
# Should show: Tesla T4, 16GB

# Activate PyTorch environment
source activate pytorch

# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
# Should print: True

# Clone your repo
git clone <your-repo>
cd DREAMT_FE

# Install dependencies
pip install -r requirements.txt
```

### **Step 3: Test GPU Speed**

```python
import torch
import time

device = torch.device('cuda')

# Test tensor operations
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

start = time.time()
for _ in range(1000):
    z = torch.mm(x, y)
torch.cuda.synchronize()
print(f"GPU time: {time.time() - start:.2f}s")

# Expected: < 1 second (vs 10+ seconds on CPU)
```

### **Step 4: Enable Mixed Precision**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        with autocast():  # Use FP16 where safe
            outputs = model(batch)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# 2-4x speedup for "free"
```

### **Step 5: Implement Checkpointing**

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

# Save every epoch (for spot instance recovery)
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    save_checkpoint(model, optimizer, epoch, train_loss, 'checkpoint.pth')
```

---

## 📈 Success Metrics

### **Phase 1 Success (Week 3):**
- ✅ GPU instance running
- ✅ Data pipeline working
- ✅ First model training
- ✅ Validation accuracy > 65%

### **Phase 2 Success (Week 6):**
- ✅ Optimized hyperparameters
- ✅ Accuracy > 72%
- ✅ REM F1 > 0.55
- ✅ N1 F1 > 0.40

### **Phase 3 Success (Week 10):**
- ✅ Advanced architecture implemented
- ✅ Accuracy > 75%
- ✅ All classes F1 > 0.50
- ✅ Publishable results

---

## 🎯 Next Steps

1. **This Week:**
   - [ ] Review this plan with team
   - [ ] Get AWS account approval
   - [ ] Launch test g4dn.xlarge instance
   - [ ] Verify 64Hz data access

2. **Next Week:**
   - [ ] Upload data to S3
   - [ ] Build data loading pipeline
   - [ ] Implement first CNN model

3. **Month 1:**
   - [ ] Complete Phase 1-3 (baseline model)
   - [ ] Report initial results
   - [ ] Decide on Phase 4 approach

---

## 📚 References & Resources

### **Papers:**
1. DeepSleepNet (2017) - Multi-scale CNN for sleep staging
2. SeqSleepNet (2019) - LSTM for sleep sequence modeling
3. U-Time (2020) - U-Net inspired architecture for sleep staging
4. SleepTransformer (2021) - Attention mechanisms for sleep

### **Code Repositories:**
- pytorch/examples: CNN baselines
- akaraspt/deepsleepnet: Original implementation
- RobRomijnders/sleep_staging: Clean PyTorch implementation

### **Datasets:**
- PhysioNet Sleep-EDF (for comparison)
- MESA (multi-ethnic study, wearable data)
- Your dataset: 100 participants @ 64Hz ⭐

---

## 💡 Key Takeaways

1. **64Hz raw data is a game-changer** - enables proper deep learning
2. **GPU is essential** - 10x speedup, enables rapid iteration
3. **g4dn.xlarge spot is perfect** - $25-50/month for professional setup
4. **Expected improvement: 59% → 75-80%** - matches/exceeds commercial wearables
5. **Timeline: 10 weeks** - achievable with focused effort
6. **Budget: $100-300** - extremely cost-effective for the value

---

**Questions or need help getting started? Let's do this! 🚀**

---

*Last Updated: October 21, 2025*  
*Contact: [Your Team]*

