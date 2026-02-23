# Learning Probability Density Function using GAN

**Roll Number:** 102303505

## 1️⃣ Transformation Parameters

```
r = 102303505
a_r = 0.5 × (r mod 7) = 1.5
b_r = 0.3 × (r mod 5 + 1) = 0.3

```

**Transformation Function:**

z = x + 1.5 × sin(0.3 × x)

where **x** is the NO₂ concentration and **z** is the transformed variable.

---

## 2️⃣ GAN Architecture

### Generator
**Input:** Noise ε ~ N(0,1), dimension = 32

**Architecture:**
- Linear(32 → 64) + ReLU
- Linear(64 → 128) + ReLU
- Linear(128 → 64) + ReLU
- Linear(64 → 1) + Tanh

### Discriminator
**Input:** z (real or generated)

**Architecture:**
- Linear(1 → 64) + LeakyReLU
- Linear(64 → 128) + LeakyReLU
- Linear(128 → 64) + LeakyReLU
- Linear(64 → 1) + Sigmoid

---

## 3️⃣ Training Configuration

- **Optimizer:** Adam (lr = 0.001)
- **Loss Function:** Binary Cross-Entropy
- **Batch Size:** 256
- **Epochs:** 50
- **Noise Distribution:** N(0,1)
- **Dataset:** India Air Quality NO₂ data (subset of 50,000 samples used for training)

---

## 4️⃣ PDF Estimation

After training:
- 30,000 samples were generated using the trained generator
- Kernel Density Estimation (KDE) was used to approximate the learned probability density
- Real and generated PDFs were compared visually and statistically

---

## 5️⃣ Results

### Statistical Metrics (From Execution)

```
KS Statistic = 0.174
Wasserstein Distance = 4.04
```

### Interpretation

✅ **The GAN successfully captured the overall shape of the transformed distribution**
- The generated distribution follows the main mode of the real data
- Some differences exist in the tails, as reflected by the Wasserstein distance
- Training remained stable for 50 epochs without mode collapse

---

## 6️⃣ Generated Plots

### Architecture Diagram
<img src="architecture_diagram.png" width="700">

### Training Progress
<img src="training_progress.png" width="700">

### Learned PDF Comparison
<img src="pdf_comparison.png" width="700">

### Distribution Analysis
<img src="distribution_analysis.png" width="800">

**These plots show:**
- GAN structure and configuration
- Training loss curves
- Real vs generated PDF comparison
- Q-Q plot and statistical summary

---

## Observations

### 1. Mode Coverage ✓
- **Primary mode** successfully captured by the generator
- Generated distribution covers the main range of real data
- No mode collapse observed throughout training

### 2. Training Stability ✓
- **Loss convergence:** Both D and G losses stabilized smoothly
- Training completed without oscillations or divergence
- Stable training for 50 epochs

### 3. Quality of Generated Distribution
- **KS Statistic (0.174):** Indicates reasonable similarity between distributions
- **Wasserstein Distance (4.04):** Shows some differences, particularly in tails
- **Visual Quality:** KDE curves show good overlap in main distribution region

---

## 7️⃣ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the GAN training and generate plots
python main.py
```

**Expected Runtime:** 1-2 minutes

**Generated Files:**
- `architecture_diagram.png` - GAN architecture visualization
- `training_progress.png` - Loss curves and training stability
- `pdf_comparison.png` - Main PDF learning result 
- `distribution_analysis.png` - Comprehensive statistical analysis

---

**Dataset Source:** [India Air Quality Data - Kaggle](https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data)
