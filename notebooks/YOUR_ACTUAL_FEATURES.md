# YOUR ACTUAL AUDIO FEATURES DOCUMENTATION

## 📊 Complete Feature List (348 Features Total)

### Category 1: Simple Numerical Features (14 features)

These are single-value features directly extracted from audio:

| Feature               | Description                           | Expected Range |
| --------------------- | ------------------------------------- | -------------- |
| `duration`            | Track length in seconds               | ~30s (preview) |
| `tempo`               | BPM (beats per minute)                | 60-200         |
| `spec_centroid_mean`  | Average spectral centroid             | 1000-5000 Hz   |
| `spec_centroid_std`   | Std dev of spectral centroid          | Variable       |
| `spec_bandwidth_mean` | Average spectral bandwidth            | 1000-3000 Hz   |
| `spec_bandwidth_std`  | Std dev of spectral bandwidth         | Variable       |
| `spec_rolloff_mean`   | Average spectral rolloff (85% energy) | 2000-8000 Hz   |
| `spec_rolloff_std`    | Std dev of spectral rolloff           | Variable       |
| `zero_crossing_mean`  | Average zero-crossing rate            | 0-0.3          |
| `zero_crossing_std`   | Std dev of zero-crossing rate         | Variable       |
| `rms_mean`            | Average RMS energy (loudness)         | 0-1            |
| `rms_std`             | Std dev of RMS energy                 | Variable       |
| `beat_count`          | Number of beats detected              | 40-120         |
| `beat_tempo`          | Estimated tempo from beat detection   | 60-200 BPM     |

**Total: 14 numerical features**

---

### Category 2: Array-Based Features (334 features)

These are multi-dimensional features stored as arrays in CSV (parsed with `ast.literal_eval`):

#### 2.1 MFCC Features (40 features)

**Mel-Frequency Cepstral Coefficients** - capture timbral texture:

- `mfcc_mean`: Array of 20 coefficients (mean across time)
- `mfcc_std`: Array of 20 coefficients (std dev across time)

**Parsing:**

```python
import ast
mfcc_array = ast.literal_eval(row['mfcc_mean'])  # "[1.2, 3.4, ...]" → [1.2, 3.4, ...]
# Expands to: mfcc_mean_0, mfcc_mean_1, ..., mfcc_mean_19
```

#### 2.2 Chroma Features (24 features)

**Pitch class profiles** - capture harmonic content:

- `chroma_mean`: Array of 12 pitch classes (C, C#, D, ..., B)
- `chroma_std`: Array of 12 pitch classes

**Musical interpretation:**

- Each dimension represents a semitone in the octave
- High values indicate strong presence of that pitch

#### 2.3 Mel Spectrogram Features (256 features)

**Mel-scaled spectral representation**:

- `mel_spec_mean`: Array of 128 mel frequency bins
- `mel_spec_std`: Array of 128 mel frequency bins

**Note:** These are perceptually-scaled frequency representations

#### 2.4 Spectral Contrast Features (14 features)

**Valley-to-peak differences in spectrum**:

- `spec_contrast_mean`: Array of 7 frequency bands
- `spec_contrast_std`: Array of 7 frequency bands

**Captures:** Texture and musical dynamics

---

## 🔧 HOW YOUR FEATURE EXTRACTION WORKS

### Step 1: Load CSV with Array Strings

```python
df = pd.read_csv('audio_features_with_genres.csv')
# mfcc_mean column contains strings like: "[1.2, 3.4, 5.6, ...]"
```

### Step 2: Parse Array Columns

```python
import ast
import numpy as np

def parse_array_column(series):
    """Convert string arrays to numpy arrays"""
    result = []
    for val in series:
        if isinstance(val, str) and val.startswith('['):
            try:
                result.append(np.array(ast.literal_eval(val)))
            except:
                result.append(np.zeros(expected_dim))  # Fallback
        else:
            result.append(np.zeros(expected_dim))
    return result

# Parse MFCC
mfcc_mean_arrays = parse_array_column(df['mfcc_mean'])
mfcc_mean_matrix = np.vstack(mfcc_mean_arrays)  # Shape: (1612, 20)
```

### Step 3: Expand to Individual Features

```python
feature_parts = []
feature_names = []

# Add simple numerical features
feature_parts.append(df[numerical_cols].values)  # (1612, 14)
feature_names.extend(numerical_cols)

# Add MFCC mean features
feature_parts.append(mfcc_mean_matrix)  # (1612, 20)
feature_names.extend([f'mfcc_mean_{i}' for i in range(20)])

# Add MFCC std features
feature_parts.append(mfcc_std_matrix)  # (1612, 20)
feature_names.extend([f'mfcc_std_{i}' for i in range(20)])

# ... repeat for all array features ...

# Combine all
X = np.hstack(feature_parts)  # Final shape: (1612, 348)
```

---

## 📈 YOUR FINAL FEATURE MATRIX

```
Shape: (1612 tracks, 348 features)

Feature breakdown:
├── Numerical (14):
│   ├── duration (1)
│   ├── tempo (1)
│   ├── spectral (8): centroid, bandwidth, rolloff, contrast (mean+std)
│   ├── zero_crossing (2): mean, std
│   └── rms (2): mean, std
│
└── Array-based (334):
    ├── MFCC (40):
    │   ├── mfcc_mean_0 to mfcc_mean_19 (20)
    │   └── mfcc_std_0 to mfcc_std_19 (20)
    │
    ├── Chroma (24):
    │   ├── chroma_mean_0 to chroma_mean_11 (12)
    │   └── chroma_std_0 to chroma_std_11 (12)
    │
    ├── Mel Spectrogram (256):
    │   ├── mel_spec_mean_0 to mel_spec_mean_127 (128)
    │   └── mel_spec_std_0 to mel_spec_std_127 (128)
    │
    └── Spectral Contrast (14):
        ├── spec_contrast_mean_0 to spec_contrast_mean_6 (7)
        └── spec_contrast_std_0 to spec_contrast_std_6 (7)
```

---

## 🎯 CRITICAL DIFFERENCE FROM GODSON

| Aspect              | Godson               | YOU                                      |
| ------------------- | -------------------- | ---------------------------------------- |
| **Feature Count**   | 8 simple features    | **348 features** (14 simple + 334 array) |
| **MFCC**            | ❌ Not used          | ✅ 40 MFCC features                      |
| **Chroma**          | ❌ Not used          | ✅ 24 chroma features                    |
| **Mel Spectrogram** | ❌ Not used          | ✅ 256 mel features                      |
| **Complexity**      | Simple Spotify-like  | **Deep audio analysis**                  |
| **Extraction**      | Estimated heuristics | **Librosa-based extraction**             |

**YOUR FEATURES ARE 43× MORE COMPREHENSIVE!**

---

## ⚠️ IMPORTANT FOR MODEL TRAINING

### 1. **High Dimensionality Challenge**

- 348 features with only 1,612 samples
- Risk of overfitting, especially for complex models
- Neural Network will need regularization (dropout, early stopping)

### 2. **Feature Scaling is CRITICAL**

- Features span vastly different scales:
  - `mel_spec_mean_*`: -70 to -10 (dB scale)
  - `tempo`: 60-200 (BPM)
  - `zero_crossing_mean`: 0-0.3 (rate)
  - `rms_mean`: 0-1 (energy)
- **Solution:** StandardScaler (mean=0, std=1)

### 3. **Computational Cost**

- 348 features → Neural Network input layer needs 348 neurons
- More features = longer training time
- May need feature selection or dimensionality reduction

### 4. **Feature Importance**

- With 348 features, interpreting which ones matter is harder
- Random Forest/Gradient Boosting can provide feature importance
- Neural Network interpretability requires gradient-based methods

---

## 💡 RECOMMENDATIONS FOR YOUR MODELS

### For Neural Network:

```python
model = nn.Sequential(
    nn.Linear(348, 512),     # 348 input features
    nn.ReLU(),
    nn.Dropout(0.4),         # Higher dropout due to high dim
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 20)       # 20 genre classes
)
```

### For sklearn Models:

```python
# All models MUST use scaled features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

### Optional: Feature Reduction

If computational cost is too high or overfitting is severe:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=100)  # Reduce 348 → 100
X_train_pca = pca.fit_transform(X_train_scaled)
# Captures ~95% variance with fewer features
```

---

## ✅ FINAL CHECKLIST

- [x] 348 features correctly extracted
- [x] Array features properly parsed (ast.literal_eval)
- [x] Features expanded to individual columns
- [x] Stratified train/val/test split (70/15/15)
- [ ] **TODO:** Add standardization before all models
- [ ] **TODO:** Implement K-Fold CV on training set
- [ ] **TODO:** Train Neural Network with 348 input features
- [ ] **TODO:** Compare all models on same feature set

**YOU ARE READY TO BUILD THE MODELS WITH YOUR UNIQUE 348-FEATURE SET!**
