import torch
import torch.nn as nn
import numpy as np
import json

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# REALISTIC LIBROSA ACOUSTIC FEATURE SIMULATOR & DATASET GENERATION
# Grounded in physical Librosa DSP scales and Music Information Retrieval (MIR)
# ==============================================================================
N_SAMPLES = 16000
X_list = []
y_list = []

for _ in range(N_SAMPLES):
    mood = np.random.choice([0, 1, 2, 3])  # 0: Calm, 1: Energetic, 2: Happy, 3: Sad
    
    # 20 MFCCs matching real Librosa scales
    # MFCC[0] is physical log-energy (typically -500 to -180 in librosa)
    # MFCC[1] is spectral balance / tilt (typically 20 to 140)
    # MFCC[2..19] are cepstral coefficients (typically -50 to +50)
    mfccs = np.zeros(20, dtype=np.float32)
    mfccs[2:] = np.random.normal(0.0, 15.0, 18)
    
    if mood == 0:  # Calm / Chill 🌿
        # Low tempo, warm/low brightness, gentle dynamics
        tempo = np.random.uniform(55, 88)
        spec_cent = np.random.uniform(350, 1500)
        spec_rolloff = spec_cent * np.random.uniform(1.8, 2.4)
        zcr = np.random.uniform(0.015, 0.050)
        rms = np.random.uniform(0.02, 0.11)
        mfcc_var = np.random.uniform(5, 30)
        
        mfccs[0] = np.random.uniform(-420, -310)  # Quieter log energy
        mfccs[1] = np.random.uniform(30, 85)     # Warm low-frequency tilt
        mfccs[2] = np.random.uniform(-10, 20)

    elif mood == 1:  # Energetic / Hype 🔥
        # Fast tempo, high brightness, punchy transients, loud
        tempo = np.random.uniform(120, 175)
        spec_cent = np.random.uniform(2000, 4800)
        spec_rolloff = spec_cent * np.random.uniform(1.8, 2.6)
        zcr = np.random.uniform(0.060, 0.190)
        rms = np.random.uniform(0.12, 0.35)
        mfcc_var = np.random.uniform(35, 95)
        
        mfccs[0] = np.random.uniform(-270, -180)  # High loud energy
        mfccs[1] = np.random.uniform(80, 145)    # Sharp bright tilt
        mfccs[2] = np.random.uniform(-30, 10)
        mfccs[3:8] += np.random.uniform(5, 20)

    elif mood == 2:  # Happy / Upbeat ✨
        # Upbeat tempo, bright, lively, consonant
        tempo = np.random.uniform(108, 140)
        spec_cent = np.random.uniform(1700, 3600)
        spec_rolloff = spec_cent * np.random.uniform(1.8, 2.4)
        zcr = np.random.uniform(0.040, 0.100)
        rms = np.random.uniform(0.08, 0.22)
        mfcc_var = np.random.uniform(20, 60)
        
        mfccs[0] = np.random.uniform(-320, -220)  # Medium-high energy
        mfccs[1] = np.random.uniform(70, 125)    # Bright pleasant tilt
        mfccs[2] = np.random.uniform(0, 30)

    else:  # Sad / Melancholic 🌧️
        # Slow tempo, dark/subdued brightness, low transients, acoustic/piano-like
        tempo = np.random.uniform(50, 88)
        spec_cent = np.random.uniform(250, 1100)  # Very dark, warm, muted highs
        spec_rolloff = spec_cent * np.random.uniform(1.5, 2.1)
        zcr = np.random.uniform(0.010, 0.045)
        rms = np.random.uniform(0.03, 0.14)
        mfcc_var = np.random.uniform(4, 25)
        
        mfccs[0] = np.random.uniform(-440, -320)  # Low/medium energy
        mfccs[1] = np.random.uniform(20, 70)     # Subdued treble
        mfccs[2] = np.random.uniform(-25, 5)

    extra = np.array([spec_cent, spec_rolloff, zcr, rms, tempo, mfcc_var], dtype=np.float32)
    feature_vec = np.concatenate((mfccs, extra))
    X_list.append(feature_vec)
    y_list.append(mood)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

# ==============================================================================
# ACCURATE NORMALIZATION SCALER COMPUTATION
# ==============================================================================
mean_vals = X.mean(axis=0)
std_vals = X.std(axis=0) + 1e-6

print(f"Calculated Realistic Scaler Mean[0] (Log-Energy): {mean_vals[0]:.2f}, Std[0]: {std_vals[0]:.2f}")
print(f"Calculated Realistic Scaler Mean[20] (Centroid): {mean_vals[20]:.2f} Hz, Mean[24] (Tempo): {mean_vals[24]:.2f} BPM")

X_norm = (X - mean_vals) / std_vals
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split train & validation
split_idx = int(0.85 * len(X))
X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

# ==============================================================================
# PYTORCH DEEP NEURAL NETWORK ARCHITECTURE
# ==============================================================================
class AudioClassifierDNN(nn.Module):
    def __init__(self, input_dim=26, num_classes=4):
        super(AudioClassifierDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = AudioClassifierDNN(input_dim=26, num_classes=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
print("Beginning Full Deep Neural Network Training...")
model.train()
batch_size = 256
n_batches = len(X_train) // batch_size

for epoch in range(1, 201):
    perm = torch.randperm(len(X_train))
    epoch_loss = 0.0
    for b in range(n_batches):
        idx = perm[b*batch_size : (b+1)*batch_size]
        xb, yb = X_train[idx], y_train[idx]
        
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    scheduler.step()
    
    if epoch % 40 == 0 or epoch == 200:
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_acc = (val_out.argmax(dim=1) == y_val).float().mean()
            print(f"Epoch [{epoch:3d}/200] - Loss: {epoch_loss/n_batches:.4f} - Val Accuracy: {val_acc.item()*100:.2f}%")
        model.train()

# Final validation evaluation
model.eval()
with torch.no_grad():
    val_out = model(X_val)
    final_acc = (val_out.argmax(dim=1) == y_val).float().mean()
    print(f"\nFinal Deep Neural Network Accuracy: {final_acc.item()*100:.2f}%")

# Save model checkpoint and realistic scaler parameters
torch.save(model.state_dict(), "model_weights.pth")
scaler_dict = {
    "mean": mean_vals.tolist(),
    "std": std_vals.tolist()
}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_dict, f, indent=2)

print("Saved model_weights.pth and scaler_params.json successfully!")
