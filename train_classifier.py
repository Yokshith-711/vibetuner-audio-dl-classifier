import torch
import torch.nn as nn
import numpy as np
import json

torch.manual_seed(42)
np.random.seed(42)

# Generate 6,000 synthetic acoustic feature vectors grounded in Music Information Retrieval (MIR)
N = 6000
X = []
y = []

for _ in range(N):
    mood = np.random.choice([0, 1, 2, 3]) # 0: Calm, 1: Energetic, 2: Happy, 3: Sad
    mfccs = np.random.normal(0, 1.0, 20)
    
    if mood == 0:  # Calm / Chill 🌿
        tempo = np.random.uniform(55, 88)
        spec_cent = np.random.uniform(400, 1600)
        spec_rolloff = spec_cent * np.random.uniform(1.8, 2.4)
        zcr = np.random.uniform(0.015, 0.050)
        rms = np.random.uniform(0.02, 0.11)
        mfcc_var = np.random.uniform(5, 25)
        mfccs[0] = np.random.uniform(-10, -2)
        mfccs[1:5] += np.random.uniform(2, 6)

    elif mood == 1:  # Energetic / Hype 🔥
        tempo = np.random.uniform(120, 175)
        spec_cent = np.random.uniform(2200, 5200)
        spec_rolloff = spec_cent * np.random.uniform(1.8, 2.5)
        zcr = np.random.uniform(0.070, 0.200)
        rms = np.random.uniform(0.12, 0.35)
        mfcc_var = np.random.uniform(35, 95)
        mfccs[0] = np.random.uniform(6, 16)
        mfccs[5:15] += np.random.uniform(2, 8)

    elif mood == 2:  # Happy / Upbeat ✨
        tempo = np.random.uniform(105, 140)
        spec_cent = np.random.uniform(1800, 3800)
        spec_rolloff = spec_cent * np.random.uniform(1.8, 2.4)
        zcr = np.random.uniform(0.040, 0.100)
        rms = np.random.uniform(0.08, 0.22)
        mfcc_var = np.random.uniform(20, 60)
        mfccs[0] = np.random.uniform(1, 10)
        mfccs[2:8] += np.random.uniform(3, 7)

    else:  # Sad / Melancholic 🌧️
        tempo = np.random.uniform(48, 85)
        spec_cent = np.random.uniform(200, 1100)  # Very low brightness (mellow, warm, dark)
        spec_rolloff = spec_cent * np.random.uniform(1.4, 2.0)
        zcr = np.random.uniform(0.010, 0.042)
        rms = np.random.uniform(0.02, 0.15)
        mfcc_var = np.random.uniform(4, 22)
        mfccs[0] = np.random.uniform(-16, -3)
        mfccs[1:4] += np.random.uniform(3, 8)

    extra = np.array([spec_cent, spec_rolloff, zcr, rms, tempo, mfcc_var], dtype=np.float32)
    feature_vec = np.concatenate((mfccs, extra))
    X.append(feature_vec)
    y.append(mood)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Normalization parameters
mean_vals = X.mean(axis=0)
std_vals = X.std(axis=0) + 1e-6
X_norm = (X - mean_vals) / std_vals

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

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

model = AudioClassifierDNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Train
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    preds = model(X_tensor)
    loss = criterion(preds, y_tensor)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    acc = (model(X_tensor).argmax(dim=1) == y_tensor).float().mean()
    print(f"Model Training Complete! Benchmark Accuracy: {acc.item()*100:.2f}%")

# Save checkpoint and scaler parameters
torch.save(model.state_dict(), "model_weights.pth")

scaler_data = {
    "mean": mean_vals.tolist(),
    "std": std_vals.tolist()
}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_data, f, indent=2)

print("Saved model_weights.pth and scaler_params.json successfully!")
