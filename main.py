import copy
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

WIN = 15          # temporal window (odd). Larger = more noise averaging, more signal blur
CAUSAL = False    # True = use only past samples (needed for real-time deployment)

def make_windows(X, win=WIN, causal=CAUSAL):
    """(n, F) -> (n, win*F). Each row = the window of samples around/ending at that row."""
    left, right = (win - 1, 0) if causal else (win // 2, win // 2)
    Xp = np.pad(X, ((left, right), (0, 0)), mode='edge')
    idx = np.arange(len(X))[:, None] + np.arange(win)[None, :]
    return Xp[idx].reshape(len(X), -1)

BOTTLENECK = 8    # was 3
PCA_K = 3   

np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# STEP 1: DATA ACQUISITION & PREPROCESSING
# ==========================================
print("Step 1: Loading and preprocessing data...")
# The two CSVs have different row counts and duplicate timestamps, so they are
# joined on DeviceTimeStamp (NOT by row position).
df_cv = pd.read_csv('data/CurrentVoltage.csv', parse_dates=['DeviceTimeStamp']).drop_duplicates('DeviceTimeStamp')
df_overview = pd.read_csv('data/Overview.csv', parse_dates=['DeviceTimeStamp']).drop_duplicates('DeviceTimeStamp')
df_merged = df_cv.merge(df_overview, on='DeviceTimeStamp')

# Drop outage / zero rows and sensor glitches (e.g. OTI = 250)
df_merged = df_merged[(df_merged.IL1 > 5) & (df_merged.VL1 > 100) &
                      (df_merged.OTI > 0) & (df_merged.OTI < 100) &
                      (df_merged.ATI > 0)].sort_values('DeviceTimeStamp')

# Inputs: 3 phase currents, 3 phase voltages, ambient temperature indicator
# Target: top-oil temperature (WTI in this file is a 0/1 flag, not a temperature)
FEATURES = ['IL1', 'IL2', 'IL3', 'VL1', 'VL2', 'VL3', 'ATI']
TARGETS = ['OTI']
TARGET_NAMES = TARGETS
INPUT_DIM = len(FEATURES)
OUTPUT_DIM = len(TARGETS)

df_combined = df_merged[FEATURES + TARGETS].dropna().reset_index(drop=True)
print(f"Usable rows after merge + cleaning: {len(df_combined)}")

# Chronological 70/15/15 split: 15-minute samples are strongly autocorrelated,
# so a random split would leak neighbouring rows from train into test.
n_total = len(df_combined)
i_tr, i_va = int(0.70 * n_total), int(0.85 * n_total)
train_data = df_combined.iloc[:i_tr]
val_data = df_combined.iloc[i_tr:i_va]
test_data = df_combined.iloc[i_va:]

X_train, y_train = train_data.iloc[:, :INPUT_DIM].values, train_data.iloc[:, INPUT_DIM:].values
X_val, y_val = val_data.iloc[:, :INPUT_DIM].values, val_data.iloc[:, INPUT_DIM:].values
X_test, y_test = test_data.iloc[:, :INPUT_DIM].values, test_data.iloc[:, INPUT_DIM:].values

# Scalers are fitted on the TRAIN split only (no leakage into val/test)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_clean = scaler_X.fit_transform(X_train)
X_val_clean = scaler_X.transform(X_val)
X_test_clean = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

print(f"Rows -> train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

# ==========================================
# STEP 2: STOCHASTIC NOISE INJECTION
# ==========================================
print("Step 2: Injecting White Gaussian Random Noise...")
def inject_noise(features, snr_db, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    signal_power = np.mean(features ** 2, axis=0)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=features.shape)
    return features + noise

def make_noisy_splits(snr_db, seed=SEED):
    """Fixed-seed noisy copies of train/val/test so every run is reproducible."""
    return (inject_noise(X_train_clean, snr_db, np.random.default_rng(seed * 10)),
            inject_noise(X_val_clean,   snr_db, np.random.default_rng(seed * 10 + 1)),
            inject_noise(X_test_clean,  snr_db, np.random.default_rng(seed * 10 + 2)))

# LOWERED TO 10dB FOR HEAVY NOISE STRESS-TESTING
SNR_DB = 10
X_train_noisy, X_val_noisy, X_test_noisy = make_noisy_splits(SNR_DB)

# ==========================================
# STEP 3: DENOISING AUTOENCODER (DAE) FILTER
# ==========================================
print("Step 3: Initializing Deep Denoising Autoencoder...")
BOTTLENECK = 3   # PCA shows ~3 components carry the signal in these channels

class DAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM * WIN, out_dim=INPUT_DIM, bottleneck=BOTTLENECK):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# STEP 4: FEED-FORWARD NEURAL NETWORK (FNN)
# ==========================================
print("Step 4: Initializing Feed-Forward Neural Network Predictor...")
class FNN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
        super(FNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# STEP 5: TWO-PHASE SYSTEM INTEGRATION
# ==========================================
DAE_EPOCHS = 100
FNN_EPOCHS = 50
PATIENCE = 15   # early stopping on validation loss

def train_dae(snr_db, epochs=DAE_EPOCHS, verbose=True):
    model = DAE()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(SEED)

    sig_pow = np.mean(X_train_clean ** 2, axis=0)
    noise_std = np.sqrt(sig_pow / (10 ** (snr_db / 10)))
    Xc = torch.FloatTensor(X_train_clean)
    Xv_clean = torch.FloatTensor(X_val_clean)

    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        noisy = X_train_clean + rng.normal(size=X_train_clean.shape) * noise_std
        loader = DataLoader(TensorDataset(torch.FloatTensor(make_windows(noisy)), Xc),
                            batch_size=64, shuffle=True)
        total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)   # noisy window -> clean centre sample
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            Xv_noisy = X_val_clean + rng.normal(size=X_val_clean.shape) * noise_std
            val_loss = criterion(model(torch.FloatTensor(make_windows(Xv_noisy))), Xv_clean).item()

        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if verbose and (epoch + 1) % 20 == 0:
            print(f"DAE Epoch [{epoch+1}/{epochs}] | Train Loss: {total/len(loader):.4f} | Val Loss: {val_loss:.4f}")
        if wait >= PATIENCE:
            if verbose:
                print(f"DAE early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model

def denoise(dae, X):
    with torch.no_grad():
        return dae(torch.FloatTensor(make_windows(X))).numpy()

def train_fnn(X_tr, y_tr, X_va, y_va, epochs=FNN_EPOCHS, verbose=True):
    """Phase 2: features -> steady-state temperatures (normalised space)."""
    model = FNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
                        batch_size=64, shuffle=True)
    Xv, yv = torch.FloatTensor(X_va), torch.FloatTensor(y_va)

    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if verbose and (epoch + 1) % 10 == 0:
            print(f"FNN Epoch [{epoch+1}/{epochs}] | Train Loss: {total/len(loader):.4f} | Val Loss: {val_loss:.4f}")
        if wait >= PATIENCE:
            if verbose:
                print(f"FNN early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model

def predict(model, X):
    with torch.no_grad():
        return model(torch.FloatTensor(X)).numpy()

def evaluate(pred_scaled, y_scaled):
    """Metrics in real units (deg C) after inverse scaling."""
    p = scaler_y.inverse_transform(pred_scaled)
    t = scaler_y.inverse_transform(y_scaled)
    mse = mean_squared_error(t, p)
    return {"MSE": mse, "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(t, p), "R2": r2_score(t, p)}

print(f"Step 5A: Pre-training the DAE Filter (Isolation Phase) at {SNR_DB} dB SNR...")
dae_model = train_dae(SNR_DB, verbose=True)

print("\nStep 5B: Training the FNN Predictor on Cleaned Data...")
# DAE is frozen (eval mode, no gradients): it acts as a static filter.
cleaned_train = denoise(dae_model, X_train_noisy)
cleaned_val = denoise(dae_model, X_val_noisy)
cleaned_test = denoise(dae_model, X_test_noisy)
fnn_model = train_fnn(cleaned_train, y_train_scaled, cleaned_val, y_val_scaled, verbose=True)

# Benchmarking Robustness on the Test Set
print("\nEvaluating pipeline robustness on noisy test data...")
final_test_predictions = predict(fnn_model, cleaned_test)
test_loss = mean_squared_error(y_test_scaled, final_test_predictions)
print(f"Final Pipeline Test MSE (Normalized): {test_loss:.4f}")

# ==========================================
# STEP 6: ADVANCED PERFORMANCE METRICS
# ==========================================
print("\n==========================================")
print("STEP 6: ADVANCED PERFORMANCE METRICS")
print("==========================================")

hybrid = evaluate(final_test_predictions, y_test_scaled)

print("\n--- Phase 2: FNN Predictor (Thermal Estimation) ---")
print(f"Mean Squared Error (MSE):       {hybrid['MSE']:.4f} °C²")
print(f"Root Mean Squared Error (RMSE): {hybrid['RMSE']:.4f} °C")
print(f"Mean Absolute Error (MAE):      {hybrid['MAE']:.4f} °C")
print(f"R² Score:                       {hybrid['R2']:.4f}")

final_preds_actual = scaler_y.inverse_transform(final_test_predictions)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)
for i, name in enumerate(TARGET_NAMES):
    rmse_i = np.sqrt(mean_squared_error(y_test_actual[:, i], final_preds_actual[:, i]))
    print(f"  RMSE [{name}]: {rmse_i:.4f} °C")

def calculate_snr(clean_signal, corrupted_signal):
    noise = clean_signal - corrupted_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

# --- diagnostic: best SNR a plain moving average could reach (no noise involved) ---
from scipy.ndimage import uniform_filter1d     # (or move this import to the top)
sm = uniform_filter1d(X_test_clean, size=WIN, axis=0, mode='nearest')
print(f"Smoothing floor @ WIN={WIN}: {calculate_snr(X_test_clean, sm):.2f} dB")

input_snr = calculate_snr(X_test_clean, X_test_noisy)
output_snr = calculate_snr(X_test_clean, cleaned_test)

print("\n--- Phase 1: DAE Filter (Noise Reduction) ---")
print(f"Input Signal-to-Noise Ratio:    {input_snr:.2f} dB")
print(f"Output Signal-to-Noise Ratio:   {output_snr:.2f} dB")
print(f"Net SNR Improvement:            {output_snr - input_snr:.2f} dB")

# Linear reference: PCA with the same number of components as the DAE bottleneck
pca = PCA(n_components=PCA_K).fit(X_train_clean)
print(f"PCA (k={PCA_K}) reference output SNR: ...")

# ==========================================
# STEP 7: BASELINE COMPARISON (OBJECTIVES 1 & 4)
# ==========================================
# Seed-to-seed variation on this dataset (~0.3 °C) is as large as the gap between
# methods, so every comparison is repeated over N_SEEDS and reported as mean ± std.
N_SEEDS = 5
SNR_LEVELS = [30, 20, 15, 10, 5, 0]
assert SNR_DB in SNR_LEVELS

print("\n==========================================")
print(f"STEP 7: BASELINE vs HYBRID ({N_SEEDS} seeds x {len(SNR_LEVELS)} SNR levels, may take several minutes)")
print("==========================================")
records = []
for seed in range(N_SEEDS):
    np.random.seed(seed); torch.manual_seed(seed)
    # (a) Upper bound: FNN trained and tested on clean data
    fnn_clean = train_fnn(X_train_clean, y_train_scaled, X_val_clean, y_val_scaled, verbose=False)
    m_up = evaluate(predict(fnn_clean, X_test_clean), y_test_scaled)

    for snr in SNR_LEVELS:
        np.random.seed(seed); torch.manual_seed(seed)
        Xtr_n, Xva_n, Xte_n = make_noisy_splits(snr, seed)
        dae = train_dae(snr, verbose=False)
        Xte_d = denoise(dae, Xte_n)
        # (b) Problem case: clean-trained FNN deployed on noisy measurements
        m_cn = evaluate(predict(fnn_clean, Xte_n), y_test_scaled)
        # (c) Noise-aware FNN with no filter: trained and tested on noisy data
        f_n = train_fnn(Xtr_n, y_train_scaled, Xva_n, y_val_scaled, verbose=False)
        m_n = evaluate(predict(f_n, Xte_n), y_test_scaled)
        # (d) Proposed: frozen DAE filter -> FNN
        f_h = train_fnn(denoise(dae, Xtr_n), y_train_scaled, denoise(dae, Xva_n), y_val_scaled, verbose=False)
        m_h = evaluate(predict(f_h, Xte_d), y_test_scaled)

        row = {"seed": seed, "SNR_dB": snr,
               "DAE_out_SNR_dB": calculate_snr(X_test_clean, Xte_d),
               "PCA_out_SNR_dB": calculate_snr(X_test_clean, pca.inverse_transform(pca.transform(Xte_n)))}
        for key, m in [("upper_bound", m_up), ("clean_FNN_on_noisy", m_cn),
                       ("noisy_FNN", m_n), ("DAE_FNN", m_h)]:
            for k, v in m.items():
                row[f"{key}_{k}"] = v
        records.append(row)
    print(f"seed {seed + 1}/{N_SEEDS} done", flush=True)

df_runs = pd.DataFrame(records)
MODELS = [("upper_bound", "FNN, clean train / clean test (upper bound)"),
          ("clean_FNN_on_noisy", "FNN, clean train / noisy test (no defence)"),
          ("noisy_FNN", "FNN, noisy train / noisy test (no filter)"),
          ("DAE_FNN", "DAE + FNN (proposed)")]

sub = df_runs[df_runs.SNR_dB == SNR_DB]
print(f"\nResults @ {SNR_DB} dB SNR (mean ± std over {N_SEEDS} seeds)")
print(f"{'Model':<46}{'RMSE (°C)':>16}{'MAE (°C)':>16}{'R²':>16}")
for key, name in MODELS:
    cells = [f"{sub[f'{key}_{k}'].mean():.3f} ± {sub[f'{key}_{k}'].std():.3f}" for k in ("RMSE", "MAE", "R2")]
    print(f"{name:<46}{cells[0]:>16}{cells[1]:>16}{cells[2]:>16}")

# ==========================================
# STEP 8: SNR SWEEP (FINDING THE BREAKING POINT)
# ==========================================
print("\n==========================================")
print("STEP 8: SNR SWEEP (test RMSE in °C, mean ± std)")
print("==========================================")
os.makedirs("results", exist_ok=True)
df_runs.to_csv("results/all_runs.csv", index=False)

sweep_rows = []
for snr in SNR_LEVELS:
    g = df_runs[df_runs.SNR_dB == snr]
    row = {"SNR_dB": snr,
           "DAE_out_SNR_dB": g.DAE_out_SNR_dB.mean(), "PCA_out_SNR_dB": g.PCA_out_SNR_dB.mean()}
    for key, _ in MODELS:
        row[f"{key}_RMSE_mean"] = g[f"{key}_RMSE"].mean()
        row[f"{key}_RMSE_std"] = g[f"{key}_RMSE"].std()
    sweep_rows.append(row)
df_sweep = pd.DataFrame(sweep_rows)
df_sweep.to_csv("results/snr_sweep.csv", index=False)

print(f"{'SNR':>4}{'DAE SNR out':>13}{'PCA SNR out':>13}{'clean FNN':>16}{'noisy FNN':>16}{'DAE+FNN':>16}")
for _, r in df_sweep.iterrows():
    fmt = lambda k: f"{r[k + '_RMSE_mean']:.2f} ± {r[k + '_RMSE_std']:.2f}"
    print(f"{int(r.SNR_dB):>4}{r.DAE_out_SNR_dB:>13.2f}{r.PCA_out_SNR_dB:>13.2f}"
          f"{fmt('clean_FNN_on_noisy'):>16}{fmt('noisy_FNN'):>16}{fmt('DAE_FNN'):>16}")

plt.figure(figsize=(7, 4.5))
for key, label, style in [("clean_FNN_on_noisy", "FNN (clean-trained)", "o--"),
                          ("noisy_FNN", "FNN (noisy-trained)", "s--"),
                          ("DAE_FNN", "DAE + FNN", "^-")]:
    plt.errorbar(df_sweep["SNR_dB"], df_sweep[f"{key}_RMSE_mean"],
                 yerr=df_sweep[f"{key}_RMSE_std"], fmt=style, capsize=3, label=label)
plt.axhline(df_runs["upper_bound_RMSE"].mean(), color="gray", linestyle=":", label="Clean upper bound")
plt.gca().invert_xaxis()
plt.xlabel("Input SNR (dB)")
plt.ylabel("Test RMSE (°C)")
plt.title(f"Robustness to broadband noise ({N_SEEDS} seeds)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/snr_sweep.png", dpi=150)
print("\nSaved results/all_runs.csv, results/snr_sweep.csv and results/snr_sweep.png")