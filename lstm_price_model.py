"""
LSTM Price Prediction Model for Erode Turmeric
================================================
Predicts the next 30-day average modal price from the last 60 days
of price and weather features.

Input: erode_turmeric_aligned.csv
Output: lstm_best_model.pth, lstm_training_loss.png, lstm_test_eval.png, lstm_forecast_summary.txt
"""

# ── Requirements ────────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch", "scikit-learn", "pandas", "numpy", "matplotlib"])

import os, warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    print("ERROR: PyTorch is not available. Please install it with: pip install torch")
    sys.exit(1)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ─────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# STEP 1 — Load and prepare
# ============================================================================
print("=" * 70)
print("STEP 1 — Load and prepare")
print("=" * 70)

df = pd.read_csv(os.path.join(BASE_DIR, "erode_turmeric_aligned.csv"), parse_dates=["arrival_date"])
df = df.sort_values("arrival_date").reset_index(drop=True)

# Drop rows where modal_price is null
df = df.dropna(subset=["modal_price"]).reset_index(drop=True)

# Feature columns
FEATURES = [
    "modal_price", "rolling_avg_7d", "rolling_avg_30d", "price_volatility",
    "temp_max_c", "temp_min_c", "rainfall_mm", "humidity_avg_pct",
    "evapotranspiration_mm", "temp_range_c"
]

# Forward-fill nulls, then drop remaining
df[FEATURES] = df[FEATURES].ffill()
df = df.dropna(subset=FEATURES).reset_index(drop=True)

print(f"  Shape after cleaning: {df.shape}")
print(f"  Date range: {df['arrival_date'].min().date()} → {df['arrival_date'].max().date()}")
print(f"  Features used: {FEATURES}\n")

# ============================================================================
# STEP 2 — Build sequences
# ============================================================================
print("=" * 70)
print("STEP 2 — Build sequences")
print("=" * 70)

LOOKBACK = 60
HORIZON  = 30
MIN_VALID = 15

feature_data = df[FEATURES].values.astype(np.float32)
price_data   = df["modal_price"].values.astype(np.float32)
dates        = df["arrival_date"].values

X_list, y_list, date_list = [], [], []

for i in range(len(df) - LOOKBACK - HORIZON + 1):
    x_window = feature_data[i : i + LOOKBACK]
    target_window = price_data[i + LOOKBACK : i + LOOKBACK + HORIZON]

    # Count valid (non-NaN) prices in target window
    valid_count = np.sum(~np.isnan(target_window))
    if valid_count < MIN_VALID:
        continue

    y_val = np.nanmean(target_window)
    X_list.append(x_window)
    y_list.append(y_val)
    date_list.append(dates[i + LOOKBACK - 1])  # date of last day in lookback

X_all = np.array(X_list, dtype=np.float32)
y_all = np.array(y_list, dtype=np.float32)
date_all = np.array(date_list)

print(f"  Total sequences created: {len(X_all)}")
print(f"  X shape: {X_all.shape}  |  y shape: {y_all.shape}\n")

# ============================================================================
# STEP 3 — Split and normalize
# ============================================================================
print("=" * 70)
print("STEP 3 — Split and normalize")
print("=" * 70)

train_mask = date_all < np.datetime64("2022-01-01")
val_mask   = (date_all >= np.datetime64("2022-01-01")) & (date_all < np.datetime64("2024-01-01"))
test_mask  = date_all >= np.datetime64("2024-01-01")

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

# Fit MinMaxScaler on train X only
n_features = X_train.shape[2]
scaler = MinMaxScaler()
# Reshape to 2D for fitting
X_train_2d = X_train.reshape(-1, n_features)
scaler.fit(X_train_2d)

# Transform all splits
X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
X_val   = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
X_test  = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

print(f"  Train sequences: {len(X_train)}  (dates < 2022-01-01)")
print(f"  Val   sequences: {len(X_val)}    (2022-01-01 to 2023-12-31)")
print(f"  Test  sequences: {len(X_test)}   (2024-01-01 onwards)")
print(f"  y NOT scaled — values in Rs/quintal")
print(f"  Scaler fitted on train X with {n_features} features\n")

# ============================================================================
# STEP 4 — Build and train LSTM
# ============================================================================
print("=" * 70)
print("STEP 4 — Build and train LSTM")
print("=" * 70)


class LSTMPredictor(nn.Module):
    """LSTM model for 30-day average price prediction."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]      # take last timestep
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.fc2(out)
        return out.squeeze(-1)


# Convert to tensors
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
model = LSTMPredictor(input_size=n_features).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

EPOCHS = 50
best_val_loss = float("inf")
train_losses, val_losses = [], []
model_path = os.path.join(BASE_DIR, "lstm_best_model.pth")

print(f"  Architecture: {model}")
print(f"  Optimizer: Adam (lr=0.001)  |  Loss: MSELoss")
print(f"  Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}")
print(f"  Device: {DEVICE}\n")

for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    epoch_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * xb.size(0)
    epoch_train_loss /= len(train_dataset)

    # ---- Validate ----
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            epoch_val_loss += loss.item() * xb.size(0)
    epoch_val_loss /= len(val_dataset)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

    # Save best model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), model_path)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  |  Train Loss: {epoch_train_loss:12.2f}  |  Val Loss: {epoch_val_loss:12.2f}")

print(f"\n  Best val loss: {best_val_loss:.2f}")
print(f"  Best model saved to: {model_path}")

# Loss curve plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", linewidth=2)
ax.plot(range(1, EPOCHS + 1), val_losses, label="Val Loss", linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("MSE Loss", fontsize=12)
ax.set_title("LSTM Training & Validation Loss", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
loss_plot_path = os.path.join(BASE_DIR, "lstm_training_loss.png")
fig.savefig(loss_plot_path, dpi=150)
plt.close(fig)
print(f"  Loss curve saved to: {loss_plot_path}\n")

# ============================================================================
# STEP 5 — Evaluate on test set
# ============================================================================
print("=" * 70)
print("STEP 5 — Evaluate on test set")
print("=" * 70)

# Load best model
model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    y_pred = model(X_test_tensor).cpu().numpy()

y_actual = y_test

mae  = np.mean(np.abs(y_actual - y_pred))
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))

print(f"\n  {'Metric':<12} {'Value':>12}")
print(f"  {'─' * 25}")
print(f"  {'MAE':<12} {mae:>12.2f} Rs/q")
print(f"  {'MAPE':<12} {mape:>11.2f}%")
print(f"  {'RMSE':<12} {rmse:>12.2f} Rs/q")

# Test evaluation plot
test_dates = date_all[test_mask]
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_dates, y_actual, label="Actual (30d avg)", linewidth=2, color="#2196F3")
ax.plot(test_dates, y_pred,   label="Predicted",        linewidth=2, color="#FF5722", linestyle="--")
ax.fill_between(test_dates, y_actual, y_pred, alpha=0.15, color="#FF5722")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Modal Price (Rs/quintal)", fontsize=12)
ax.set_title("LSTM Test Set: Actual vs Predicted 30-day Average Price", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
eval_plot_path = os.path.join(BASE_DIR, "lstm_test_eval.png")
fig.savefig(eval_plot_path, dpi=150)
plt.close(fig)
print(f"\n  Evaluation plot saved to: {eval_plot_path}\n")

# ============================================================================
# STEP 6 — 30-day forecast
# ============================================================================
print("=" * 70)
print("STEP 6 — 30-day forecast")
print("=" * 70)

# Use the last 60 days of data as input
last_60 = feature_data[-LOOKBACK:]
last_60_scaled = scaler.transform(last_60.reshape(-1, n_features)).reshape(1, LOOKBACK, n_features)
last_60_tensor = torch.tensor(last_60_scaled, dtype=torch.float32).to(DEVICE)

model.eval()
with torch.no_grad():
    forecast = model(last_60_tensor).cpu().item()

last_date = pd.Timestamp(dates[-1])
forecast_start = last_date + pd.Timedelta(days=1)
forecast_end   = last_date + pd.Timedelta(days=30)

print(f"\n  Input window: last 60 days ending {last_date.date()}")
print(f"  Forecast period: {forecast_start.date()} → {forecast_end.date()}")
print(f"  ╔{'═' * 48}╗")
print(f"  ║  Predicted 30-day avg price: {forecast:10.2f} Rs/q   ║")
print(f"  ╚{'═' * 48}╝")

# Save forecast summary
summary_path = os.path.join(BASE_DIR, "lstm_forecast_summary.txt")
with open(summary_path, "w") as f:
    f.write("LSTM 30-Day Price Forecast Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model: LSTM (2-layer, 64 hidden, dropout=0.2)\n")
    f.write(f"Lookback: {LOOKBACK} days\n")
    f.write(f"Target: Mean modal_price over next {HORIZON} days\n\n")
    f.write(f"Input window: last 60 days ending {last_date.date()}\n")
    f.write(f"Forecast period: {forecast_start.date()} to {forecast_end.date()}\n\n")
    f.write(f"Predicted 30-day average price: {forecast:.2f} Rs/quintal\n\n")
    f.write(f"Test Set Metrics:\n")
    f.write(f"  MAE:  {mae:.2f} Rs/quintal\n")
    f.write(f"  MAPE: {mape:.2f}%\n")
    f.write(f"  RMSE: {rmse:.2f} Rs/quintal\n")

print(f"\n  Forecast summary saved to: {summary_path}")
print("\n" + "=" * 70)
print("DONE — All steps completed successfully.")
print("=" * 70)
