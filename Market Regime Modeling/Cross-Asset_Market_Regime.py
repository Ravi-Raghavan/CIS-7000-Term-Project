import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os

warnings.filterwarnings("ignore")

# =================================================================================
# 1. AUTHORITATIVE MARKET REGIME DEFINITION & DATA PREPARATION
# =================================================================================

def get_macro_csv_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    csv_path = os.path.join(project_root, 'Finance Data', 'Historical_Data_MACRO.csv')
    if not os.path.exists(csv_path):
        csv_path = "Finance Data/Historical_Data_MACRO.csv"
    return csv_path

def load_and_prepare_authoritative_data():
    csv_path = get_macro_csv_path()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}. Please run download_macro_data.py first!")
        
    print(f"Loading local macro data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # Calculate daily returns & basic features for BOTH SPY and QQQ
    df['SPY_Return'] = df['SPY_Close'].pct_change()
    df['QQQ_Return'] = df['QQQ_Close'].pct_change()
    
    df['SPY_Log_Return'] = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
    df['QQQ_Log_Return'] = np.log(df['QQQ_Close'] / df['QQQ_Close'].shift(1))
    
    df['SPY_Intraday_Vol'] = (df['SPY_High'] - df['SPY_Low']) / df['SPY_Open']
    df['QQQ_Intraday_Vol'] = (df['QQQ_High'] - df['QQQ_Low']) / df['QQQ_Open']
    
    df['SPY_Volume_Ratio'] = df['SPY_Volume'] / df['SPY_Volume'].rolling(window=20).mean()
    df['QQQ_Volume_Ratio'] = df['QQQ_Volume'] / df['QQQ_Volume'].rolling(window=20).mean()
    
    # Calculate 200-Day SMA for BOTH
    df['SPY_200_SMA'] = df['SPY_Close'].rolling(window=200).mean()
    df['QQQ_200_SMA'] = df['QQQ_Close'].rolling(window=200).mean()
    
    # TARGET: Continuous Regime Label (Market Temperature) for BOTH
    df['SPY_Regime_Continuous'] = (df['SPY_Close'] - df['SPY_200_SMA']) / df['SPY_200_SMA']
    df['QQQ_Regime_Continuous'] = (df['QQQ_Close'] - df['QQQ_200_SMA']) / df['QQQ_200_SMA']
    
    df = df.dropna().reset_index(drop=True)
    return df

# ==========================================
# 2. VISUALIZATION (SAVING TO FOLDER)
# ==========================================

def get_plot_dir():
    base_dir = os.path.dirname(get_macro_csv_path())
    plot_dir = os.path.join(base_dir, 'Plots')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def visualize_regime_and_splits(df):
    """Visualizes the Train/Val/Test splits (SPY vs QQQ) and saves to file."""
    print("Generating visualization and saving to folder...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot Price & Splits (SPY for Train/Val, QQQ for Test)
    train_mask = df['Date'].dt.year < 2018
    val_mask = (df['Date'].dt.year >= 2018) & (df['Date'].dt.year < 2020)
    test_mask = df['Date'].dt.year >= 2020
    
    ax1.plot(df[train_mask]['Date'], df[train_mask]['SPY_Close'], label='Train SPY (Pre-2018)', color='blue')
    ax1.plot(df[val_mask]['Date'], df[val_mask]['SPY_Close'], label='Val SPY (2018-2019)', color='orange')
    ax1.plot(df[test_mask]['Date'], df[test_mask]['QQQ_Close'], label='Test QQQ (2020+)', color='red')
    
    ax1.axvline(pd.to_datetime('2020-01-01', utc=True), color='black', linestyle='--', alpha=0.7)
    ax1.set_title('Cross-Asset Split: SPY (Train/Val) vs QQQ (Test)')
    ax1.set_ylabel('Index Close Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Continuous Regime Label (Market Temperature)
    ax2.plot(df[~test_mask]['Date'], df[~test_mask]['SPY_Regime_Continuous'] * 100, color='purple', linewidth=1.5, label='SPY Regime')
    ax2.plot(df[test_mask]['Date'], df[test_mask]['QQQ_Regime_Continuous'] * 100, color='magenta', linewidth=1.5, label='QQQ Regime (Test)')
    ax2.axhline(0, color='black', linestyle='-')
    ax2.axhline(-20, color='red', linestyle='--', label='-20% Bear Market Threshold')
    
    ax2.set_title('Continuous Market Regime Target (% Distance from 200-Day SMA)')
    ax2.set_ylabel('Regime Score (%)')
    ax2.set_xlabel('Year')
    
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(get_plot_dir(), '1_Cross_Asset_Split.png')
    plt.savefig(plot_path)
    print(f"Saved split visualization to: {plot_path}")
    plt.close()

def visualize_predictions(dates, y_true, predictions_dict):
    """Visualizes Model Predictions vs Actual Continuous Regime on the Test Set (QQQ) and saves."""
    plt.figure(figsize=(14, 7))
    
    plt.plot(dates, y_true, label='Actual QQQ Regime (True Market Temp)', color='black', linewidth=2, zorder=10)
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for (name, preds), color in zip(predictions_dict.items(), colors):
        plt.plot(dates, preds, label=f'{name} Prediction', color=color, alpha=0.7, linestyle='--')
        
    plt.axhline(0, color='gray', linestyle='-')
    plt.title('Backbone Comparison: Predictions vs Actual QQQ Regime (Test Set: 2020+)')
    plt.xlabel('Year')
    plt.ylabel('Regime Score (Distance from 200 SMA)')
    
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(get_plot_dir(), '2_QQQ_Test_Predictions.png')
    plt.savefig(plot_path)
    print(f"Saved prediction comparison to: {plot_path}")
    plt.close()

# ==========================================
# 3. CONNECTION TO MULTI-TASK MODEL
# ==========================================
class MultiTaskTargetGenerator:
    """Utility class to merge all targets (Price, Sharpe, Regime) for the final model."""
    def __init__(self, df):
        self.df = df
        
    def get_multitask_targets(self):
        targets = self.df[['SPY_Close', 'Target_Sharpe', 'SPY_Regime_Continuous']].values
        return targets

# ==========================================
# 4. DATASETS & CROSS-ASSET SPLITTING
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) 
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(features, targets, date_array, sequence_length=30):
    X, y, dates = [], [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])
        dates.append(date_array[i+sequence_length])
    return np.array(X), np.array(y), pd.to_datetime(dates)

# ==========================================
# 5. CONTINUOUS BASELINE MODELS (REGRESSORS)
# ==========================================

class FFNNRegressor(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size * sequence_length, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x))).squeeze()

class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

class TCNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.relu(self.conv(x))
        out, _ = torch.max(out, dim=2)
        return self.fc(out).squeeze()

class TransformerRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        out = out.mean(dim=1)
        return self.fc(out).squeeze()

# ==========================================
# 6. TRAINING LOOP & EVALUATION
# ==========================================

def run_continuous_regime_pipeline():
    df = load_and_prepare_authoritative_data()
    visualize_regime_and_splits(df)
    
    # Define generic features the models will learn from
    spy_cols = ['SPY_Return', 'SPY_Log_Return', 'SPY_Volume_Ratio', 'SPY_Intraday_Vol']
    qqq_cols = ['QQQ_Return', 'QQQ_Log_Return', 'QQQ_Volume_Ratio', 'QQQ_Intraday_Vol']
    
    # Scale features based on SPY (training data distribution)
    # Appending .values strips the pandas column names to prevent StandardScaler feature name errors
    scaler = StandardScaler()
    spy_scaled = scaler.fit_transform(df[spy_cols].values)
    qqq_scaled = scaler.transform(df[qqq_cols].values) # Apply same scaler to QQQ to avoid look-ahead bias
    
    seq_length = 30
    batch_size = 64
    epochs = 10
    
    # Create SPY Sequences (For Train/Val)
    X_spy, y_spy, dates_spy = create_sequences(spy_scaled, df['SPY_Regime_Continuous'].values, df['Date'].values, seq_length)
    
    # Create QQQ Sequences (For Test)
    X_qqq, y_qqq, dates_qqq = create_sequences(qqq_scaled, df['QQQ_Regime_Continuous'].values, df['Date'].values, seq_length)
    
    # CROSS-ASSET SPLITTING LOGIC
    train_mask = dates_spy.year < 2018
    val_mask = (dates_spy.year >= 2018) & (dates_spy.year < 2020)
    test_mask = dates_qqq.year >= 2020
    
    X_train, y_train = X_spy[train_mask], y_spy[train_mask]
    X_val, y_val = X_spy[val_mask], y_spy[val_mask]
    
    X_test, y_test, dates_test = X_qqq[test_mask], y_qqq[test_mask], dates_qqq[test_mask]
    
    print(f"Data Split -> SPY Train: {len(X_train)} | SPY Val: {len(X_val)} | QQQ Test: {len(X_test)}")
    
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    
    # Initialize models with generic input size
    input_features = len(spy_cols)
    models = {
        "FFNN/LR": FFNNRegressor(input_size=input_features, sequence_length=seq_length),
        "RNN": RNNRegressor(input_size=input_features),
        "LSTM": LSTMRegressor(input_size=input_features),
        "TCN": TCNRegressor(input_size=input_features),
        "Transformer": TransformerRegressor(input_size=input_features)
    }
    
    criterion = nn.MSELoss() 
    results = []
    test_predictions = {}
    
    print("\n" + "="*75)
    print("--- Cross-Asset Training: Learn on SPY, Evaluate on QQQ ---")
    print("="*75)
    
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        # Evaluate on Validation (SPY) & Test Set (QQQ)
        model.eval()
        with torch.no_grad():
            X_val_tensor, y_val_tensor = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
            X_test_tensor, y_test_tensor = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
            
            val_loss = criterion(model(X_val_tensor), y_val_tensor).item()
            test_loss = criterion(model(X_test_tensor), y_test_tensor).item()
            test_predictions[name] = model(X_test_tensor).numpy()
            
        results.append((name, total_loss/len(train_loader), val_loss, test_loss))
    
    print(f"{'Backbone':<15} | {'SPY Val MSE (2018-19)':<21} | {'QQQ Test MSE (2020+)':<15}")
    print("-" * 75)
    for name, train, val, test in sorted(results, key=lambda x: x[2]):
        print(f"{name:<15} | {val:>15.4f}         | {test:>14.4f}")
    print("="*75)
    
    visualize_predictions(dates_test, y_test, test_predictions)

if __name__ == "__main__":
    run_continuous_regime_pipeline()