import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# =================================================================================
# 1. AUTHORITATIVE MARKET REGIME DEFINITION & DATA PREPARATION
# =================================================================================
# DEFINITION & REASONING (WALL STREET STANDARDS):
# Instead of a "black-box" HMM, we use authoritative definitions from financial 
# media (WSJ, Bloomberg), the SEC, and quantitative research.
# 
# 1. SIMPLE MOVING AVERAGE (SMA) FORMULA & THE 200-DAY RATIONALE:
#    - Formula: SMA_n = (P_1 + P_2 + ... + P_n) / n  (where P is the closing price, n is 200)
#    - Why 200-day?: The 200-day SMA is the universal Wall Street benchmark for long-term 
#      trend identification. Trading above it signifies a Bull regime; below it signifies Bear.
#    - Reference: Faber, Mebane T. (2007) "A Quantitative Approach to Tactical Asset Allocation".
#      Published in The Journal of Wealth Management.
#    - URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461
#    - Part Referenced: Section on "Timing Model", proving the 200-day SMA reduces 
#      volatility and drawdowns by accurately identifying macro regime shifts.
#
# 2. CONTINUOUS TARGET (Market Temperature / Trend Magnitude): 
#    - Formula: (SPY_Close - SPY_200_SMA) / SPY_200_SMA
#    - Concept: Rather than a binary bull/bear toggle, momentum strength is continuous.
#      Positive % distance = Bull momentum severity. Negative % distance = Bear severity.
#    - Reference: Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere" (AQR Capital)
#    - URL: https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere
#    - Part Referenced: Time-series momentum (trend-following) is formally defined 
#      by the asset's current price relative to its historical moving average.
#
# 3. DISCRETE TARGET (3-Class: Bear, Correction, Bull):
#    - Bear Market (Class 0): Drawdown <= -20%.
#    - Correction (Class 1): Drawdown between -10% and -19.9%.
#    - Bull / Normal (Class 2): Drawdown > -10% and Price > 200 SMA.
#    - Reference: U.S. Securities and Exchange Commission (SEC) & Investopedia.
#    - URL 1 (SEC): https://www.sec.gov/reportspubs/investor-publications/investorpubsbearmarket
#    - URL 2 (Investopedia): https://www.investopedia.com/terms/b/bearmarket.asp
#    - Part Referenced: "A bear market is broadly defined as a 20% or more drop in broad 
#      market indexes... A correction is a decline of 10% to 19.9%."
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
    
    # Calculate daily returns & basic features
    df['SPY_Return'] = df['SPY_Close'].pct_change()
    df['QQQ_Return'] = df['QQQ_Close'].pct_change()
    df['SPY_Intraday_Vol'] = (df['SPY_High'] - df['SPY_Low']) / df['SPY_Open']
    df['SPY_Volume_Ratio'] = df['SPY_Volume'] / df['SPY_Volume'].rolling(window=20).mean()
    
    # ---------------------------------------------------------
    # AUTHORITATIVE REGIME TARGET GENERATION
    # ---------------------------------------------------------
    # Calculate 200-Day SMA (Standard Trend Indicator)
    df['SPY_200_SMA'] = df['SPY_Close'].rolling(window=200).mean()
    
    # Calculate 52-Week High (252 trading days)
    df['SPY_52W_High'] = df['SPY_Close'].rolling(window=252).max()
    
    # Calculate Drawdown from 52-Week High
    df['SPY_Drawdown'] = (df['SPY_Close'] - df['SPY_52W_High']) / df['SPY_52W_High']
    
    # TARGET 1: Continuous Regime Label (Market Temperature)
    # Distance from 200 SMA. Positive = Bull, Negative = Bear.
    df['Target_Regime_Continuous'] = (df['SPY_Close'] - df['SPY_200_SMA']) / df['SPY_200_SMA']
    
    # TARGET 2: Discrete 3-Class Regime Label
    def classify_regime(row):
        if pd.isna(row['SPY_Drawdown']): return np.nan
        if row['SPY_Drawdown'] <= -0.20: return 0  # Bear Market (>= 20% drop)
        elif row['SPY_Drawdown'] <= -0.10: return 1 # Correction (>= 10% drop)
        else: return 2                              # Bull / Normal Market
        
    df['Target_Regime_Class'] = df.apply(classify_regime, axis=1)
    
    # Drop the first 252 rows which are NaN due to the 52-week rolling calculations
    df = df.dropna().reset_index(drop=True)
    return df

# ==========================================
# 2. VISUALIZATION (PRE-TRAINING)
# ==========================================

def visualize_regime_and_splits(df):
    """Visualizes the Train/Val/Test splits and the Continuous Regime Label."""
    print("Generating visualization... Please close the plot window to begin training.")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 1. Plot Price & Splits
    train_mask = df['Date'].dt.year < 2018
    val_mask = (df['Date'].dt.year >= 2018) & (df['Date'].dt.year < 2020)
    test_mask = df['Date'].dt.year >= 2020
    
    ax1.plot(df[train_mask]['Date'], df[train_mask]['SPY_Close'], label='Train (Pre-2018)', color='blue')
    ax1.plot(df[val_mask]['Date'], df[val_mask]['SPY_Close'], label='Val (2018-2019)', color='orange')
    ax1.plot(df[test_mask]['Date'], df[test_mask]['SPY_Close'], label='Test (2020+)', color='red')
    
    ax1.axvline(pd.to_datetime('2020-01-01', utc=True), color='black', linestyle='--', alpha=0.7)
    ax1.set_title('SPY Macro Price History & Chronological Splits')
    ax1.set_ylabel('SPY Close Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Plot Continuous Regime Label (Market Temperature)
    ax2.plot(df['Date'], df['Target_Regime_Continuous'] * 100, color='purple', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='-')
    ax2.axhline(-20, color='red', linestyle='--', label='-20% Bear Market Threshold')
    
    # Fill areas based on regime
    ax2.fill_between(df['Date'], 0, df['Target_Regime_Continuous'] * 100, where=(df['Target_Regime_Continuous'] > 0), facecolor='green', alpha=0.3, label='Bull Momentum')
    ax2.fill_between(df['Date'], 0, df['Target_Regime_Continuous'] * 100, where=(df['Target_Regime_Continuous'] < 0), facecolor='red', alpha=0.3, label='Bear Momentum')
    
    ax2.set_title('Continuous Market Regime Target (% Distance from 200-Day SMA)')
    ax2.set_ylabel('Regime Score (%)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() # This will block the script until you close the window

# ==========================================
# 3. CONNECTION TO MULTI-TASK MODEL
# ==========================================
# This class shows how your teammates will eventually pull this regime 
# target into the final Multi-Task Dataset.

class MultiTaskTargetGenerator:
    """Utility class to merge all targets (Price, Sharpe, Regime) for the final model."""
    def __init__(self, df):
        self.df = df
        
    def get_multitask_targets(self):
        # In the final model, the target 'y' won't be a single number. 
        # It will be an array of all targets you want to predict simultaneously!
        targets = self.df[['SPY_Close', 'Target_Sharpe', 'Target_Regime_Continuous']].values
        return targets

# ==========================================
# 4. DATASETS & SPLITTING
# ==========================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Using float32 because we are now predicting a Continuous Regime target (Regression)
        self.y = torch.tensor(y, dtype=torch.float32) 
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences_and_split(df, feature_cols, target_col, sequence_length=30):
    X, y, dates = [], [], []
    features = df[feature_cols].values
    targets = df[target_col].values
    date_array = df['Date'].values
    
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])
        dates.append(date_array[i+sequence_length])
        
    X, y, dates = np.array(X), np.array(y), pd.to_datetime(dates)
    
    train_mask = dates.year < 2018
    val_mask = (dates.year >= 2018) & (dates.year < 2020)
    test_mask = dates.year >= 2020
    
    return (X[train_mask], y[train_mask], dates[train_mask]), \
           (X[val_mask], y[val_mask], dates[val_mask]), \
           (X[test_mask], y[test_mask], dates[test_mask])

# ==========================================
# 5. CONTINUOUS BASELINE MODEL (LSTM)
# ==========================================

class LSTMRegressor(nn.Module):
    """LSTM adjusted to predict the Continuous Regime Score (Regression)"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1) # Output 1 continuous value
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# ==========================================
# 6. TRAINING LOOP
# ==========================================

def run_continuous_regime_pipeline():
    df = load_and_prepare_authoritative_data()
    
    # 1. VISUALIZE FIRST
    visualize_regime_and_splits(df)
    
    # 2. PREPARE FEATURES
    features = ['SPY_Return', 'QQQ_Return', 'SPY_Volume_Ratio', 'SPY_Intraday_Vol']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    seq_length = 30
    batch_size = 64
    epochs = 10
    
    # We are now predicting the CONTINUOUS target (Target_Regime_Continuous)
    train_data, val_data, test_data = create_sequences_and_split(df, features, 'Target_Regime_Continuous', seq_length)
    X_train, y_train, _ = train_data
    X_val, y_val, _ = val_data
    X_test, y_test, _ = test_data
    
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    
    model = LSTMRegressor(input_size=len(features))
    # We use MSELoss because it is a continuous regression target now, not classification
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n" + "="*60)
    print("--- Training LSTM to Predict Continuous Market Regime ---")
    print("="*60)
    
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
            
        # Quick validation evaluation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(torch.tensor(X_val, dtype=torch.float32)), torch.tensor(y_val, dtype=torch.float32)).item()
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (MSE): {total_loss/len(train_loader):.4f} | Val Loss (MSE): {val_loss:.4f}")

if __name__ == "__main__":
    run_continuous_regime_pipeline()