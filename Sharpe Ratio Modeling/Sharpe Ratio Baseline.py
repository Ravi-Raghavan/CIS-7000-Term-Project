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

# ==========================================
# 1. DATA PREPARATION
# ==========================================

def get_default_csv_path():
    """Helper to resolve the default AAPL path based on your VS Code workspace."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    csv_path = os.path.join(project_root, 'Finance Data', 'Historical_Data_AAPL.csv')
    
    if not os.path.exists(csv_path):
        csv_path = "Finance Data/Historical_Data_AAPL.csv"
    return csv_path

def prepare_ticker_data(csv_path, sharpe_window=60):
    print(f"Loading ticker data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Restrict to 2010 onwards for relevant history
    df = df[df['Date'].dt.year >= 2010].reset_index(drop=True)
    
    df['Return'] = df['Close'].pct_change()
    
    # Target: Rolling Sharpe Ratio
    rolling_mean = df['Return'].rolling(window=sharpe_window).mean()
    rolling_std = df['Return'].rolling(window=sharpe_window).std()
    df['Target_Sharpe'] = np.sqrt(252) * (rolling_mean / rolling_std)
    
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Rolling_Vol'] = df['Return'].rolling(window=20).std()
    
    df = df.dropna().reset_index(drop=True)
    return df

# ==========================================
# 2. SEQUENCE CREATION & CHRONOLOGICAL SPLIT
# ==========================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences_and_split(df, feature_cols, target_col, sequence_length=30):
    """Creates sequences and splits chronologically per the project proposal."""
    X, y, dates = [], [], []
    features = df[feature_cols].values
    targets = df[target_col].values
    date_array = df['Date'].values
    
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])
        dates.append(date_array[i+sequence_length])
        
    X, y, dates = np.array(X), np.array(y), pd.to_datetime(dates)
    
    # CHRONOLOGICAL SPLIT (Matching Proposal Section 4)
    train_mask = dates.year < 2018
    val_mask = (dates.year >= 2018) & (dates.year < 2020)
    test_mask = dates.year >= 2020
    
    return (X[train_mask], y[train_mask], dates[train_mask]), \
           (X[val_mask], y[val_mask], dates[val_mask]), \
           (X[test_mask], y[test_mask], dates[test_mask])

def visualize_split(dates_train, dates_val, dates_test, df):
    """Visualizes the Train/Val/Test split and highlights domain shift."""
    plt.figure(figsize=(12, 6))
    
    plt.plot(df[df['Date'].isin(dates_train)]['Date'], df[df['Date'].isin(dates_train)]['Close'], label='Train (2010-2017)', color='blue')
    plt.plot(df[df['Date'].isin(dates_val)]['Date'], df[df['Date'].isin(dates_val)]['Close'], label='Validation (2018-2019)', color='orange')
    plt.plot(df[df['Date'].isin(dates_test)]['Date'], df[df['Date'].isin(dates_test)]['Close'], label='Test / Shifted Regime (2020+)', color='red')
    
    plt.axvline(pd.to_datetime('2020-01-01', utc=True), color='black', linestyle='--', alpha=0.7)
    plt.text(pd.to_datetime('2020-03-01', utc=True), df['Close'].max()*0.8, 'DOMAIN SHIFT\n(COVID, ZIRP, Tech Boom)', color='black', fontweight='bold')
    
    plt.title('Chronological Data Split for Sharpe Ratio Prediction')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# ==========================================
# 3. BASELINE MODELS (BACKBONE COMPARISON)
# ==========================================

class FFNNRegressor(nn.Module):
    """Feed-Forward Neural Network Backbone"""
    def __init__(self, input_size, sequence_length, hidden_size=64):
        super(FFNNRegressor, self).__init__()
        # Flatten the sequence: seq_len * num_features
        self.fc1 = nn.Linear(input_size * sequence_length, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten
        out = self.dropout(self.relu(self.fc1(x)))
        out = self.fc2(out)
        return out.squeeze()

class RNNRegressor(nn.Module):
    """Standard Recurrent Neural Network Backbone"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :]) 
        return out.squeeze()

class LSTMRegressor(nn.Module):
    """Long Short-Term Memory Backbone"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out.squeeze()

# ==========================================
# 4. TRAINING & EVALUATION LOOP
# ==========================================

def evaluate_model(model, X_val, y_val, criterion):
    """Evaluates the model on the validation set."""
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        predictions = model(X_val_tensor)
        loss = criterion(predictions, y_val_tensor)
    return loss.item()

def run_sharpe_pipeline():
    csv_path = get_default_csv_path()
    df = prepare_ticker_data(csv_path)
    
    features = ['Return', 'Log_Return', 'Rolling_Vol']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    seq_length = 30
    batch_size = 64
    epochs = 10
    
    # Split data chronologically
    train_data, val_data, test_data = create_sequences_and_split(df, features, 'Target_Sharpe', seq_length)
    
    X_train, y_train, dates_train = train_data
    X_val, y_val, dates_val = val_data
    X_test, y_test, dates_test = test_data
    
    print(f"Data Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    visualize_split(dates_train, dates_val, dates_test, df)
    
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    
    # Initialize the different backbones to compare
    models = {
        "FFNN": FFNNRegressor(input_size=len(features), sequence_length=seq_length),
        "RNN": RNNRegressor(input_size=len(features)),
        "LSTM": LSTMRegressor(input_size=len(features))
    }
    
    results = {}
    criterion = nn.MSELoss()
    
    print("\n" + "="*50)
    print("--- Starting Backbone Comparison ---")
    print("="*50)
    
    for name, model in models.items():
        print(f"\nTraining {name} Backbone...")
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
                
        # Evaluate on Validation Set (2018-2019)
        val_mse = evaluate_model(model, X_val, y_val, criterion)
        results[name] = val_mse
        print(f"-> {name} Final Train Loss: {total_loss/len(train_loader):.4f} | Validation MSE: {val_mse:.4f}")

    print("\n" + "="*50)
    print("--- Backbone Comparison Results (Lower is Better) ---")
    print("="*50)
    for name, mse in sorted(results.items(), key=lambda item: item[1]):
        print(f"{name: <10}: {mse:.4f}")

if __name__ == "__main__":
    run_sharpe_pipeline()