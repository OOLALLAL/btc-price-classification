# Price Movement Classification

This project builds a feature-rich dataset from historical BTC-USD data and trains
a Random Forest classifier to predict short-term price movement classes:
- **0**: Large downside move  
- **1**: Neutral / sideways  
- **2**: Large upside move  

The model uses a wide range of features derived from returns, volatility, volume,
ATR, moving averages, streak patterns, and price/volume dynamics.

## Features
The dataset includes:
- Log-return statistics over multiple horizons  
- Up/Down streak length  
- Volatility indicators (7D, 30D, vol-of-vol)  
- ATR(14)  
- Volume momentum & volume deviation  
- Price vs MA(7) and MA slope  
- Max drawdown, skewness, kurtosis  

## Model
A `RandomForestClassifier` (300 trees, balanced class weights) is trained on a
chronologically split dataset to avoid leakage.

The script outputs:
- Train/Test accuracy  
- Classification report  
- Confusion matrix  
- Feature importance ranking  
- Probabilities for the last few samples  

## Usage
```
python main.py
```

The script automatically downloads BTC-USD from Yahoo Finance and builds the
dataset.

## Requirements
```
numpy
pandas
yfinance
scikit-learn
```

## License
MIT License
