import numpy as np
import yfinance as yf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt


def download_btc(start="2000-01-01"):
    btc = yf.download("BTC-USD", start=start)
    close = btc["Close"]
    high = btc["High"]
    low = btc["Low"]
    volume = btc["Volume"]
    return close, high, low, volume


def to_log(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


def compute_down_streak(window: pd.Series) -> int:
    cnt = 0
    max_cnt = 0
    for r in window.values:
        if r < 0:
            cnt += 1
        else:
            max_cnt = max(max_cnt, cnt)
            cnt = 0
    return max(max_cnt, cnt)


def compute_up_streak(window: pd.Series) -> int:
    cnt = 0
    max_cnt = 0
    for r in window.values:
        if r > 0:
            cnt += 1
        else:
            max_cnt = max(max_cnt, cnt)
            cnt = 0
    return max(max_cnt, cnt)


def max_drawdown(ret_window: pd.Series) -> float:
    cum = ret_window.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    return dd.min()


def compute_ATR(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window).mean()
    return atr


def build_feature_dataset(close, high, low, volume, N, K, atr_window):
    rets = to_log(close)

    close = close.reindex(rets.index)
    high = high.reindex(rets.index)
    low = low.reindex(rets.index)
    volume = volume.reindex(rets.index)

    atr = compute_ATR(high, low, close, window=atr_window).reindex(rets.index)
    ma7 = close.rolling(7).mean().reindex(rets.index)

    feature_list = []
    label_list = []
    index_list = []

    for i in range(N, len(rets) - K):
        past_window = rets.iloc[i-N:i]
        future_window = rets.iloc[i:i+K]
        volume_window = volume.iloc[i-N:i]

        feat = {}
        feat["ret_3d"] = past_window[-3:].sum()
        feat["ret_7d"] = past_window[-7:].sum()
        feat["ret_14d"] = past_window[-14:].sum()
        feat["ret_21d"] = past_window[-21:].sum()
        feat["ret_N"] = past_window.sum()

        feat["down_ratio"] = (past_window < 0).mean()
        feat["up_ratio"] = (past_window > 0).mean()

        feat["down_streak"] = compute_down_streak(past_window)
        feat["up_streak"] = compute_up_streak(past_window)

        feat["vol_7"] = past_window[-7:].std()
        feat["vol_30"] = past_window[-30:].std()
        feat["vol_of_vol"] = past_window.rolling(7).std().std()

        feat["atr"] = atr.iloc[i-1]

        last_volume = volume_window.iloc[-1]
        vol_7ago = volume_window.iloc[-7]
        feat["vol_change"] = (last_volume - volume_window.mean()) / volume_window.mean()
        feat["vol_mom"] = np.log(last_volume / vol_7ago)

        feat["price_over_ma7"] = close.iloc[i-1] / ma7.iloc[i-1]
        feat["ma7_slope"] = ma7.iloc[i-1] - ma7.iloc[i-2]

        feat["maxdd_N"] = max_drawdown(past_window)
        feat["skew"] = past_window.skew()
        feat["kurt"] = past_window.kurtosis()

        future_ret = float(future_window.sum())
        if future_ret >= 0.03:
            y = 2
        elif future_ret <= -0.02:
            y = 0
        else:
            y = 1

        feature_list.append(feat)
        label_list.append(y)
        index_list.append(rets.index[i-1])

    X = pd.DataFrame(feature_list, index=index_list)
    y = np.array(label_list)

    return X, y


if __name__ == "__main__":
    close, high, low, volume = download_btc("2000-01-01")

    N, K, ATR_WINDOW = 30, 3, 14
    X, y = build_feature_dataset(close, high, low, volume, N, K, ATR_WINDOW)

    print("X samples:", X.shape[0], " / y samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("Train Acc:", clf.score(X_train, y_train))
    print("Test  Acc:", clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.head(10))

    plt.figure(figsize=(7,4))
    importances.head(10)[::-1].plot(kind="barh")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.show()

    print("Proba last 5:")
    print(clf.predict_proba(X_test.tail(5)))
    print("True last 5:", y_test[-5:])
