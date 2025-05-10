import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib import font_manager

# 1. Binance에서 3년치 비트코인 데이터 수집
binance = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1d'
since = binance.parse8601((datetime.utcnow() - timedelta(days=3*365)).strftime('%Y-%m-%dT%H:%M:%S'))

ohlcv = []
while True:
    batch = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=200)
    if not batch:
        break
    ohlcv += batch
    since = batch[-1][0] + 1
    time.sleep(0.3)
    if len(ohlcv) >= 2000:
        break

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('datetime', inplace=True)

# 2. 전처리
data = df[['close']].copy()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

SEQ_LEN = 60
X, y = create_sequences(scaled_data, SEQ_LEN)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = X_train.reshape((-1, SEQ_LEN, 1))
X_test = X_test.reshape((-1, SEQ_LEN, 1))

# 3. LSTM 모델 학습
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32)

# 4. 예측 및 평가
predicted = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted)
actual_price = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(actual_price, predicted_price)
rmse = np.sqrt(mean_squared_error(actual_price, predicted_price))
print(f"\n📊 예측 성능 평가")
print(f"MAE (평균 절대 오차): {mae:.2f}")
print(f"RMSE (평균 제곱근 오차): {rmse:.2f}")

# 5. 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["font.family"] = font_name

# 6. 시각화 (전체 + 실제 + 예측 + 성능 지표 표시)
plt.figure(figsize=(12,6))
plt.plot(df.index, df['close'], label='전체 비트코인 가격', alpha=0.3)
plt.plot(df.index[-len(actual_price):], actual_price, label='실제 비트코인 가격 (USD)', color='blue')
plt.plot(df.index[-len(predicted_price):], predicted_price, label='예측 가격 (LSTM)', linestyle='--', color='orange')

# 성능 지표 텍스트 추가 (우측 상단)
textstr = f"MAE: ${mae:,.2f}\nRMSE: ${rmse:,.2f}"
plt.gca().annotate(textstr, xy=(0.98, 0.95), xycoords='axes fraction',
                   fontsize=10, ha='right', va='top',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.7),
                   fontproperties=font_manager.FontProperties(fname=font_path))

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${int(x):,}'))

plt.xlabel('날짜')
plt.ylabel('가격 (USD)')
plt.title('저장된 모델 기반 비트코인 가격 예측 결과', fontsize=14)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ... 앞부분 동일 (데이터 수집, 전처리, 학습 등)

# 7. 미래 3개월 예측을 기존 그래프에 이어붙이기
future_days = 90
last_sequence = scaled_data[-60:].reshape((1, 60, 1))
future_predictions = []

for _ in range(future_days):
    next_pred = model.predict(last_sequence)[0]
    future_predictions.append(next_pred)
    last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)

future_predictions = scaler.inverse_transform(future_predictions)
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

# 기존 가격 + 예측 가격 이어붙이기
full_dates = list(df.index) + future_dates
full_prices = list(df['close']) + list(future_predictions.flatten())

# 전체 그래프 출력 (과거 + 미래 예측)
plt.figure(figsize=(14,6))
plt.plot(df.index, df['close'], label='과거 비트코인 가격 (USD)', alpha=0.5)
plt.plot(future_dates, future_predictions, label='미래 3개월 예측 가격 (USD)', linestyle='--', color='red')

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${int(x):,}'))

plt.xlabel('날짜')
plt.ylabel('가격 (USD)')
plt.title('LSTM 기반 비트코인 가격 예측: 과거 3년 + 미래 3개월', fontsize=14)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

