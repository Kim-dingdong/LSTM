import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib import font_manager
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Binance에서 3년치 비트코인 가격 수집
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

SEQ_LEN = 60
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LEN)
X = X.reshape((-1, SEQ_LEN, 1))

# 3. 모델 불러오기 및 예측
model = load_model('bitcoin_lstm_model.h5')
predicted = model.predict(X)
predicted_price = scaler.inverse_transform(predicted)
actual_price = scaler.inverse_transform(y.reshape(-1, 1))

# 4. 평가
mae = mean_absolute_error(actual_price, predicted_price)
rmse = np.sqrt(mean_squared_error(actual_price, predicted_price))

# 5. 시각화 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 한글 폰트
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["font.family"] = font_name

plt.figure(figsize=(14,6))

# 전체 실제 데이터 (파란색)
plt.plot(df.index[-len(actual_price):], actual_price, label='실제 비트코인 가격 (USD)', color='royalblue')

# 예측선: 최근 3개월 (90일)만 점선 주황색으로 강조
predict_range = 90  # 90일
plt.plot(df.index[-predict_range:], predicted_price[-predict_range:], label='예측 가격 (LSTM)', linestyle='--', color='darkorange')

# 축 포맷
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${int(x):,}'))

# 성능 텍스트 박스
textstr = f"MAE: ${mae:,.2f}\nRMSE: ${rmse:,.2f}"
plt.text(df.index[-1], predicted_price[-1], textstr,
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top', horizontalalignment='left')

plt.xlabel('날짜')
plt.ylabel('가격 (USD)')
plt.title('저장된 모델 기반 비트코인 가격 예측 결과')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
