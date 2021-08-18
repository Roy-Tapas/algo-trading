import numpy as np
import pandas as pd

# Function to compute exponential moving average cross-over signal 
def ema_crossover_signal(df, short_window, long_window):
    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    ema_df = pd.DataFrame()
    ema_df['fast_close'] = df['close'].ewm(halflife=short_window).mean()
    ema_df['slow_close'] = df['close'].ewm(halflife=long_window).mean()

    # Construct a EMA crossover trading signal
    ema_df['crossover_long'] = np.where(ema_df['fast_close'] > ema_df['slow_close'], 1.0, 0.0)
    ema_df['crossover_short'] = np.where(ema_df['fast_close'] < ema_df['slow_close'], -1.0, 0.0)
    ema_df['crossover_signal'] = ema_df['crossover_long'] + ema_df['crossover_short']
    
    return ema_df.loc[:,['fast_close','slow_close','crossover_signal']]

# Function to compute Daily Return Volatility signal 
def vol_signal(df, short_window, long_window):
    vol_df = pd.DataFrame()
    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    vol_df['fast_vol'] = df['daily_return'].ewm(halflife=short_window).std()
    vol_df['slow_vol'] = df['daily_return'].ewm(halflife=long_window).std()

    # Construct a Volatility signal
    vol_df['vol_trend_long'] = np.where(vol_df['fast_vol'] < vol_df['slow_vol'], 1.0, 0.0)
    vol_df['vol_trend_short'] = np.where(vol_df['fast_vol'] > vol_df['slow_vol'], -1.0, 0.0) 
    vol_df['vol_trend_signal'] = vol_df['vol_trend_long'] + vol_df['vol_trend_short']

    return vol_df.loc[:,['fast_vol','slow_vol','vol_trend_signal']]


# Function to compute Bollinger Band
def bollinger_band_signal(df, bollinger_window):
    # Calculate rolling mean and standard deviation
    bol_df = pd.DataFrame()
    bol_df['bollinger_mid_band'] = df['close'].rolling(window=bollinger_window).mean()
    bol_df['bollinger_std'] = df['close'].rolling(window=bollinger_window).std()

    # Calculate upper and lowers bands of bollinger band
    bol_df['bollinger_upper_band']  = bol_df['bollinger_mid_band'] + (bol_df['bollinger_std'] * 1)
    bol_df['bollinger_lower_band']  = bol_df['bollinger_mid_band'] - (bol_df['bollinger_std'] * 1)

    # Calculate bollinger band trading signal
    bol_df['bollinger_long'] = np.where(df['close'] < bol_df['bollinger_lower_band'], 1.0, 0.0)
    bol_df['bollinger_short'] = np.where(df['close'] > bol_df['bollinger_upper_band'], -1.0, 0.0)
    bol_df['bollinger_signal'] = bol_df['bollinger_long'] + bol_df['bollinger_short']

    return bol_df[['bollinger_mid_band','bollinger_upper_band','bollinger_lower_band','bollinger_signal']]


# Funtion to calculate rate of change (ROC)
def roc(df, n):  
    M = df.diff(n - 1)  
    N = df.shift(n - 1)  
    ROC = pd.Series(((M / N) * 100))   
    return ROC

# Function to derive rate-of-change signal
def rate_of_change_signal(df, short_window, long_window):
    roc_df = pd.DataFrame()
    # Construct a `Fast` and `Slow` Rate-Of-Change from short and long windows, respectively
    roc_df['fast_roc'] = roc(df['close'], short_window) 
    roc_df['slow_roc'] = roc(df['close'], long_window)

    # Construct a Rate_of_Change signal
    roc_df['roc_trend_long'] = np.where(roc_df['fast_roc'] < roc_df['slow_roc'], 1.0, 0.0)
    roc_df['roc_trend_short'] = np.where(roc_df['fast_roc'] > roc_df['slow_roc'], -1.0, 0.0) 
    roc_df['roc_trend_signal'] = roc_df['roc_trend_long'] + roc_df['roc_trend_short']

    return roc_df.loc[:,['fast_roc','slow_roc','roc_trend_signal']]


#function to cacculate price momentum (MOM)
def mom(df, n):   
    MOM = pd.Series(df.diff(n))   
    return MOM

# Function to derive momentum signal
def momentum_signal(df, short_window, long_window):
    mom_df = pd.DataFrame()
    # Construct a `Fast` and `Slow` Rate-Of-Change from short and long windows, respectively
    mom_df['fast_mom'] = mom(df['close'], short_window) 
    mom_df['slow_mom'] = mom(df['close'], long_window)

    # Construct a Momentum signal
    mom_df['mom_trend_long'] = np.where(mom_df['fast_mom'] < mom_df['slow_mom'], 1.0, 0.0)
    mom_df['mom_trend_short'] = np.where(mom_df['fast_mom'] > mom_df['slow_mom'], -1.0, 0.0) 
    mom_df['mom_trend_signal'] = mom_df['mom_trend_long'] + mom_df['mom_trend_short']

    return mom_df.loc[:,['fast_mom','slow_mom','mom_trend_signal']]


#Function to caclulate relative strength index (RSI)
def rsi(series, period):
 delta = series.diff().dropna()
 u = delta * 0
 d = u.copy()
 u[delta > 0] = delta[delta > 0]
 d[delta < 0] = -delta[delta < 0]
 u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
 u = u.drop(u.index[:(period-1)])
 d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
 d = d.drop(d.index[:(period-1)])
 rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
 return 100 - 100 / (1 + rs)

# Function to derive rsi signal
def rsi_signal(df, short_window, long_window):
    rsi_df = pd.DataFrame()
    # Construct a `Fast` and `Slow` Rate-Of-Change from short and long windows, respectively
    rsi_df['fast_rsi'] = rsi(df['close'], short_window) 
    rsi_df['slow_rsi'] = rsi(df['close'], long_window)

    # Construct a RSI signal
    rsi_df['rsi_trend_long'] = np.where(rsi_df['fast_rsi'] < rsi_df['slow_rsi'], 1.0, 0.0)
    rsi_df['rsi_trend_short'] = np.where(rsi_df['fast_rsi'] > rsi_df['slow_rsi'], -1.0, 0.0) 
    rsi_df['rsi_trend_signal'] = rsi_df['rsi_trend_long'] + rsi_df['rsi_trend_short']

    return rsi_df.loc[:,['fast_rsi','slow_rsi','rsi_trend_signal']]


#Function to calculate stochastic osillator - STOK: slow indicator.
def stok(close, low, high, window): 
 stok = ((close - low.rolling(window).min()) / (high.rolling(window).max() - low.rolling(window).min())) * 100
 return stok

# Function to derive stok signal
def stok_signal(df, short_window, long_window):
    stok_df = pd.DataFrame()
    # Construct a `Fast` and `Slow` STOK from short and long windows, respectively
    stok_df['fast_stok'] = stok(df['close'], df['low'], df['high'], short_window) 
    stok_df['slow_stok'] = stok(df['close'], df['low'], df['high'], long_window)

    # Construct a STOK signal
    stok_df['stok_trend_long'] = np.where(stok_df['fast_stok'] < stok_df['slow_stok'], 1.0, 0.0)
    stok_df['stok_trend_short'] = np.where(stok_df['fast_stok'] > stok_df['slow_stok'], -1.0, 0.0) 
    stok_df['stok_trend_signal'] = stok_df['stok_trend_long'] + stok_df['stok_trend_short']

    return stok_df.loc[:,['fast_stok','slow_stok','stok_trend_signal']]
 

#Function to calculate stochastic osillators - STOD: fast indicator.
def stod(close, low, high, window):
    stok = ((close - low.rolling(window).min()) / (high.rolling(window).max() - low.rolling(window).min())) * 100
    stod = stok.rolling(window).mean()
    return stod

# Function to derive stod signal
def stod_signal(df, short_window, long_window):
    stod_df = pd.DataFrame()
    # Construct a `Fast` and `Slow` STOD from short and long windows, respectively
    stod_df['fast_stod'] = stod(df['close'], df['low'], df['high'], short_window) 
    stod_df['slow_stod'] = stod(df['close'], df['low'], df['high'], long_window)

    # Construct a STOD signal
    stod_df['stod_trend_long'] = np.where(stod_df['fast_stod'] < stod_df['slow_stod'], 1.0, 0.0)
    stod_df['stod_trend_short'] = np.where(stod_df['fast_stod'] > stod_df['slow_stod'], -1.0, 0.0) 
    stod_df['stod_trend_signal'] = stod_df['stod_trend_long'] + stod_df['stod_trend_short']

    return stod_df.loc[:,['fast_stod','slow_stod','stod_trend_signal']]