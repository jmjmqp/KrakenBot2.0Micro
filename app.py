from flask import Flask, render_template, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import os
import ccxt
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings

def fractional_diff(series, d, window=20):
    """Calcula la diferenciación fraccional para mantener memoria de tendencia."""
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights[::-1])
    diff_series = series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)
    return diff_series.fillna(0)

import datetime
import traceback
import sys

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==========================================
# ESTADO GLOBAL DEL BOT Y UI
# ==========================================
class BotState:
    def __init__(self):
        self.is_running = False
        self.api_key = ''
        self.api_secret = ''
        self.max_margin_usd = 200.0
        self.symbols = ['BTC/USD:USD']
        self.base_currency = 'USD'
        # MICRO-TREND: timeframe corto por defecto
        self.timeframe = '5m'
        self.sl_pct = 0.8          # SL ajustado para micro-tendencias
        self.tp_pct = 1.5          # TP ajustado para micro-tendencias
        self.ts_activation_pct = 0.5
        self.ts_distance_pct = 0.3
        # MICRO-TREND: parámetros adicionales
        self.max_trade_duration_minutes = 60   # Cierre forzoso tras N minutos
        self.min_signal_confidence = 0.62      # Umbral de confianza mínima

        self.logs = []
        self.balance = 0.0
        self.total_balance = 0.0

        self.open_positions = {}
        self.last_updates = {}
        self.current_probs_dict = {}
        self.indicators = {}
        self.trade_histories = {}

        self.engine = None

        self.last_candle_timestamp = {}
        self.current_hmm_signal = {}
        self.current_hmm_confidence = {}
        self.chart_data = {}
        self.signal_history = {}
        self.best_seeds = {}
        self.last_calibration = {}
        self.sl_cooldown = {}
        # MICRO-TREND: tracking de tiempo de entrada para timeout
        self.trade_open_time = {}

        self.STATE_LABELS = {
            'LONG': ['Alcista'],
            'SHORT': ['Bajista']
        }

bot_state = BotState()

_log_lock = threading.Lock()

def log_msg(msg):
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    with _log_lock:
        bot_state.logs.insert(0, formatted)
        if len(bot_state.logs) > 100:
            bot_state.logs.pop()
    print(formatted)

# ==========================================
# UTILIDADES MICRO-TENDENCIA
# ==========================================
def get_timeframe_minutes(tf):
    """Convierte un timeframe de ccxt a minutos."""
    mapping = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15,
        '30m': 30, '1h': 60, '2h': 120, '4h': 240
    }
    return mapping.get(tf, 15)

def candles_needed_for_window(tf, train_days=3, val_days=1):
    """Calcula cuántas velas necesitamos para el HMM según el timeframe."""
    mins = get_timeframe_minutes(tf)
    total_mins = (train_days + val_days) * 24 * 60
    return max(200, int(total_mins / mins))

# ==========================================
# MOTOR HMM TRADING — MICRO-TENDENCIAS
# ==========================================
class HMMTradingEngine:
    def __init__(self, api_key, api_secret, symbols, timeframe, base_currency,
                 sl_pct, tp_pct, ts_act_pct, ts_dist_pct,
                 max_trade_duration_minutes=60, min_signal_confidence=0.62):
        self.active_trades = {}
        self.sl_pct = sl_pct / 100.0
        self.tp_pct = tp_pct / 100.0
        self.ts_act_pct = ts_act_pct / 100.0
        self.ts_dist_pct = ts_dist_pct / 100.0
        self.max_leverage = 5
        self.kraken_fee = 0.0005
        self.max_trade_duration_minutes = max_trade_duration_minutes
        self.min_signal_confidence = min_signal_confidence

        self.symbols = symbols
        self.timeframe = timeframe
        self.base_currency = base_currency
        self.tf_minutes = get_timeframe_minutes(timeframe)

        self.scaler = StandardScaler()

        self.exchange = ccxt.krakenfutures({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

    def load_kraken_history(self, symbol):
        log_msg(f"🔄 Sincronizando historial {symbol}...")
        history = []
        try:
            trades = self.exchange.fetch_my_trades(symbol, limit=20)
            for t in trades:
                side = 'LONG' if t.get('side') == 'buy' else 'SHORT'
                price = t.get('price', 0)
                qty   = t.get('amount', 0)
                ts    = t.get('timestamp', time.time() * 1000)
                date_str = datetime.datetime.fromtimestamp(ts / 1000).strftime('%d/%m %H:%M')
                history.append({
                    'id': str(t.get('id', 'N/A'))[:8],
                    'side': side, 'entry': price, 'qty': qty, 'lev': '--',
                    'status': 'CLOSED', 'timestamp': date_str, 'pnl_pct': 'Kraken'
                })
        except Exception as e:
            log_msg(f"⚠️ fetch_my_trades {symbol} falló: {e}")

        if not history:
            try:
                orders = self.exchange.fetch_orders(symbol, limit=20, params={'status': 'closed'})
                for o in orders:
                    if o.get('status') not in ('closed', 'filled'): continue
                    side = 'LONG' if o.get('side') == 'buy' else 'SHORT'
                    price = float(o.get('average') or o.get('price') or 0)
                    qty   = float(o.get('filled') or o.get('amount') or 0)
                    ts    = o.get('timestamp', time.time() * 1000)
                    date_str = datetime.datetime.fromtimestamp(ts / 1000).strftime('%d/%m %H:%M')
                    if price == 0 or qty == 0: continue
                    history.append({
                        'id': str(o.get('id', 'N/A'))[:8],
                        'side': side, 'entry': price, 'qty': qty, 'lev': '--',
                        'status': 'CLOSED', 'timestamp': date_str, 'pnl_pct': 'Kraken',
                        'invested_usd': '--', 'pnl_usd': '--'
                    })
            except Exception as e:
                log_msg(f"⚠️ fetch_orders fallback {symbol} falló: {e}")

        history.sort(key=lambda x: x['timestamp'], reverse=True)
        bot_state.trade_histories[symbol] = history[:20]
        log_msg(f"✅ Historial {symbol}: {len(history)} operaciones.")

    def get_real_data(self, symbol):
        """Descarga OHLCV adaptado al timeframe micro con indicadores rápidos."""
        # Calculamos cuántas velas necesitamos según el timeframe
        n_candles = candles_needed_for_window(self.timeframe, train_days=3, val_days=1)
        n_candles = min(n_candles, 800)  # Límite API

        ohlcv = []
        for attempt in range(3):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=n_candles)
                break
            except Exception as e:
                log_msg(f"⚠️ Kraken API ({symbol}): Error intento {attempt+1}/3. Reintentando...")
                time.sleep(3)

        if not ohlcv:
            raise ValueError(f"No se pudieron descargar datos de {symbol}.")

        df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        if len(df) < 100:
            raise ValueError(f"Muy pocas velas ({len(df)}) para {symbol}.")

        df['c'] = pd.to_numeric(df['c'], errors='coerce')
        df['o'] = pd.to_numeric(df['o'], errors='coerce')
        df['h'] = pd.to_numeric(df['h'], errors='coerce')
        df['l'] = pd.to_numeric(df['l'], errors='coerce')
        df['v'] = pd.to_numeric(df['v'], errors='coerce')

        # === INDICADORES BASE ===
        df['ret'] = np.log(df['c'] / df['c'].shift(1)).fillna(0)

        # Volumen relativo (más útil en timeframes cortos)
        df['vol_sma'] = df['v'].rolling(20).mean()
        df['vol_ratio'] = (df['v'] / (df['vol_sma'] + 1e-9)).clip(0, 5).fillna(1.0)

        df['rsi'] = self.calculate_rsi(df['c'], 14)

        # OBV normalizado
        df['obv'] = np.where(df['c'] > df['c'].shift(1), df['v'],
                    np.where(df['c'] < df['c'].shift(1), -df['v'], 0)).cumsum()
        df['obv_diff'] = df['obv'].diff().rolling(5).mean().fillna(0)

        # Fractional diff con ventana más corta para timeframes rápidos
        frac_window = min(20, max(10, len(df) // 20))
        df['frac_diff_ret'] = fractional_diff(df['ret'], 0.4, window=frac_window)

        # === ANATOMÍA DE VELA ===
        candle_range = (df['h'] - df['l']).replace(0, np.nan)
        body = (df['c'] - df['o'])
        df['body_ratio']  = (body / candle_range).fillna(0)
        df['upper_wick']  = ((df['h'] - df[['c','o']].max(axis=1)) / candle_range).fillna(0)
        df['lower_wick']  = ((df[['c','o']].min(axis=1) - df['l']) / candle_range).fillna(0)

        # === CHOPPINESS INDEX (adaptado al TF) ===
        ci_period = min(14, max(8, len(df) // 30))
        atr_sum   = (df['h'] - df['l']).rolling(ci_period).sum()
        high_max  = df['h'].rolling(ci_period).max()
        low_min   = df['l'].rolling(ci_period).min()
        price_range = (high_max - low_min).replace(0, np.nan)
        df['choppiness'] = (100 * np.log10(atr_sum / price_range) / np.log10(ci_period)).fillna(100)

        # === CRYSTAL BALL FEATURES ===
        # MACD rápido adaptado a timeframes cortos
        fast_span = 8 if self.tf_minutes <= 5 else 12
        slow_span = 17 if self.tf_minutes <= 5 else 26
        ema_fast = df['c'].ewm(span=fast_span, adjust=False).mean()
        ema_slow = df['c'].ewm(span=slow_span, adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist   = macd_line - signal_line
        atr14 = (df['h'] - df['l']).rolling(14).mean()
        df['macd_n'] = (macd_hist / (atr14 + 1e-9)).clip(-3, 3).fillna(0)

        # StochRSI
        rsi_min = df['rsi'].rolling(14).min()
        rsi_max = df['rsi'].rolling(14).max()
        stoch = (df['rsi'] - rsi_min) / (rsi_max - rsi_min + 1e-9)
        df['stoch_rsi'] = (stoch * 2 - 1).fillna(0)

        # ADX (Wilder)
        tr_vals = pd.concat([
            df['h'] - df['l'],
            (df['h'] - df['c'].shift(1)).abs(),
            (df['l'] - df['c'].shift(1)).abs()
        ], axis=1).max(axis=1)
        dm_plus  = np.where((df['h'] - df['h'].shift(1)) > (df['l'].shift(1) - df['l']),
                            np.maximum(df['h'] - df['h'].shift(1), 0), 0)
        dm_minus = np.where((df['l'].shift(1) - df['l']) > (df['h'] - df['h'].shift(1)),
                            np.maximum(df['l'].shift(1) - df['l'], 0), 0)
        alpha_adx = 1 / 14
        atr_w  = tr_vals.ewm(alpha=alpha_adx, adjust=False).mean()
        dip    = 100 * pd.Series(dm_plus,  index=df.index).ewm(alpha=alpha_adx, adjust=False).mean() / (atr_w + 1e-9)
        dim    = 100 * pd.Series(dm_minus, index=df.index).ewm(alpha=alpha_adx, adjust=False).mean() / (atr_w + 1e-9)
        dx     = 100 * (dip - dim).abs() / (dip + dim + 1e-9)
        df['adx'] = (dx.ewm(alpha=alpha_adx, adjust=False).mean() / 100).fillna(0)

        # MFI
        tp      = (df['h'] + df['l'] + df['c']) / 3
        raw_mf  = tp * df['v']
        pos_mf  = pd.Series(np.where(tp > tp.shift(1), raw_mf, 0), index=df.index)
        neg_mf  = pd.Series(np.where(tp < tp.shift(1), raw_mf, 0), index=df.index)
        mfi_raw = 100 - (100 / (1 + pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + 1e-9)))
        df['mfi_n'] = ((mfi_raw - 50) / 50).fillna(0)

        # Bollinger %B
        sma20    = df['c'].rolling(20).mean()
        std20    = df['c'].rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        df['bb_pct'] = ((df['c'] - bb_lower) / (bb_upper - bb_lower + 1e-9) - 0.5).clip(-1.5, 1.5).fillna(0)

        # VWAP rolling (ventana adaptada al TF)
        vwap_window = max(12, min(24, 60 // self.tf_minutes))
        tp_vol      = tp * df['v']
        vwap_roll   = tp_vol.rolling(vwap_window).sum() / (df['v'].rolling(vwap_window).sum() + 1e-9)
        df['vwap_dev'] = ((df['c'] - vwap_roll) / (vwap_roll + 1e-9)).clip(-0.05, 0.05).fillna(0)

        # === MICRO-TREND FEATURES EXTRA ===
        # Momentum ultra-corto (clave para microtendencias)
        df['mom1'] = df['ret'].rolling(1).sum().fillna(0)
        df['mom2'] = df['ret'].rolling(2).sum().fillna(0)
        # Aceleración del precio
        df['accel'] = (df['ret'] - df['ret'].shift(1)).fillna(0)
        # Relación volumen vs movimiento (eficiencia)
        df['vol_eff'] = (df['ret'].abs() / (df['vol_ratio'] + 1e-9)).clip(-0.02, 0.02).fillna(0)

        # Funding rate (si disponible)
        df['funding_rate'] = 0.0
        try:
            if self.exchange.has.get('fetchFundingRateHistory', False):
                fr_hist = self.exchange.fetch_funding_rate_history(symbol, limit=200)
            else:
                fr_hist = []
            if fr_hist:
                fr_df = pd.DataFrame([{
                    't_ms': r['timestamp'],
                    'fr':   float(r.get('fundingRate', r.get('funding_rate', 0)))
                } for r in fr_hist]).sort_values('t_ms').drop_duplicates('t_ms').reset_index(drop=True)
                df_s = df.sort_values('t').reset_index(drop=True)
                merged = pd.merge_asof(df_s, fr_df, left_on='t', right_on='t_ms', direction='backward')
                df['funding_rate'] = (merged['fr'].ffill().fillna(0) * 10000).clip(-5, 5).values
            else:
                raise ValueError("fr_hist vacío")
        except:
            try:
                fr = self.exchange.fetch_funding_rate(symbol)
                df['funding_rate'] = float(fr.get('fundingRate', 0)) * 10000
            except:
                df['funding_rate'] = 0.0

        df = df.dropna()

        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        return df, current_price

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def predict_next_candle(self, df, symbol):
        """
        HMM Micro-Trend v12: Optimizado para timeframes cortos (5m-15m).
        - Ventana de entrenamiento adaptada al timeframe
        - Features de momentum ultra-corto
        - Umbral de confianza más alto para reducir señales falsas
        """
        from sklearn.preprocessing import RobustScaler
        from hmmlearn.hmm import GaussianHMM

        start_t = time.time()
        df2 = df.copy()

        # === FEATURES ADAPTADAS A MICRO-TENDENCIAS ===
        df2['ret'] = np.log(df2['c'] / df2['c'].shift(1)).fillna(0)
        df2['mom3'] = df2['ret'].rolling(3).sum().fillna(0)
        df2['mom5'] = df2['ret'].rolling(5).sum().fillna(0)
        df2['atr5'] = ((df2['h'] - df2['l']) / df2['c']).rolling(5).mean().fillna(0)
        df2['rsi_n'] = (df2['rsi'] - 50.0) / 50.0
        df2['obv_roc'] = (df2['obv_diff'] / (df2['obv_diff'].abs().rolling(10).mean() + 1e-8)).clip(-3, 3).fillna(0)

        features = [
            # Momentum corto (MUY importantes para microtendencias)
            'mom1', 'mom2', 'accel', 'vol_eff',
            # Momentum medio
            'frac_diff_ret', 'mom3', 'mom5', 'atr5', 'macd_n',
            # Osciladores
            'rsi_n', 'stoch_rsi', 'mfi_n',
            # Volatilidad / Bandas
            'bb_pct', 'adx',
            # Flujo / Volumen
            'obv_roc', 'vwap_dev',
            # Microestructura
            'body_ratio', 'upper_wick', 'lower_wick',
            # Sentimiento derivados
            'funding_rate'
        ]
        df2 = df2.dropna(subset=features)

        # === VENTANA ADAPTADA AL TIMEFRAME ===
        # Para 5m: ~3 días train + 1 día val = ~576+192 velas
        # Para 15m: ~3 días train + 1 día val = ~192+64 velas
        # Para 1h: ~7 días train + 2 días val = ~168+48 velas
        if self.tf_minutes <= 5:
            val_len   = min(100, len(df2) // 6)
            train_len = min(400, len(df2) - val_len - 10)
        elif self.tf_minutes <= 15:
            val_len   = min(80, len(df2) // 6)
            train_len = min(280, len(df2) - val_len - 10)
        else:
            val_len   = min(60, len(df2) // 6)
            train_len = min(200, len(df2) - val_len - 10)

        total_window = val_len + train_len
        df2 = df2.tail(total_window).reset_index(drop=True)

        if len(df2) < 80:
            return "NEUTRAL", 0.0, 0, {}

        n_total = len(df2)
        n_train = n_total - val_len

        X = df2[features].values

        scaler = RobustScaler()
        X_train_raw = X[:n_train]
        scaler.fit(X_train_raw)
        X_scaled = scaler.transform(X)
        X_scaled = np.clip(X_scaled, -4, 4)

        # PCA adaptado: más componentes para más features
        n_pca = min(8, len(features) - 1, n_train - 1)
        pca = PCA(n_components=n_pca, random_state=42)
        X_pca_train = pca.fit_transform(X_scaled[:n_train])
        X_pca_val   = pca.transform(X_scaled[n_train:])
        X_pca       = np.vstack((X_pca_train, X_pca_val))

        X_train_base = X_pca[:n_train]
        X_val        = X_pca[n_train:]
        ret_val      = df2['ret'].values[n_train:]

        current_time = time.time()
        # Recalibración más frecuente para TFs cortos
        calib_interval = max(900, self.tf_minutes * 60 * 10)  # 10 velas del TF seleccionado
        needs_calibration = (
            symbol not in bot_state.best_seeds or
            len(bot_state.best_seeds[symbol]) < 3 or
            symbol not in bot_state.last_calibration or
            current_time - bot_state.last_calibration[symbol] > calib_interval
        )
        is_calib = needs_calibration

        def _train_single_seed(seed):
            try:
                if len(X_train_base) < 50: return None

                rng = np.random.default_rng(seed)
                noise_scale = rng.uniform(1e-7, 1e-4)
                X_train_noisy = X_train_base + rng.normal(0, noise_scale, X_train_base.shape)

                m = GaussianHMM(
                    n_components=2,
                    covariance_type="full",
                    n_iter=300,   # Menos iteraciones para TFs rápidos
                    random_state=seed,
                    tol=1e-6,
                    min_covar=1e-4
                )
                m.fit(X_train_noisy)
                if m is None: return None

                train_states  = m.predict(X_train_noisy)
                train_returns = df2['ret'].values[:n_train]
                active_state_returns = {}
                for i in range(m.n_components):
                    mask = (train_states == i)
                    if np.any(mask):
                        active_state_returns[i] = np.mean(train_returns[mask])

                if len(active_state_returns) < 2:
                    if is_calib: return None
                    else: return (0, 0, 0, m, seed, 0, 1, 0.5, 0.5)

                sorted_states = sorted(active_state_returns.items(), key=lambda x: x[1])
                bear_s, bull_s = sorted_states[0][0], sorted_states[-1][0]

                means_diff = np.abs(active_state_returns[bull_s] - active_state_returns[bear_s])
                if is_calib and means_diff < 0.00005: return None

                try:
                    val_states = m.predict(X_val)
                except:
                    if is_calib: return None

                if is_calib:
                    state_counts = np.bincount(val_states, minlength=m.n_components)
                    if (state_counts.max() / len(val_states)) > 0.97: return None

                sharpe = 0.0
                try:
                    val_probs = m.predict_proba(X_val)
                    pnl_per_step = []
                    for i in range(len(val_probs) - 1):
                        pred_next    = val_probs[i] @ m.transmat_
                        p_bull, p_bear = pred_next[bull_s], pred_next[bear_s]
                        conviction   = abs(p_bull - p_bear)
                        if conviction < 0.08: continue
                        direction = 1.0 if p_bull > p_bear else -1.0
                        pnl_per_step.append(direction * ret_val[i + 1] * conviction)

                    if pnl_per_step:
                        pnl_arr   = np.array(pnl_per_step)
                        total_pnl = pnl_arr.sum()
                        sharpe    = total_pnl / (pnl_arr.std() + 1e-6)
                except: pass

                if is_calib and sharpe < 0.0: return None

                try: ll_score = m.score(X_train_noisy)
                except: ll_score = 0.0

                full_probs  = m.predict_proba(X_pca)
                recent_post = full_probs[-1]
                pred_signal = recent_post @ m.transmat_
                final_bull  = float(pred_signal[bull_s])
                final_bear  = float(pred_signal[bear_s])

                return (ll_score, means_diff, sharpe, m, seed, bull_s, bear_s, final_bull, final_bear)
            except:
                return None

        # === ENSEMBLE — menos seeds para TFs rápidos (más frecuente) ===
        if needs_calibration:
            log_msg(f"⚙️ {symbol} Calibrando HMM Micro-Trend ({self.timeframe})...")
            seeds_to_test = list(range(1, 61))  # 60 seeds (antes 100)
        else:
            seeds_to_test = bot_state.best_seeds[symbol]

        try:
            max_workers = min(6, len(seeds_to_test))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = [r for r in executor.map(_train_single_seed, seeds_to_test) if r is not None]
        except Exception as e:
            log_msg(f"⚠️ Error en ThreadPool: {str(e)}")
            results = []

        if len(results) < 1:
            return "NEUTRAL", 0.0, 0, {}

        results.sort(key=lambda x: (x[2], x[1]), reverse=True)
        top_results = results[:3]
        log_msg(f"✅ {symbol} HMM OK: {len(results)} semillas | Sharpe: {results[0][2]:.4f}")

        if needs_calibration:
            bot_state.best_seeds[symbol] = [r[4] for r in top_results]
            bot_state.last_calibration[symbol] = current_time
            log_msg(f"✅ {symbol} Semillas fijadas: {bot_state.best_seeds[symbol]}")

        # === WEIGHTED ENSEMBLE ===
        ensemble_bull, ensemble_bear = [], []
        weights = []
        series_len = min(60, len(df2))
        acc_bull_series = np.zeros(series_len)
        acc_bear_series = np.zeros(series_len)

        for r in top_results:
            model  = r[3]
            bull_s = r[5]
            bear_s = r[6]
            weight = max(r[2], 0.01)

            try:
                state_probs = model.predict_proba(X_pca)
                ensemble_bull.append(r[7] * weight)
                ensemble_bear.append(r[8] * weight)
                weights.append(weight)

                for i in range(series_len):
                    idx_c = len(df2) - series_len + i
                    p_pred = state_probs[idx_c - 1] @ model.transmat_ if idx_c > 0 else state_probs[idx_c]
                    acc_bull_series[i] += p_pred[bull_s] * weight
                    acc_bear_series[i] += p_pred[bear_s] * weight
            except: continue

        if not ensemble_bull:
            return "NEUTRAL", 0.0, 0, {}

        w_total = sum(weights)
        prob_next_bullish = float(np.sum(ensemble_bull) / w_total)
        prob_next_bearish = float(np.sum(ensemble_bear) / w_total)

        seed_details = []
        for r in top_results:
            seed_details.append({
                "seed":  r[4],
                "score": float(round(r[2], 5)),
                "bull":  float(round(r[7], 4)),
                "bear":  float(round(r[8], 4))
            })

        current_probs = {
            "Bajista (t+1)":      prob_next_bearish,
            "Neutral/Rango (t+1)": 0.0,
            "Alcista (t+1)":      prob_next_bullish,
            "seeds":              seed_details
        }

        # === FILTRO DE SEÑAL MICRO-TREND ===
        # Umbral más alto para reducir señales falsas en TFs cortos
        signal     = "NEUTRAL"
        confidence = max(prob_next_bullish, prob_next_bearish)
        margin     = 0.06  # Margen de diferencia requerido

        # Umbral adaptado: más exigente en TFs muy cortos para filtrar ruido
        if self.tf_minutes <= 5:
            threshold = max(self.min_signal_confidence, 0.63)
        elif self.tf_minutes <= 15:
            threshold = max(self.min_signal_confidence, 0.61)
        else:
            threshold = max(self.min_signal_confidence, 0.58)

        if prob_next_bullish >= threshold and (prob_next_bullish - prob_next_bearish) > margin:
            signal = "LONG"
        elif prob_next_bearish >= threshold and (prob_next_bearish - prob_next_bullish) > margin:
            signal = "SHORT"

        # === ACTUALIZAR CHART DATA ===
        chart_series = []
        for i in range(series_len):
            idx_c  = len(df2) - series_len + i
            candle = df2.iloc[idx_c]
            chart_series.append({
                'time':        int(float(candle['t']) / 1000),
                'open':        float(candle['o']),
                'high':        float(candle['h']),
                'low':         float(candle['l']),
                'close':       float(candle['c']),
                'probBull':    float(acc_bull_series[i] / w_total) if w_total > 0 else 0.0,
                'probBear':    float(acc_bear_series[i] / w_total) if w_total > 0 else 0.0,
                'probNeutral': 0.0
            })

        bot_state.chart_data[symbol] = chart_series
        elapsed = time.time() - start_t
        log_msg(f"🧠 {symbol} [{self.timeframe}] → {signal} ({elapsed:.1f}s) | Bull:{prob_next_bullish:.1%} Bear:{prob_next_bearish:.1%}")
        return signal, confidence, 5, current_probs

    def execute_order(self, symbol, signal, confidence, target_lev, current_price):
        try:
            lev = target_lev
            if lev == 0: return
            self.exchange.set_leverage(lev, symbol)

            balance = self.exchange.fetch_balance()
            if self.base_currency not in balance or 'free' not in balance[self.base_currency]:
                log_msg(f"❌ Error: No capital libre en {self.base_currency}.")
                return

            usd_available     = float(balance[self.base_currency]['free'])
            bot_state.balance = usd_available

            margin_per_asset = bot_state.max_margin_usd / len(self.symbols)
            margin_to_use    = min(usd_available, margin_per_asset)

            if margin_to_use < 5:
                log_msg(f"❌ Saldo insuficiente para {symbol} (${margin_to_use:.2f}).")
                return

            ticker_actual  = self.exchange.fetch_ticker(symbol)['last']
            notional_value = margin_to_use / ((1.0 / lev) + self.kraken_fee)
            qty = float(self.exchange.amount_to_precision(symbol, notional_value / ticker_actual))
            if qty <= 0:
                log_msg(f"❌ Error {symbol}: qty=0.")
                return

            side  = 'buy' if signal == "LONG" else 'sell'
            order = self.exchange.create_market_order(symbol, side, qty)
            real_entry_price = float(order.get('average') or order.get('price') or ticker_actual)

            trade = {
                'id':           str(order.get('id', 'N/A'))[:8],
                'side':         signal,
                'entry':        real_entry_price,
                'qty':          abs(qty),
                'lev':          lev,
                'invested_usd': float(margin_to_use),
                'pnl_usd':      0.0,
                'pnl_pct':      0.0,
                'max_pnl_pct':  0.0,
                'current_price': real_entry_price,
                'status':       'OPEN',
                'timestamp':    datetime.datetime.now().strftime('%d/%m %H:%M'),
                'open_time':    time.time()   # MICRO-TREND: tracking de tiempo
            }
            self.active_trades[symbol] = trade
            bot_state.trade_open_time[symbol] = time.time()
            log_msg(f"🚀 {symbol}: {signal} | {lev}x | Entry: {real_entry_price:.4f} | TF: {self.timeframe}")

            if symbol not in bot_state.trade_histories:
                bot_state.trade_histories[symbol] = []
            bot_state.trade_histories[symbol].insert(0, trade.copy())
            if len(bot_state.trade_histories[symbol]) > 20:
                bot_state.trade_histories[symbol].pop()

        except Exception as e:
            log_msg(f"❌ Error en Kraken {symbol}: {e}")

    def manage_risk(self, symbol, current_price, rsi=None):
        if symbol not in self.active_trades: return
        trade = self.active_trades[symbol]

        entry = trade['entry']
        side  = trade['side']
        qty   = trade['qty']

        diff_decimal = (current_price - entry) / entry if side == "LONG" else (entry - current_price) / entry
        pnl_pct  = diff_decimal * 100 * trade.get('lev', 5)
        pnl_usd  = trade.get('invested_usd', 0) * diff_decimal * trade.get('lev', 5)

        trade['pnl_pct']      = pnl_pct
        trade['pnl_usd']      = pnl_usd
        trade['current_price'] = current_price

        current_max_pnl = trade.get('max_pnl_pct', 0.0)
        if pnl_pct > current_max_pnl:
            trade['max_pnl_pct'] = pnl_pct
            current_max_pnl      = pnl_pct

        motivo_cierre = None

        # === MICRO-TREND: TIMEOUT POR DURACIÓN MÁXIMA ===
        open_time   = trade.get('open_time', bot_state.trade_open_time.get(symbol, time.time()))
        elapsed_min = (time.time() - open_time) / 60.0
        if elapsed_min >= self.max_trade_duration_minutes:
            motivo_cierre = f"Timeout ({elapsed_min:.0f}min ≥ {self.max_trade_duration_minutes}min)"

        # === SMART EXITS POR RSI (umbrales ajustados para TFs cortos) ===
        if not motivo_cierre and pnl_pct > 0.3:
            rsi_exit_long  = 78 if self.tf_minutes <= 15 else 80
            rsi_exit_short = 22 if self.tf_minutes <= 15 else 20
            if side == "LONG" and rsi is not None and rsi > rsi_exit_long:
                motivo_cierre = f"Smart Exit RSI ({rsi:.1f} > {rsi_exit_long})"
            elif side == "SHORT" and rsi is not None and rsi < rsi_exit_short:
                motivo_cierre = f"Smart Exit RSI ({rsi:.1f} < {rsi_exit_short})"

        # === GESTIÓN DE RIESGO TRADICIONAL ===
        sl_threshold = self.sl_pct * 100
        tp_threshold = self.tp_pct * 100

        if not motivo_cierre:
            if pnl_pct <= -sl_threshold:
                motivo_cierre = f"Stop Loss ({pnl_pct:.2f}%)"
                bot_state.sl_cooldown[symbol] = time.time()
            elif pnl_pct >= tp_threshold:
                motivo_cierre = f"Take Profit ({pnl_pct:.2f}%)"
            else:
                act_threshold_pct = self.ts_act_pct * 100
                dist_pct          = self.ts_dist_pct * 100
                if (current_max_pnl >= act_threshold_pct) and (pnl_pct <= current_max_pnl - dist_pct):
                    motivo_cierre = f"Trailing Stop (Máx PnL {current_max_pnl:.2f}%)"

        if motivo_cierre:
            log_msg(f"🔒 {symbol}: Cerrando ({motivo_cierre}) | {pnl_pct:.2f}% | {elapsed_min:.1f}min")
            try:
                close_side = 'sell' if side == "LONG" else 'buy'
                safe_qty   = float(self.exchange.amount_to_precision(symbol, qty))
                self.exchange.create_market_order(symbol, close_side, safe_qty, params={'reduceOnly': True})

                for t in bot_state.trade_histories.get(symbol, []):
                    if t.get('status') == 'OPEN' and t.get('id') == trade['id']:
                        t['status']       = 'CLOSED'
                        t['pnl_pct']      = round(pnl_pct, 2)
                        t['pnl_usd']      = round(pnl_usd, 2)
                        t['invested_usd'] = round(trade.get('invested_usd', 0), 2)
                        break

                del self.active_trades[symbol]
                if symbol in bot_state.trade_open_time:
                    del bot_state.trade_open_time[symbol]
                log_msg(f"✅ {symbol} cerrada | PnL: {pnl_pct:+.2f}%")

            except Exception as e:
                log_msg(f"❌ Error cerrar {symbol}: {e}")

    def force_close_position(self, symbol, reason):
        if symbol not in self.active_trades: return
        trade = self.active_trades[symbol]
        side  = trade['side']
        qty   = trade['qty']
        pnl   = trade.get('pnl_pct', 0.0)
        log_msg(f"🔒 Cerrando forzosamente {symbol}: {reason}")
        try:
            close_side = 'sell' if side == "LONG" else 'buy'
            safe_qty   = float(self.exchange.amount_to_precision(symbol, qty))
            self.exchange.create_market_order(symbol, close_side, safe_qty, params={'reduceOnly': True})

            for t in bot_state.trade_histories.get(symbol, []):
                if t.get('status') == 'OPEN' and t.get('id') == trade['id']:
                    t['status']       = 'CLOSED'
                    t['pnl_pct']      = f"{pnl:.2f}%" if isinstance(pnl, float) else str(pnl)
                    pnl_u  = trade.get('pnl_usd', 0.0)
                    inv_u  = trade.get('invested_usd', 0.0)
                    t['pnl_usd']      = f"{pnl_u:.2f}$" if isinstance(pnl_u, float) else str(pnl_u)
                    t['invested_usd'] = f"{inv_u:.2f}$" if isinstance(inv_u, float) else str(inv_u)
                    break

            del self.active_trades[symbol]
            if symbol in bot_state.trade_open_time:
                del bot_state.trade_open_time[symbol]
            log_msg(f"✅ {symbol} cerrada forzosamente.")
        except Exception as e:
            log_msg(f"❌ Error force close {symbol}: {e}")


# ==========================================
# HILO SECUNDARIO DEL BOT — MICRO-TREND
# ==========================================
def bot_worker():
    log_msg("🤖 Motor HMM Micro-Trend iniciando...")
    bot_state.engine = HMMTradingEngine(
        api_key=bot_state.api_key,
        api_secret=bot_state.api_secret,
        symbols=bot_state.symbols,
        timeframe=bot_state.timeframe,
        base_currency=bot_state.base_currency,
        sl_pct=bot_state.sl_pct,
        tp_pct=bot_state.tp_pct,
        ts_act_pct=bot_state.ts_activation_pct,
        ts_dist_pct=bot_state.ts_distance_pct,
        max_trade_duration_minutes=bot_state.max_trade_duration_minutes,
        min_signal_confidence=bot_state.min_signal_confidence
    )

    tf_minutes = get_timeframe_minutes(bot_state.timeframe)

    try:
        bal = bot_state.engine.exchange.fetch_balance()
        if bot_state.base_currency in bal and 'free' in bal[bot_state.base_currency]:
            bot_state.balance = float(bal[bot_state.base_currency]['free'])
            log_msg(f"💰 Saldo inicial: {bot_state.balance:.2f} {bot_state.base_currency}")

        for sym in bot_state.symbols:
            bot_state.engine.load_kraken_history(sym)

            open_positions = bot_state.engine.exchange.fetch_positions(symbols=[sym])
            for pos in (open_positions or []):
                qty_raw = pos.get('contracts')
                qty     = float(qty_raw) if qty_raw is not None else 0.0
                if qty > 0:
                    side      = 'LONG' if str(pos.get('side')).lower() in ['long', 'buy'] else 'SHORT'
                    entry_raw = pos.get('entryPrice')
                    entry     = float(entry_raw) if entry_raw is not None else 0.0
                    im_raw    = pos.get('initialMargin')
                    inv_usd   = float(im_raw) if im_raw is not None else (qty * entry / 5.0)
                    trade     = {
                        'id': 'RECUPERADA', 'side': side, 'entry': entry, 'qty': qty, 'lev': 5,
                        'invested_usd': inv_usd, 'pnl_usd': 0.0,
                        'pnl_pct': float(pos.get('percentage') or 0.0),
                        'max_pnl_pct': float(pos.get('percentage') or 0.0),
                        'current_price': entry, 'status': 'OPEN',
                        'timestamp': datetime.datetime.now().strftime('%d/%m %H:%M'),
                        'open_time': time.time()
                    }
                    bot_state.engine.active_trades[sym] = trade
                    bot_state.trade_open_time[sym]      = time.time()
                    if sym not in bot_state.trade_histories:
                        bot_state.trade_histories[sym] = []
                    bot_state.trade_histories[sym].insert(0, trade.copy())
                    log_msg(f"🔄 {sym} recuperado.")
                    break

    except Exception as e:
        log_msg(f"⚠️ Aviso inicialización: {e}")

    while bot_state.is_running:

        # === SINCRONIZACIÓN DE POSICIONES EN TIEMPO REAL ===
        try:
            open_positions = bot_state.engine.exchange.fetch_positions(symbols=bot_state.symbols)
            open_syms      = []
            for pos in (open_positions or []):
                qty_raw = pos.get('contracts')
                qty     = float(qty_raw) if qty_raw is not None else 0.0
                sym     = pos.get('symbol')
                if qty > 0 and sym in bot_state.symbols:
                    open_syms.append(sym)
                    if sym not in bot_state.engine.active_trades:
                        side    = 'LONG' if str(pos.get('side')).lower() in ['long', 'buy'] else 'SHORT'
                        entry_raw = pos.get('entryPrice')
                        entry   = float(entry_raw) if entry_raw is not None else 0.0
                        im_raw  = pos.get('initialMargin')
                        inv_usd = float(im_raw) if im_raw is not None else (qty * entry / 5.0)
                        trade   = {
                            'id': 'SYNC', 'side': side, 'entry': entry, 'qty': qty, 'lev': 5,
                            'invested_usd': inv_usd, 'pnl_usd': 0.0,
                            'pnl_pct': float(pos.get('percentage') or 0.0),
                            'max_pnl_pct': float(pos.get('percentage') or 0.0),
                            'current_price': entry, 'status': 'OPEN',
                            'timestamp': datetime.datetime.now().strftime('%d/%m %H:%M'),
                            'open_time': time.time()
                        }
                        bot_state.engine.active_trades[sym] = trade
                        bot_state.trade_open_time[sym]      = time.time()
                        log_msg(f"🔄 {sym} sincronizado desde Kraken.")

            for sym in list(bot_state.engine.active_trades.keys()):
                if sym not in open_syms:
                    for t in bot_state.trade_histories.get(sym, []):
                        if t.get('status') == 'OPEN':
                            t['status'] = 'CLOSED'
                    del bot_state.engine.active_trades[sym]
                    if sym in bot_state.trade_open_time:
                        del bot_state.trade_open_time[sym]
                    log_msg(f"🧹 {sym} limpiado (cerrado externamente).")
        except:
            pass

        for symbol in bot_state.symbols:
            if not bot_state.is_running: break

            bot_state.last_updates[symbol] = datetime.datetime.now().strftime('%H:%M:%S')
            try:
                df, price = bot_state.engine.get_real_data(symbol)

                rsi = float(df['rsi'].iloc[-1])
                ci_val = float(df['choppiness'].iloc[-1])
                bot_state.indicators[symbol] = {'rsi': rsi}

                if symbol not in bot_state.chart_data or len(bot_state.chart_data[symbol]) == 0:
                    series_len = 60
                    df_v       = df.tail(series_len)
                    chart_series = []
                    for _, candle in df_v.iterrows():
                        chart_series.append({
                            'time':   int(float(candle['t']) / 1000),
                            'open':   float(candle['o']), 'high':  float(candle['h']),
                            'low':    float(candle['l']), 'close': float(candle['c']),
                            'probBull': 0.0, 'probBear': 0.0, 'probNeutral': 1.0
                        })
                    bot_state.chart_data[symbol] = chart_series

                current_candle_ts = df['t'].iloc[-1]

                if symbol not in bot_state.last_candle_timestamp:
                    bot_state.last_candle_timestamp[symbol] = 0

                if current_candle_ts > bot_state.last_candle_timestamp[symbol]:
                    log_msg(f"⏳ {symbol} Nueva vela [{bot_state.timeframe}]. Prediciendo...")

                    df_closed = df.iloc[:-1]
                    signal, confidence, target_lev, current_probs = bot_state.engine.predict_next_candle(df_closed, symbol)

                    bot_state.last_candle_timestamp[symbol]   = current_candle_ts
                    bot_state.current_hmm_signal[symbol]      = signal
                    bot_state.current_hmm_confidence[symbol]  = confidence
                    bot_state.current_probs_dict[symbol]      = current_probs

                    # Cierre predictivo si señal cambia
                    if symbol in bot_state.engine.active_trades:
                        current_side = bot_state.engine.active_trades[symbol]['side']
                        if (current_side == "LONG" and signal == "SHORT") or \
                           (current_side == "SHORT" and signal == "LONG"):
                            log_msg(f"🔀 {symbol} Reversión de señal → {signal}. Cerrando posición.")
                            bot_state.engine.force_close_position(symbol, f"Reversión HMM → {signal}")

                if symbol in bot_state.chart_data and len(bot_state.chart_data[symbol]) > 0:
                    last_c = bot_state.chart_data[symbol][-1]
                    if last_c['time'] == int(current_candle_ts / 1000):
                        last_c['close'] = price
                        if price > last_c['high']: last_c['high'] = price
                        if price < last_c['low']:  last_c['low']  = price

                hmm_signal     = bot_state.current_hmm_signal.get(symbol, "NEUTRAL")
                hmm_confidence = bot_state.current_hmm_confidence.get(symbol, 0)
                bot_state.indicators[symbol]['last_signal'] = hmm_signal

                if symbol in bot_state.engine.active_trades:
                    bot_state.engine.manage_risk(symbol, price, rsi=rsi)
                else:
                    # === FILTRO COOLDOWN SL (reducido para TFs cortos) ===
                    cooldown_seconds = max(900, tf_minutes * 60 * 3)  # 3 velas del TF
                    last_sl_time     = bot_state.sl_cooldown.get(symbol, 0)
                    if time.time() - last_sl_time < cooldown_seconds:
                        remaining = int((cooldown_seconds - (time.time() - last_sl_time)) / 60)
                        log_msg(f"⏸️ {symbol} Cooldown SL activo ({remaining}min). Saltando.")
                    else:
                        # === FILTRO CHOPPINESS INDEX ===
                        if ci_val > 61.8:
                            log_msg(f"📊 {symbol} Mercado lateral (CI={ci_val:.1f}). Saltando.")
                        elif hmm_signal == "LONG":
                            # Filtro RSI extra para TFs muy cortos
                            rsi_ok = rsi < 75 if tf_minutes <= 15 else rsi < 80
                            if rsi_ok:
                                bot_state.engine.execute_order(symbol, "LONG", hmm_confidence, 5, price)
                        elif hmm_signal == "SHORT":
                            rsi_ok = rsi > 25 if tf_minutes <= 15 else rsi > 20
                            if rsi_ok:
                                bot_state.engine.execute_order(symbol, "SHORT", hmm_confidence, 5, price)

                bot_state.open_positions[symbol] = bot_state.engine.active_trades.get(symbol)

            except Exception as e:
                log_msg(f"⚠️ Error {symbol}: {e}")

        # Balance
        try:
            bal = bot_state.engine.exchange.fetch_balance()
            if bot_state.base_currency in bal:
                if 'free'  in bal[bot_state.base_currency]:
                    bot_state.balance       = float(bal[bot_state.base_currency]['free'])
                if 'total' in bal[bot_state.base_currency]:
                    bot_state.total_balance = float(bal[bot_state.base_currency]['total'])
        except: pass

        # === CICLO ADAPTADO AL TIMEFRAME ===
        # Para 5m: poll cada 15s. Para 15m: cada 30s. Para 1h: cada 30s.
        poll_interval = max(15, min(30, tf_minutes * 3))
        for _ in range(poll_interval):
            if not bot_state.is_running: break
            time.sleep(1)

    log_msg("🛑 Motor Micro-Trend finalizado.")


# ==========================================
# RUTAS DEL SERVIDOR WEB FLASK
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_bot():
    if bot_state.is_running:
        return jsonify({"status": "error", "message": "Ya está en ejecución."})

    data = request.json
    bot_state.api_key    = data.get('api_key', '')
    bot_state.api_secret = data.get('api_secret', '')
    bot_state.max_margin_usd = float(data.get('max_margin', 200))

    sym_data = data.get('symbol', 'BTC/USD:USD')
    if isinstance(sym_data, str):
        bot_state.symbols = [s.strip() for s in sym_data.split(',')]
    else:
        bot_state.symbols = sym_data

    bot_state.base_currency           = data.get('base_currency', 'USD')
    bot_state.timeframe               = data.get('timeframe', '5m')
    bot_state.sl_pct                  = float(data.get('sl_pct', 0.8))
    bot_state.tp_pct                  = float(data.get('tp_pct', 1.5))
    bot_state.ts_activation_pct       = float(data.get('ts_act_pct', 0.5))
    bot_state.ts_distance_pct         = float(data.get('ts_dist_pct', 0.3))
    bot_state.max_trade_duration_minutes = float(data.get('max_trade_duration', 60))
    bot_state.min_signal_confidence   = float(data.get('min_confidence', 0.62))

    bot_state.is_running        = True
    bot_state.logs              = []
    bot_state.open_positions    = {}
    bot_state.trade_histories   = {}
    bot_state.indicators        = {}
    bot_state.last_candle_timestamp    = {}
    bot_state.current_hmm_signal      = {}
    bot_state.current_hmm_confidence  = {}
    bot_state.current_probs_dict      = {}
    bot_state.last_updates            = {}
    bot_state.signal_history          = {}
    bot_state.best_seeds              = {}
    bot_state.last_calibration        = {}
    bot_state.sl_cooldown             = {}
    bot_state.trade_open_time         = {}

    t = threading.Thread(target=bot_worker)
    t.daemon = True
    t.start()

    return jsonify({"status": "started"})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    bot_state.is_running = False
    return jsonify({"status": "stopped"})

@app.route('/api/state')
def get_state():
    return jsonify({
        "is_running":       bot_state.is_running,
        "balance":          f"{bot_state.balance:.2f}",
        "total_balance":    f"{getattr(bot_state, 'total_balance', bot_state.balance):.2f}",
        "base_currency":    bot_state.base_currency,
        "symbols":          bot_state.symbols,
        "logs":             bot_state.logs,
        "open_positions":   bot_state.open_positions,
        "last_updates":     bot_state.last_updates,
        "current_probs_dict": bot_state.current_probs_dict,
        "trade_histories":  bot_state.trade_histories,
        "indicators":       bot_state.indicators
    })

@app.route('/api/chart_data')
def get_chart_data():
    symbol = request.args.get('symbol', '').strip()
    if not symbol: return jsonify([])

    data = bot_state.chart_data.get(symbol, [])

    if not data:
        try:
            public_ex = ccxt.krakenfutures({'enableRateLimit': True})
            ohlcv = public_ex.fetch_ohlcv(symbol, bot_state.timeframe or '5m', limit=100)
            chart_series = []
            for candle in ohlcv:
                chart_series.append({
                    'time':  int(float(candle[0]) / 1000),
                    'open':  float(candle[1]), 'high':  float(candle[2]),
                    'low':   float(candle[3]), 'close': float(candle[4]),
                    'probBull': 0.0, 'probBear': 0.0, 'probNeutral': 1.0
                })
            return jsonify(chart_series)
        except Exception as e:
            app.logger.error(f"Error chart data {symbol}: {e}")

    try:
        import math
        def sanitize_nans(obj):
            if isinstance(obj, dict):
                return {k: sanitize_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_nans(x) for x in obj]
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0
            return obj
        return jsonify(sanitize_nans(data))
    except Exception as e:
        import json
        safe_json = json.dumps(data, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
        safe_json = safe_json.replace('NaN', '0.0').replace('Infinity', '0.0')
        return app.response_class(response=safe_json, status=200, mimetype='application/json')

try:
    from dotenv import load_dotenv
    load_dotenv()
except: pass

@app.route('/ping')
def ping():
    return jsonify({"status": "ok", "running": bot_state.is_running})

def autostart_from_env():
    api_key    = os.environ.get('KRAKEN_API_KEY', '')
    api_secret = os.environ.get('KRAKEN_API_SECRET', '')

    if not api_key or not api_secret:
        print("[AUTOSTART] No se encontraron KRAKEN_API_KEY/SECRET. Arranca manualmente.")
        return

    print("[AUTOSTART] ✅ Variables detectadas. Iniciando bot automáticamente...")
    bot_state.api_key    = api_key
    bot_state.api_secret = api_secret
    bot_state.max_margin_usd  = float(os.environ.get('MAX_MARGIN', 200))
    sym_env               = os.environ.get('SYMBOL', 'BTC/USD:USD')
    bot_state.symbols     = [s.strip() for s in sym_env.split(',')]
    bot_state.base_currency   = os.environ.get('BASE_CURRENCY', 'USD')
    bot_state.timeframe       = os.environ.get('TIMEFRAME', '5m')
    bot_state.sl_pct          = float(os.environ.get('SL_PCT', 0.8))
    bot_state.tp_pct          = float(os.environ.get('TP_PCT', 1.5))
    bot_state.ts_activation_pct  = float(os.environ.get('TS_ACT_PCT', 0.5))
    bot_state.ts_distance_pct    = float(os.environ.get('TS_DIST_PCT', 0.3))
    bot_state.max_trade_duration_minutes = float(os.environ.get('MAX_TRADE_DURATION', 60))
    bot_state.min_signal_confidence      = float(os.environ.get('MIN_CONFIDENCE', 0.62))

    bot_state.is_running = True
    t = threading.Thread(target=bot_worker)
    t.daemon = True
    t.start()

autostart_from_env()

if __name__ == '__main__':
    print("\n-----------------------------------------------------------")
    print("PANEL INTERACTIVO MICRO-TREND LISTO")
    print("1. Abre tu navegador: http://127.0.0.1:5000")
    print("-----------------------------------------------------------\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
