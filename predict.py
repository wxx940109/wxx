import os
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import scipy.signal as signal
import multiprocessing as mp
from scipy.signal import hanning
from scipy.signal import hilbert
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor


import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm
warnings.filterwarnings("ignore")

OUTPUT_DIR = './result'  # set for local environment
DATA_DIR = './input'  # set for local environment

SIG_LEN = 150000
NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6        # 线程数

NY_FREQ_IDX = 75000  # 测试信号有150000个采样点，故奈奎斯特信号至少是75000
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500


# 特征创建。数据平均分成6份，每一份包括前面的一个数据
def split_raw_data():
    print('start read')
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))  # 629145480
    print('read over')

    max_start_index = len(df.index) - SIG_LEN    # SIG_LEN = 150000
    slice_len = int(max_start_index / 6)

    for i in range(NUM_THREADS):
        print('working', i)
        df0 = df.iloc[slice_len * i: (slice_len * (i + 1)) + SIG_LEN]
        df0.to_csv(os.path.join(DATA_DIR, 'raw_data_%d.csv' % i), index=False)
        del df0
    del df


# 采用离散均匀分布，在（0， 100000000）抽取4000个点
def build_rnd_idxs():
    rnd_idxs = np.zeros(shape=(NUM_THREADS, NUM_SEG_PER_PROC), dtype=np.int32)  # NUM_SEG_PER_PROC=4000
    max_start_idx = 100000000  # 一个文件有104832580个数据

    for i in range(NUM_THREADS):
        np.random.seed(5591 + i)
        start_indices = np.random.randint(0, max_start_idx, size=NUM_SEG_PER_PROC, dtype=np.int32)
        rnd_idxs[i, :] = start_indices

    for i in range(NUM_THREADS):
        print(rnd_idxs[i, :8])
        print(rnd_idxs[i, -8:])
        print(min(rnd_idxs[i, :]), max(rnd_idxs[i, :]))

    np.savetxt(fname=os.path.join(OUTPUT_DIR, 'start_indices_4k.csv'), X=np.transpose(rnd_idxs), fmt='%d', delimiter=',')


# linear regression on a portion of the signal
# 用曲线拟合后的斜率来表示趋势
def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]


# short term average divided by the long term average.
def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta


# 巴特沃兹4阶递归滤波器——低通    Wn：归一化截止频率。Wn=2*截止频率/采样频率
# b，a: IIR滤波器的分子（b）和分母（a）多项式系数向量
def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter
    b, a = signal.butter(4, Wn=cutoff/NY_FREQ_IDX)
    return b, a


# 高通   截止频率20000
def des_bw_filter_hp(cutoff=CUTOFF):  # high pass filter
    b, a = signal.butter(4, Wn=cutoff/NY_FREQ_IDX, btype='highpass')
    return b, a


# 带通  2500~20000
def des_bw_filter_bp(low, high):  # band pass filter
    b, a = signal.butter(4, Wn=(low/NY_FREQ_IDX, high/NY_FREQ_IDX), btype='bandpass')
    return b, a


def create_features(seg_id, seg, X, st, end):
    try:
        X.loc[seg_id, 'seg_id'] = np.int32(seg_id)
        X.loc[seg_id, 'seg_start'] = np.int32(st)
        X.loc[seg_id, 'seg_end'] = np.int32(end)
    except:
        pass

    xc = pd.Series(seg['acoustic_data'].values)
    xcdm = xc - np.mean(xc)

    b, a = des_bw_filter_lp(cutoff=18000)
    xcz = signal.lfilter(b, a, xcdm)  # xcdm为要过滤的信号

    zc = np.fft.fft(xcz)    # 快速傅里叶变换
    zc = zc[:MAX_FREQ_IDX]

    # FFT transform values
    realFFT = np.real(zc)   # 实部
    imagFFT = np.imag(zc)   # 虚部

    freq_bands = [x for x in range(0, MAX_FREQ_IDX, FREQ_STEP)]  # [0, 20000, 2500]
    magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)  # 幅度
    phzFFT = np.arctan(imagFFT / realFFT)          # 相位
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0       # 替换无穷数
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)                  # 填充缺失值（0）

    for freq in freq_bands:
        X.loc[seg_id, 'FFT_Mag_01q%d' % freq] = pd.Series(magFFT[freq: freq + FREQ_STEP]).quantile(0.01)
        X.loc[seg_id, 'FFT_Mag_10q%d' % freq] = pd.Series(magFFT[freq: freq + FREQ_STEP]).quantile(0.1)
        X.loc[seg_id, 'FFT_Mag_90q%d' % freq] = pd.Series(magFFT[freq: freq + FREQ_STEP]).quantile(0.9)
        X.loc[seg_id, 'FFT_Mag_99q%d' % freq] = pd.Series(magFFT[freq: freq + FREQ_STEP]).quantile(0.99)
        X.loc[seg_id, 'FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + FREQ_STEP])

        X.loc[seg_id, 'FFT_Phz_mean%d' % freq] = np.mean(phzFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Phz_std%d' % freq] = np.std(phzFFT[freq: freq + FREQ_STEP])

    X.loc[seg_id, 'FFT_Rmean'] = realFFT.mean()
    X.loc[seg_id, 'FFT_Rstd'] = realFFT.std()
    X.loc[seg_id, 'FFT_Rmax'] = realFFT.max()
    X.loc[seg_id, 'FFT_Rmin'] = realFFT.min()
    X.loc[seg_id, 'FFT_Imean'] = imagFFT.mean()
    X.loc[seg_id, 'FFT_Istd'] = imagFFT.std()
    X.loc[seg_id, 'FFT_Imax'] = imagFFT.max()
    X.loc[seg_id, 'FFT_Imin'] = imagFFT.min()

    X.loc[seg_id, 'FFT_Rmean_first_6000'] = realFFT[:6000].mean()
    X.loc[seg_id, 'FFT_Rstd__first_6000'] = realFFT[:6000].std()
    X.loc[seg_id, 'FFT_Rmax_first_6000'] = realFFT[:6000].max()
    X.loc[seg_id, 'FFT_Rmin_first_6000'] = realFFT[:6000].min()
    X.loc[seg_id, 'FFT_Rmean_first_18000'] = realFFT[:18000].mean()
    X.loc[seg_id, 'FFT_Rstd_first_18000'] = realFFT[:18000].std()
    X.loc[seg_id, 'FFT_Rmax_first_18000'] = realFFT[:18000].max()
    X.loc[seg_id, 'FFT_Rmin_first_18000'] = realFFT[:18000].min()

    del xcz
    del zc

    b, a = des_bw_filter_lp(cutoff=2500)
    xc0 = signal.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=2500, high=5000)
    xc1 = signal.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=5000, high=7500)
    xc2 = signal.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=7500, high=10000)
    xc3 = signal.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=10000, high=12500)
    xc4 = signal.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=12500, high=15000)
    xc5 = signal.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=15000, high=17500)
    xc6 = signal.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=17500, high=20000)
    xc7 = signal.lfilter(b, a, xcdm)

    b, a = des_bw_filter_hp(cutoff=20000)
    xc8 = signal.lfilter(b, a, xcdm)

    sigs = [xc, pd.Series(xc0), pd.Series(xc1), pd.Series(xc2), pd.Series(xc3),
            pd.Series(xc4), pd.Series(xc5), pd.Series(xc6), pd.Series(xc7), pd.Series(xc8)]

    for i, sig in enumerate(sigs):
        X.loc[seg_id, 'mean_%d' % i] = sig.mean()
        X.loc[seg_id, 'std_%d' % i] = sig.std()
        X.loc[seg_id, 'max_%d' % i] = sig.max()
        X.loc[seg_id, 'min_%d' % i] = sig.min()

        X.loc[seg_id, 'mean_change_abs_%d' % i] = np.mean(np.diff(sig))
        X.loc[seg_id, 'mean_change_rate_%d' % i] = np.mean(np.nonzero((np.diff(sig) / sig[:-1]))[0])
        X.loc[seg_id, 'abs_max_%d' % i] = np.abs(sig).max()
        X.loc[seg_id, 'abs_min_%d' % i] = np.abs(sig).min()

        X.loc[seg_id, 'std_first_50000_%d' % i] = sig[:50000].std()
        X.loc[seg_id, 'std_last_50000_%d' % i] = sig[-50000:].std()
        X.loc[seg_id, 'std_first_10000_%d' % i] = sig[:10000].std()
        X.loc[seg_id, 'std_last_10000_%d' % i] = sig[-10000:].std()

        X.loc[seg_id, 'avg_first_50000_%d' % i] = sig[:50000].mean()
        X.loc[seg_id, 'avg_last_50000_%d' % i] = sig[-50000:].mean()
        X.loc[seg_id, 'avg_first_10000_%d' % i] = sig[:10000].mean()
        X.loc[seg_id, 'avg_last_10000_%d' % i] = sig[-10000:].mean()

        X.loc[seg_id, 'min_first_50000_%d' % i] = sig[:50000].min()
        X.loc[seg_id, 'min_last_50000_%d' % i] = sig[-50000:].min()
        X.loc[seg_id, 'min_first_10000_%d' % i] = sig[:10000].min()
        X.loc[seg_id, 'min_last_10000_%d' % i] = sig[-10000:].min()

        X.loc[seg_id, 'max_first_50000_%d' % i] = sig[:50000].max()
        X.loc[seg_id, 'max_last_50000_%d' % i] = sig[-50000:].max()
        X.loc[seg_id, 'max_first_10000_%d' % i] = sig[:10000].max()
        X.loc[seg_id, 'max_last_10000_%d' % i] = sig[-10000:].max()

        X.loc[seg_id, 'max_to_min_%d' % i] = sig.max() / np.abs(sig.min())
        X.loc[seg_id, 'max_to_min_diff_%d' % i] = sig.max() - np.abs(sig.min())
        X.loc[seg_id, 'count_big_%d' % i] = len(sig[np.abs(sig) > 500])
        X.loc[seg_id, 'sum_%d' % i] = sig.sum()

        X.loc[seg_id, 'mean_change_rate_first_50000_%d' % i] = np.mean(np.nonzero((np.diff(sig[:50000]) / sig[:50000][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_last_50000_%d' % i] = np.mean(np.nonzero((np.diff(sig[-50000:]) / sig[-50000:][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_first_10000_%d' % i] = np.mean(np.nonzero((np.diff(sig[:10000]) / sig[:10000][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_last_10000_%d' % i] = np.mean(np.nonzero((np.diff(sig[-10000:]) / sig[-10000:][:-1]))[0])

        X.loc[seg_id, 'q95_%d' % i] = pd.Series(sig).quantile(0.95)
        X.loc[seg_id, 'q99_%d' % i] = pd.Series(sig).quantile(0.99)
        X.loc[seg_id, 'q05_%d' % i] = pd.Series(sig).quantile(0.05)
        X.loc[seg_id, 'q01_%d' % i] = pd.Series(sig).quantile(0.01)

        X.loc[seg_id, 'abs_q95_%d' % i] = pd.Series(np.abs(sig)).quantile(0.95)
        X.loc[seg_id, 'abs_q99_%d' % i] = pd.Series(np.abs(sig)).quantile(0.99)
        X.loc[seg_id, 'abs_q05_%d' % i] = pd.Series(np.abs(sig)).quantile(0.05)
        X.loc[seg_id, 'abs_q01_%d' % i] = pd.Series(np.abs(sig)).quantile(0.01)

        X.loc[seg_id, 'trend_%d' % i] = add_trend_feature(sig)
        X.loc[seg_id, 'abs_trend_%d' % i] = add_trend_feature(sig, abs_values=True)
        X.loc[seg_id, 'abs_mean_%d' % i] = np.abs(sig).mean()
        X.loc[seg_id, 'abs_std_%d' % i] = np.abs(sig).std()

        X.loc[seg_id, 'mad_%d' % i] = sig.mad()  # 不知道
        X.loc[seg_id, 'kurt_%d' % i] = sig.kurtosis()  # 峰态系数  表征概率密度分布曲线在平均值处峰值高低的特征数
        X.loc[seg_id, 'skew_%d' % i] = sig.skew()  # 衡量信号扁平程度
        X.loc[seg_id, 'med_%d' % i] = sig.median()

        X.loc[seg_id, 'Hilbert_mean_%d' % i] = np.abs(hilbert(sig)).mean()     # 希尔伯特变换
        X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hanning(150), mode='same') / sum(hanning(150))).mean()  # 海宁窗 卷积

        X.loc[seg_id, 'classic_sta_lta1_mean_%d' % i] = classic_sta_lta(sig, 500, 10000).mean()
        X.loc[seg_id, 'classic_sta_lta2_mean_%d' % i] = classic_sta_lta(sig, 5000, 100000).mean()
        X.loc[seg_id, 'classic_sta_lta3_mean_%d' % i] = classic_sta_lta(sig, 3333, 6666).mean()
        X.loc[seg_id, 'classic_sta_lta4_mean_%d' % i] = classic_sta_lta(sig, 10000, 25000).mean()

        X.loc[seg_id, 'Moving_average_700_mean_%d' % i] = sig.rolling(window=700).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_1500_mean_%d' % i] = sig.rolling(window=1500).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_3000_mean_%d' % i] = sig.rolling(window=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_6000_mean_%d' % i] = sig.rolling(window=6000).mean().mean(skipna=True)

        ewma = pd.Series.ewm
        X.loc[seg_id, 'exp_Moving_average_300_mean_%d' % i] = ewma(sig, span=300).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_3000_mean_%d' % i] = ewma(sig, span=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_30000_mean_%d' % i] = ewma(sig, span=6000).mean().mean(skipna=True)

        no_of_std = 2
        X.loc[seg_id, 'MA_700MA_std_mean_%d' % i] = sig.rolling(window=700).std().mean()
        X.loc[seg_id, 'MA_700MA_BB_high_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_700MA_BB_low_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_std_mean_%d' % i] = sig.rolling(window=400).std().mean()
        X.loc[seg_id, 'MA_400MA_BB_high_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_BB_low_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_1000MA_std_mean_%d' % i] = sig.rolling(window=1000).std().mean()

        X.loc[seg_id, 'iqr_%d' % i] = np.subtract(*np.percentile(sig, [75, 25]))
        X.loc[seg_id, 'q999_%d' % i] = pd.Series(sig).quantile(0.999)
        X.loc[seg_id, 'q001_%d' % i] = pd.Series(sig).quantile(0.001)
        X.loc[seg_id, 'ave10_%d' % i] = stats.trim_mean(sig, 0.1)

    for windows in [10, 100, 1000]:
        x_roll_std = xc.rolling(windows).std().dropna().values
        x_roll_mean = xc.rolling(windows).mean().dropna().values

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = pd.Series(x_roll_std).quantile(0.01)
        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = pd.Series(x_roll_std).quantile(0.05)
        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = pd.Series(x_roll_std).quantile(0.95)
        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = pd.Series(x_roll_std).quantile(0.99)
        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = pd.Series(x_roll_mean).quantile(0.01)
        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = pd.Series(x_roll_mean).quantile(0.05)
        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = pd.Series(x_roll_mean).quantile(0.95)
        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = pd.Series(x_roll_mean).quantile(0.99)
        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    return X


def create_features_pk_det(seg_id, seg, X, st, end):
    try:
        X.loc[seg_id, 'seg_id'] = np.int32(seg_id)
        X.loc[seg_id, 'seg_start'] = np.int32(st)
        X.loc[seg_id, 'seg_end'] = np.int32(end)
    except:
        pass

    sig = pd.Series(seg['acoustic_data'].values)
    b, a = des_bw_filter_lp(cutoff=18000)
    sig = signal.lfilter(b, a, sig)

    peakind = []
    noise_pct = .001
    count = 0

    while len(peakind) < 12 and count < 24:
        peakind = signal.find_peaks_cwt(sig, np.arange(1, 16), noise_perc=noise_pct, min_snr=4.0)  # 在一定窗口内寻找12个峰
        noise_pct *= 2.0
        count += 1

    if len(peakind) < 12:
        print('Warning: Failed to find 12 peaks for %d' % seg_id)

    while len(peakind) < 12:
        peakind.append(149999)

    df_pk = pd.DataFrame(data={'pk': sig[peakind], 'idx': peakind}, columns=['pk', 'idx'])
    df_pk.sort_values(by='pk', ascending=False, inplace=True)

    for i in range(12):
        X.loc[seg_id, 'pk_idx_%d' % i] = df_pk['idx'].iloc[i]
        X.loc[seg_id, 'pk_val_%d' % i] = df_pk['pk'].iloc[i]

    return X


def build_fields(proc_id):
    success = 1
    count = 0
    try:
        seg_st = int(NUM_SEG_PER_PROC * proc_id)
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'raw_data_%d.csv' % proc_id), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
        len_df = len(train_df.index)
        start_indices = (np.loadtxt(fname=os.path.join(OUTPUT_DIR, 'start_indices_4k.csv'), dtype=np.int32, delimiter=','))[:, proc_id]
        # train_X = pd.DataFrame(dtype=np.float64)
        train_X_pk = pd.DataFrame(dtype=np.float64)
        # train_y = pd.DataFrame(dtype=np.float64, columns=['time_to_failure'])
        t0 = time.time()

        for seg_id, start_idx in zip(range(seg_st, seg_st + NUM_SEG_PER_PROC), start_indices):
            end_idx = np.int32(start_idx + 150000)
            print('working: %d, %d, %d to %d of %d' % (proc_id, seg_id, start_idx, end_idx, len_df))
            seg = train_df.iloc[start_idx: end_idx]
            # train_X = create_features(seg_id, seg, train_X, start_idx, end_idx)
            train_X_pk = create_features_pk_det(seg_id, seg, train_X_pk, start_idx, end_idx)
            # train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]

            if count == 1:
                print('saving: %d, %d to %d' % (seg_id, start_idx, end_idx))
                # train_X.to_csv('train_x_%d.csv' % proc_id, index=False)
                # train_y.to_csv('train_y_%d.csv' % proc_id, index=False)
                train_X_pk.to_csv('train_x_%d_pk.csv' % proc_id, index=False)

            count += 1

        print('final_save, process id: %d, loop time: %.2f for %d iterations' % (proc_id, time.time() - t0, count))
        # train_X.to_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % proc_id), index=False)
        # train_y.to_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % proc_id), index=False)
        train_X_pk.to_csv(os.path.join(OUTPUT_DIR, 'train_x_%d_pk.csv' % proc_id), index=False)

    except:
        print(traceback.format_exc())
        success = 0

    return success  # 1 on success, 0 if fail


def run_mp_build():
    t0 = time.time()
    num_proc = NUM_THREADS
    pool = mp.Pool(processes=num_proc)
    results = [pool.apply_async(build_fields, args=(pid, )) for pid in range(NUM_THREADS)]
    output = [p.get() for p in results]
    num_built = sum(output)
    pool.close()
    pool.join()
    print(num_built)
    print('Run time: %.2f' % (time.time() - t0))


def join_mp_build():
    df0 = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % 0))
    df1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % 0))
    # df2 = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_%d_pk.csv' % 0))

    for i in range(1, NUM_THREADS):
        print('working %d' % i)
        temp = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % i))
        df0 = df0.append(temp)

        temp = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % i))
        df1 = df1.append(temp)

        # temp = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_%d_pk.csv' % i))
        # df2 = df2.append(temp)

    df0.to_csv(os.path.join(OUTPUT_DIR, 'train_x.csv'), index=False)
    df1.to_csv(os.path.join(OUTPUT_DIR, 'train_y.csv'), index=False)
    # df2.to_csv(os.path.join(OUTPUT_DIR, 'train_x_pk.csv'), index=False)


def build_test_fields():
    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x.csv'))
    train_X_pk = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_pk.csv'))
    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
        train_X_pk.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass

    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)
    test_X_pk = pd.DataFrame(columns=train_X_pk.columns, dtype=np.float64, index=submission.index)

    print('start for loop')
    count = 0
    for seg_id in tqdm(test_X.index):  # just tqdm in IDE
        seg = pd.read_csv(os.path.join(DATA_DIR, 'test', str(seg_id) + '.csv'))
        test_X_pk = create_features_pk_det(seg_id, seg, test_X_pk, 0, 0)  #start_idx, end_idx
        test_X = create_features(seg_id, seg, test_X, 0, 0)

        if count % 100 == 0:
            print('working', seg_id)
        count += 1

    test_X.to_csv(os.path.join(OUTPUT_DIR, 'test_x.csv'), index=False)
    test_X_pk.to_csv(os.path.join(OUTPUT_DIR, 'test_x_pk.csv'), index=False)


def scale_fields(fn_train='train_x_new.csv', fn_test='test_x_new.csv',
                 fn_out_train='scaled_train_X_new.csv', fn_out_test='scaled_test_X_new.csv'):
    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_train))
    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass
    test_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_test))

    print('start scaler')
    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    scaled_train_X.to_csv(os.path.join(OUTPUT_DIR, fn_out_train), index=False)
    scaled_test_X.to_csv(os.path.join(OUTPUT_DIR, fn_out_test), index=False)



params = {'num_leaves': 21,  #21,
         'min_data_in_leaf': 20,
         'objective':'huber',
         'learning_rate': 0.05,
         'max_depth': 6, #108,
         "boosting": "gbdt",
         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.91,
         "bagging_seed": 42,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 42}


def lgb_base_model():
    maes = []
    rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    scaled_train_X = pd.read_csv('./result/new_scaled_train_X.csv')
    scaled_test_X = pd.read_csv('./result/new_scaled_test_X.csv')
    train_y = pd.read_csv('./result/train_y.csv')
    predictions = np.zeros(len(scaled_test_X))

    oof = np.zeros(len(scaled_train_X))


    pcol = []
    pcor = []
    pval = []
    y = train_y['time_to_failure'].values
    for col in scaled_train_X.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
        pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))
    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True, ascending = False)
    df.dropna(inplace = True)
    df = df.iloc[: 500]

    drop_cols = []
    for col in scaled_train_X.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

    print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)

    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = scaled_train_X.columns

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]
        y_tr = y_tr['time_to_failure']
        y_val = y_val['time_to_failure']

        model = lgb.LGBMRegressor(**params, n_estimators=30000, n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
                  verbose=1000, early_stopping_rounds=200)  #

        # predictions
        preds = model.predict(scaled_test_X, num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        preds = model.predict(X_val, num_iteration=model.best_iteration_)

        oof[val_idx] = preds.reshape(-1,)


        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        fold_importance_df['importance_%d' % fold_] = model.feature_importances_[:len(scaled_train_X.columns)]

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    submission.time_to_failure = predictions
    submission.to_csv('submission_lgb.csv')

    return oof, predictions


def xgb_base_model():
    maes = []
    rmses = []
    tr_maes = []
    tr_rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')

    scaled_train_X = pd.read_csv('./result/scaled_train_X.csv')
    scaled_test_X = pd.read_csv('./result/scaled_test_X.csv')
    train_y = pd.read_csv('./result/train_y.csv')

    oof = np.zeros(len(scaled_train_X))

    pcol = []
    pcor = []
    pval = []
    y = train_y['time_to_failure'].values
    for col in scaled_train_X.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
        pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))
    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True, ascending = False)
    df.dropna(inplace = True)
    df = df.iloc[: 500]

    drop_cols = []
    for col in scaled_train_X.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

    predictions = np.zeros(len(scaled_test_X))
    preds_train = np.zeros(len(scaled_train_X))

    print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)

    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)
        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        model = xgb.XGBRegressor(n_estimators=5000,
                                learning_rate=0.1,
                                max_depth=6,  #12
                                subsample=0.8,
                                reg_lambda=1.0, # seems best within 0.5 of 2.0
                                eval_metric='mae',
                                # gamma=1,
                                random_state=777+fold_,
                                n_jobs=-1,
                                verbosity=2)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],early_stopping_rounds=200)


        # predictions
        preds = model.predict(scaled_test_X)  #, num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        # preds = model.predict(scaled_train_X)  #, num_iteration=model.best_iteration_)
        # preds_train += preds / folds.n_splits
        preds = model.predict(X_val)  #, num_iteration=model.best_iteration_)

        oof[val_idx] = preds.reshape(-1,)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)
        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)
        # training for over fit
        preds = model.predict(X_tr)  #, num_iteration=model.best_iteration_)
        mae = mean_absolute_error(y_tr, preds)
        print('Tr MAE: %.6f' % mae)
        tr_maes.append(mae)
        rmse = mean_squared_error(y_tr, preds)
        print('Tr RMSE: %.6f' % rmse)
        tr_rmses.append(rmse)
    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))
    print('Tr MAEs', tr_maes)
    print('Tr MAE mean: %.6f' % np.mean(tr_maes))
    print('Tr RMSEs', rmses)
    print('Tr RMSE mean: %.6f' % np.mean(tr_rmses))

    submission['time_to_failure'] = predictions
    submission.to_csv('submission_xgb.csv')  # index needed, it is seg id

    return oof, predictions

    # plt.figure(figsize=(18, 8))
    # plt.subplot(2, 3, 1)
    # plt.plot(y_tr[:1000], color='g', label='y_train')
    # plt.plot(oof[:1000], color='b', label='lgb')
    # plt.legend(loc=(1, 0.5))
    # plt.title('lgb')
    # plt.show()


def cat_base_model():
    maes = []
    rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    scaled_train_X = pd.read_csv('./result/scaled_train_X.csv')
    scaled_test_X = pd.read_csv('./result/scaled_test_X.csv')
    train_y = pd.read_csv('./result/train_y.csv')
    predictions = np.zeros(len(scaled_test_X))


    oof = np.zeros(len(scaled_train_X))


    pcol = []
    pcor = []
    pval = []
    y = train_y['time_to_failure'].values
    for col in scaled_train_X.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
        pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))
    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True, ascending = False)
    df.dropna(inplace = True)
    df = df.iloc[: 500]

    drop_cols = []
    for col in scaled_train_X.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

    predictions = np.zeros(len(scaled_test_X))
    preds_train = np.zeros(len(scaled_train_X))

    print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)


    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = scaled_train_X.columns

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]
        y_tr = y_tr['time_to_failure']
        y_val = y_val['time_to_failure']

        model = CatBoostRegressor(iterations=10000,
                                  learning_rate=0.3,
                                  depth=4,
                                  loss_function='MAE',
                                  random_state=42,
                                  eval_metric='MAE')
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], cat_features=[], use_best_model=True)

        # predictions
        preds = model.predict(scaled_test_X)
        predictions += preds / folds.n_splits
        preds = model.predict(X_val)

        oof[val_idx] = preds.reshape(-1,)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        # fold_importance_df['importance_%d' % fold_] = model.feature_importances_[:len(scaled_train_X.columns)]

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    submission['time_to_failure'] = predictions
    submission.to_csv('submission_cat.csv')
    return oof, predictions


def svr_base_model():
    maes = []
    rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    scaled_train_X = pd.read_csv('./result/scaled_train_X.csv')
    scaled_test_X = pd.read_csv('./result/scaled_test_X.csv')
    train_y = pd.read_csv('./result/train_y.csv')
    predictions = np.zeros(len(scaled_test_X))

    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = scaled_train_X.columns

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]
        y_tr = y_tr['time_to_failure']
        y_val = y_val['time_to_failure']

        model = SVR(kernel='rbf')  # , nu=0.9, C=1.0, tol=0.01
        model.fit(X_tr, y_tr)

        preds = model.predict(scaled_test_X)
        predictions += preds / folds.n_splits
        preds = model.predict(X_val)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        # fold_importance_df['importance_%d' % fold_] = model.feature_importances_[:len(scaled_train_X.columns)]

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    submission['time_to_failure'] = predictions
    submission.to_csv('submission_svr_8.csv')
    return predictions


def krr_base_model():
    maes = []
    rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    scaled_train_X = pd.read_csv('./result/scaled_train_X.csv')
    scaled_test_X = pd.read_csv('./result/scaled_test_X.csv')
    train_y = pd.read_csv('./result/train_y.csv')
    predictions = np.zeros(len(scaled_test_X))

    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = scaled_train_X.columns

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]
        y_tr = y_tr['time_to_failure']
        y_val = y_val['time_to_failure']

        model = KernelRidge(kernel='rbf', alpha=0.001, gamma=0.001)
        model.fit(X_tr, y_tr)

        # predictions
        preds = model.predict(scaled_test_X)
        predictions += preds / folds.n_splits
        preds = model.predict(X_val)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        # fold_importance_df['importance_%d' % fold_] = model.feature_importances_[:len(scaled_train_X.columns)]

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    submission['time_to_failure'] = predictions
    submission.to_csv('submission_krr_8.csv')
    return predictions


def rf_base_model():
    maes = []
    rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    scaled_train_X = pd.read_csv('./result/scaled_train_X.csv')
    scaled_test_X = pd.read_csv('./result/scaled_test_X.csv')
    train_y = pd.read_csv('./result/train_y.csv')
    predictions = np.zeros(len(scaled_test_X))

    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = scaled_train_X.columns

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]
        y_tr = y_tr['time_to_failure']
        y_val = y_val['time_to_failure']

        model = RandomForestRegressor(n_estimators = 1000,#决策树的数量
                                      criterion = 'mae',
                                      max_depth = 6  ,#最大深度
                                      #min_samples_split =5,#内部节点在划分的最小样本数
                                      #min_samples_leaf = 5,#叶子节点最小样本数，如果叶子节点数量少，将会被剪枝
                                      max_features = "log2",
                                      verbose =2,
                                      n_jobs=6)

        model.fit(X_tr, y_tr)

        # predictions
        preds = model.predict(scaled_test_X)
        predictions += preds / folds.n_splits
        preds = model.predict(X_val)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        # fold_importance_df['importance_%d' % fold_] = model.feature_importances_[:len(scaled_train_X.columns)]

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    submission['time_to_failure'] = predictions
    submission.to_csv('submission_rf_8.csv')
    return predictions


def stack_base_model(scaled_train_X, scaled_test_X):
    maes = []
    rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')

    train_y = pd.read_csv('./result/train_y.csv')
    predictions = np.zeros(len(scaled_test_X))

    oof = np.zeros(len(scaled_train_X))

    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]
        y_tr = y_tr['time_to_failure']
        y_val = y_val['time_to_failure']

        model = lgb.LGBMRegressor(**params, n_estimators=30000, n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
                  verbose=1000, early_stopping_rounds=200)  #

        # predictions
        preds = model.predict(scaled_test_X)
        predictions += preds / folds.n_splits
        preds = model.predict(X_val)

        oof[val_idx] = preds.reshape(-1,)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        # fold_importance_df['importance_%d' % fold_] = model.feature_importances_[:len(scaled_train_X.columns)]

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    submission['time_to_failure'] = predictions
    submission.to_csv('submission_stack.csv')
    return oof, predictions


# do this in the IDE, call the function above
if __name__ == "__main__":

    oof_lgb, prediction_lgb = lgb_base_model()
    oof_xgb, prediction_xgb = xgb_base_model()
    oof_cat, prediction_cat = cat_base_model()


    train_stack = np.vstack([oof_lgb, oof_xgb, oof_cat]).transpose()

    train_stack = pd.DataFrame(train_stack, columns=['lgb', 'xgb', 'cat'])
    test_stack = np.vstack([prediction_lgb, prediction_xgb, prediction_cat]).transpose()
    test_stack = pd.DataFrame(test_stack)

    oof_lgb_stack, prediction_lgb_stack = stack_base_model(train_stack, test_stack)


    submission = pd.read_csv('./input/sample_submission.csv', index_col='seg_id')

    submission['time_to_failure'] = (prediction_lgb+prediction_xgb+prediction_cat+prediction_lgb_stack)/4
    submission.to_csv('submission_mean.csv')
