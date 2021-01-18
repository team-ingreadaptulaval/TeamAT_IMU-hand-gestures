import numpy as np
import pandas as pd
import pickle as pkl
import os
from scipy import stats, signal, special
from statsmodels.robust import scale
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from time import time
from tslearn.metrics import dtw
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf, pacf
import glob
from itertools import chain


def sma(x, y, z):
    X = np.sum(np.abs(x), axis=1)
    Y = np.sum(np.abs(y), axis=1)
    Z = np.sum(np.abs(z), axis=1)
    return np.sum((X, Y, Z), axis=0)

def nancorrelation(*kargs):
    nan_pos = find_first_nan(kargs[0])
    C = np.zeros((kargs[0].shape[0], special.comb(len(kargs), 2, exact=True)))
    for i in range(kargs[0].shape[0]):
        c = np.corrcoef(np.vstack([karg[i, 0:nan_pos[i]] for karg in kargs]))
        C[i, :] = c[np.triu_indices(len(kargs), k=1)]
    return C

def correlation(x, y, z):
    C = np.zeros((x.shape[0], 3))
    for i in range(x.shape[0]):
        c = np.corrcoef(np.vstack([x[i, :], y[i, :], z[i, :]]))
        C[i, :] = np.array([c[0, 1], c[0, 2], c[1, 2]])
    return C

def _fft(y, L, handle_nans=False, acq_freq=100, plotit=False):
    if handle_nans and len(y.shape) != 1:
        freqs = []
        fft_theos = []
        nan_pos = find_first_nan(y)
        for row, npos in zip(y, nan_pos):
            n = npos
            L = n/acq_freq
            freq = np.fft.fftfreq(n, L/n)
            fft_vals = np.fft.fft(row[0:npos])
            fft_theo = 2 * np.abs(fft_vals / n)
            mask = freq > 0
            freqs.append(freq[mask])
            fft_theos.append(fft_theo[mask])
        return pd.DataFrame(freqs).values, pd.DataFrame(fft_theos).values

    if np.sum(np.isnan(y)) > 0:
        raise ValueError('fft contains NaNs')
    n = y.shape[1]
    freqs = np.fft.fftfreq(n, L/n)
    fft_vals = np.fft.fft(y)
    fft_theo = 2 * np.abs(fft_vals / n)
    mask = freqs > 0
    freqs = freqs[mask]
    fft_theo = fft_theo[:, mask]
    if plotit:
        fig, axs = plt.subplots(1, 2)
        tt = 10
        axs[0].plot(np.linspace(0, L, y.shape[1]), y[tt, :])
        axs[1].plot(freqs, fft_theo.T[:, tt])
        plt.show()
    return freqs, fft_theo

def fft_metric(freqs, fft_theo, metric, handle_nans=False, plotit=False):
    if handle_nans and len(fft_theo.shape) != 1:
        frq_metric = []
        nan_pos = find_first_nan(fft_theo)
        for frq, fft, npos in zip(freqs, fft_theo, nan_pos):
            idx = (np.abs((fft[0:npos] - metric(fft)))).argmin()
            frq_metric.append(frq[idx])
        return np.array(frq_metric)
    if np.sum(np.isnan(fft_theo)) > 0:
        raise ValueError('fft_metric contains NaNs')
    idx = (np.abs((fft_theo.T - metric(fft_theo, axis=1)).T)).argmin(axis=1)
    frq_metric = freqs[idx]
    if plotit:
        plt.figure()
        plt.plot(frq_metric, '.b', markersize=2)
        plt.show()
    return frq_metric

def fft_energy(fft_theo):
    peak = np.nanmax(fft_theo, axis=1)
    integral = np.nansum(fft_theo, axis=1)
    return peak / integral

def lowpass_filter(data, tau, freq):
    ts = 1/freq
    if len(data.shape) == 1:
        filtered_data = np.zeros(data.shape)
        filtered_data[0] = data[0]
        for i in range(1, data.shape[0]):
            filtered_data[i] = filtered_data[i - 1] * (1 - ts/tau) + data[i] * ts/tau
        return filtered_data
    else:
        filtered_data = np.zeros(data.shape)
        filtered_data[:, 0] = data[:, 0]
        for i in range(1, data.shape[1]):
            filtered_data[:, i] = filtered_data[:, i - 1] * (1 - ts/tau) + data[:, i] * ts/tau
        return filtered_data

def highpass_filter(data, tau, freq):
    pass

def make_cuts(signal, n_cuts):
    cuts = []
    nan_pos = find_first_nan(signal)
    cuts_pos = np.vstack([i * (nan_pos // n_cuts) for i in range(n_cuts)]).T
    for i in range(n_cuts):
        try:
            longest_cut = np.max(cuts_pos[:, i+1] - cuts_pos[:, i])
            next_cut = cuts_pos[:, i+1]
        except IndexError:
            longest_cut = np.max(nan_pos - cuts_pos[:, i])
            next_cut = nan_pos
        this_cut = np.vstack([
            np.hstack([row[cuts_pos[j, i]:next_cut[j]],
                       np.full((longest_cut-(next_cut[j] - cuts_pos[j, i])), np.nan)])
            for j, row in enumerate(signal)
        ])
        yield this_cut

def find_first_nan(array):
    table = pd.DataFrame(array)
    nan_pos = table.isna().idxmax(1).where(table.isna().any(1)).values
    nan_pos[np.isnan(nan_pos)] = array.shape[1]
    return  nan_pos.astype(int)

def std_scale(data):
    return  (data - np.mean(data)) / np.std(data)

# def load_IMU_data_other(file):
#     with open(file, 'rb') as f:
#         seq_bin = f.read()
#         data = pkl.loads(seq_bin)
#     signals = [[], [], [], [], [], []]
#     targets = []
#     target_names = []
#     for key, value in data.items():
#         target_names.append(key)
#         for i in range(6):
#             signals[i].append(value[1][i].values)
#         set_len, series_len = value[1][0].values.shape
#         targets.append(value[0] * np.ones((set_len, 1), dtype=int))
#     s_len = [s.shape[1] for s in signals[0]]
#     if all(x == s_len[0] for x in s_len):
#         signals = [np.vstack(s) for s in signals]
#     else:
#         max_len = max(s_len)
#         for ids, s in enumerate(signals):
#             for idc, c in enumerate(s):
#                 if c.shape[1] != max_len:
#                     signals[ids][idc] = np.hstack([c, np.full((c.shape[0], max_len - c.shape[1]), np.nan)])
#         signals = [np.vstack(s) for s in signals]
#     targets = np.vstack(targets).reshape(-1)
#     return signals, targets, target_names

def load_imu_data(file='../my_data/IMU_data.pkl'):
    with open(file, 'rb') as f:
        seq_bin = f.read()
        data = pkl.loads(seq_bin)
    signals = [[], [], [], [], [], []]
    targets = []
    target_names = []
    if not data:
        return signals, targets, target_names
    for key, value in data.items():
        target_names.append(key)
        for i in range(6):
            signals[i].append(value[1][i].values)
        set_len, series_len = value[1][0].values.shape
        targets.append(value[0] * np.ones((set_len, 1), dtype=int))
    s_len = [s.shape[1] for s in signals[0]]
    if all(x == s_len[0] for x in s_len):
        signals = [np.vstack(s) for s in signals]
    else:
        max_len = max(s_len)
        for ids, s in enumerate(signals):
            for idc, c in enumerate(s):
                if c.shape[1] != max_len:
                    signals[ids][idc] = np.hstack([c, np.full((c.shape[0], max_len - c.shape[1]), np.nan)])
        signals = [np.vstack(s) for s in signals]
    targets = np.vstack(targets).reshape(-1)
    return signals, targets, target_names

def make_file_from_csvs(out_filename, dir_path):
    filenames = glob.glob(dir_path + "/*.csv")
    data = []
    for i, filename in enumerate(filenames):
        file = filename.split('\\')[-1]
        name = file.split('.')[-2]
        info = name.split('_')
        label = info[-2]
        target_name = info[-1]
        this_data = pd.read_csv(filename, header=None).values
        data.append([target_name, label, this_data.tolist()])
    targets = list(set([d[0] for d in data]))
    data_dict = {t: [it, []] for it, t in enumerate(targets)}
    for d in data:
        data_dict[d[0]][1].append(d[2])
    for key, vals in data_dict.items():
        signals = [[], [], [], [], [], []]
        for recording in vals[1]:
            for s in range(6):
                signals[s].append(recording[s])
        data_dict[key][1] = [pd.DataFrame(signals[s]) for s in range(6)]
    with open(out_filename, 'wb') as f:
        f.write(pkl.dumps(data_dict))

def append_train_file(current_file_name, new_examples, new_target_name):
    with open(current_file_name, 'rb') as f:
        seq_bin = f.read()
        current_data = pkl.loads(seq_bin)
    if current_data.get(new_target_name) is not None:
        print('overwriting sing ' + new_target_name)
    print(current_data, type(current_data))
    if current_data:
        existing_targets = [val[0] for val in current_data.values()]
        new_target = 0
        for i in range(max(existing_targets) + 2):
            if i not in existing_targets:
                new_target = i
                break
    else:
        new_target = 0
    new_signals = [[], [], [], [], [], []]
    for ex in new_examples:
        for i in range(6):
            new_signals[i].append(ex[i, :])
    new_signals = [pd.DataFrame(new_signals[s]) for s in range(6)]
    current_data[new_target_name] = [new_target, new_signals]
    with open(current_file_name, 'wb') as f:
        f.write(pkl.dumps(current_data))

def delete_from_train_file(current_file_name, sign_to_del):
    with open(current_file_name, 'rb') as f:
        seq_bin = f.read()
        current_data = pkl.loads(seq_bin)
    current_data.pop(sign_to_del)
    with open(current_file_name, 'wb') as f:
        f.write(pkl.dumps(current_data))

def split_signals(signals, targets, train_size):
    signals_by_class = [[s[targets == i] for i in range(len(set(targets)))] for s in signals]
    train_signals = [[], [], [], [], [], []]
    test_signals = [[], [], [], [], [], []]
    test_targets = []
    for c in range(len(set(targets))):
        i_sample = np.random.choice(np.arange(signals_by_class[0][c].shape[0], dtype=int), train_size, replace=False)
        test_targets.append((signals_by_class[0][c].shape[0] - train_size) * [c])
        for i_signal, s in enumerate(signals_by_class):
            train_signals[i_signal].append(s[c][i_sample, :])
            test_signals[i_signal].append(np.delete(s[c], i_sample, axis=0))
    train_signals = [np.vstack(ts) for ts in train_signals]
    test_signals = [np.vstack(ts) for ts in test_signals]
    train_targets = np.array([train_size * [i] for i in range(len(set(targets)))]).reshape(-1)
    test_targets = np.hstack(test_targets)
    return train_signals, test_signals, train_targets, test_targets

def ar_coeff(series, p=2):
    phi = np.zeros((len(series), p))
    nan_pos = find_first_nan(series)
    for s, (signal, npos) in enumerate(zip(series, nan_pos)):
        r = acf(signal[0:npos], nlags=p)
        r1 = r[1::]
        r2 = r[0:-1]
        R = np.zeros((p, p))
        for k in range(p):
            R[k, :] = np.array([r2[int(np.abs(k - i))] for i in range(p)])
        phi[s, :] = np.linalg.solve(R, r1.reshape(r1.shape[0], 1)).reshape(-1)
    return phi

def dictfind_duplicate(dict0, val_to_compare=None):
    rev_dict = {}
    for key, value in dict0.items():
        rev_dict.setdefault(str(value), set()).add(key)
    result = list(chain.from_iterable(
        values for key, values in rev_dict.items()
        if len(values) > 1 and (key == str(val_to_compare) or val_to_compare is None)))
    return result