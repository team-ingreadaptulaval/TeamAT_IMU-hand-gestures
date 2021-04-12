import numpy as np
import pandas as pd
import pickle as pkl
import os
from scipy import stats, signal, linalg
from statsmodels.robust import scale
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import confusion_matrix
from tslearn.metrics import dtw, gak
from sklearn.naive_bayes import GaussianNB
from time import time
# from dimensionality import *
# import utils
from .dimensionality import *
from . import utils


class ImuSignDetectClassifier:

    def __init__(self):
        self.train_signals = []
        self.train_targets = np.array([])
        self.test_signals = []
        self.features_names = []
        self.clf = SVCWithReject(_lambda=0.50, kernel='linear', C=0.1)
        self.train_features = np.array([], dtype='float64')
        self.has_no_class = True


    def fit(self, signals, targets):
        """
        :type targets: np.array
        """
        print(f'targets: {targets}, len: {len(np.unique(targets))}')
        if len(np.unique(targets)) == 0:
            print('NO CLASSES')
            self.clf = ZeroClassClassifier()
            self.clf.fit(None, None)
            print('fit done')
            self.has_no_class = True
        else:
            self.has_no_class = False
            if len(np.unique(targets)) == 1:
                print('ONE CLASS SVM')
                self.clf = OneClassClassifier()
            else:
                print('SVM')
                self.clf = SVCWithReject(_lambda=0.50, kernel='linear', C=0.1)
            self.train_signals = signals
            self.__process_signals(self.train_signals)
            self.train_features = self.__compute_features(self.train_signals)
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(self.train_features)
            self.train_features = self.scaler.transform(self.train_features)
            self.reducer = PCAReducer(n_components=int(len(targets)))
            self.reducer.fit(self.train_features, targets)
            print('PCA comp out: ', self.reducer.reducer.n_components_)
            reduced_train_feats = self.reducer.transform(self.train_features)
            self.clf.fit(reduced_train_feats, targets)


    def __process_signals(self, signals):
        # No special processing, remove this method eventually
        for i, (accel, gyro) in enumerate(zip(signals[0:3], signals[3::])):
            signals[i] = accel  # np.sign(accel - np.mean(accel, axis=1)[:, None]) * accel**2 + np.mean(accel, axis=1)[:, None]
            signals[i+3] = gyro

    def __compute_features(self, signals, magnitude_only=True):
        feats = []
        if magnitude_only:
            try:
                accel0 = linalg.norm(np.dstack(signals[0:3]), axis=2)
                gyro0 = linalg.norm(np.dstack(signals[3::]), axis=2)
            except ValueError:
                nanpos = np.isnan(signals[0])
                accel0 = np.sqrt(np.nansum(np.dstack(signals[0:3]) ** 2, axis=2))
                gyro0 = np.sqrt(np.nansum(np.dstack(signals[3::]) ** 2, axis=2))
                accel0[nanpos] = np.nan
                gyro0[nanpos] = np.nan
            accel = utils.lowpass_filter(accel0, tau=0.03, freq=100)
            gyro = utils.lowpass_filter(gyro0, tau=0.015, freq=100)
            signals = [accel, gyro]
            p = 3
            n_cut = 4
            feats += [utils.nancorrelation(*signals).reshape(-1)]
            feats += [utils.nancorrelation(acc_cut, gyr_cut).reshape(-1)
                      for acc_cut, gyr_cut in zip(utils.make_cuts(signals[0], n_cut),utils.make_cuts(signals[1], n_cut))]
        else:
            n_cut = 4
            p = 4
            feats += [scale.mad(s, axis=1) for s in signals]
            feats += [utils.sma(*signals[0:3]), utils.sma(*signals[3::])]
            feats += [utils.correlation(*signals[0:3]).T, utils.correlation(*signals[3::]).T]
        if not self.features_names:
            self.generate_feature_names(magnitude_only, p, n_cut)
        feats += [np.nanmean(s, axis=1, dtype='float64') for s in signals]
        feats += [np.nanmedian(s, axis=1) for s in signals]
        feats += [np.nanmax(s, axis=1) for s in signals]
        feats += [np.nanmin(s, axis=1) for s in signals]
        feats += [np.nanstd(s, axis=1) for s in signals]
        feats += [stats.iqr(s, axis=1, nan_policy='omit') for s in signals]
        feats += [np.nansum(s ** 2, axis=1) for s in signals]
        feats += [np.nanmean(a_cut, axis=1) for s in signals for a_cut in utils.make_cuts(s, n_cut)]
        feats += [np.nanmedian(a_cut, axis=1) for s in signals for a_cut in utils.make_cuts(s, n_cut)]
        feats += [np.nanmax(a_cut, axis=1) for s in signals for a_cut in utils.make_cuts(s, n_cut)]
        feats += [np.nanmin(a_cut, axis=1) for s in signals for a_cut in utils.make_cuts(s, n_cut)]
        feats += [stats.iqr(a_cut, axis=1, nan_policy='omit') for s in signals for a_cut in utils.make_cuts(s, n_cut)]
        feats += [np.nansum(a_cut ** 2, axis=1) for s in signals for a_cut in utils.make_cuts(s, n_cut)]
        ffts = [utils._fft(s, 0.01 * s.shape[1], handle_nans=True) for s in signals]
        feats += [utils.fft_metric(fft[0], fft[1], np.nanmedian, handle_nans=True) for fft in ffts]
        feats += [utils.fft_metric(fft[0], fft[1], np.nanmax, handle_nans=True) for fft in ffts]
        feats += [utils.fft_energy(fft[1]) for fft in ffts]
        feats += [stats.kurtosis(s, axis=1, nan_policy='omit') for s in signals]
        feats += [utils.ar_coeff(s, p).T for s in signals]
        feats = np.vstack(feats).T
        return feats

    def generate_feature_names(self, magnitude_only, p, n_cut):
        if magnitude_only:
            sensor = ['accel', 'gyro']
            axis = ['mag']
            arcoef_id = [str(i + 1) for i in range(p)]
            splits = ''.join([str(c + 1) for c in range(n_cut)])
            self.features_names += ['correlation_acc_gyr_full']
            self.features_names += ['correlation_acc_gyr' + '_' + part  for part in splits]
            self.features_names += ['mean' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['median' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['max' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['min' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['std' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['irq' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['energy' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['mean_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['median_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['max_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['min_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['iqr_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['energy_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['fmed' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['fmax' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['fsum' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['kurtosis' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['ar_coeff_' + arid + '_' + ss + '_' + ax for ss in sensor for ax in axis for arid in arcoef_id]
        else:
            # TODO: define
            pass

    def decision_function(self, signals):
        self.test_signals = signals
        self.__process_signals(self.test_signals)
        X = self.__compute_features(self.test_signals)
        X = self.reducer.transform(X)
        h = self.clf.decision_function(X)
        return h/np.linalg.norm(h, axis=1).reshape(h.shape[0], 1)

    def predict(self, signals, with_second_choice=False):
        if self.has_no_class:
            return [-1]
        self.test_signals = signals
        self.__process_signals(self.test_signals)
        X = self.__compute_features(self.test_signals)
        X = self.scaler.transform(X)
        X = self.reducer.transform(X)
        return self.clf.predict(X)

    def score(self, signals, targets):
        if self.has_no_class:
            return 0
        self.test_signals = signals
        self.__process_signals(self.test_signals)
        X = self.__compute_features(self.test_signals)
        X = self.scaler.transform(X)
        # vars = np.var(X, axis=0)
        # vars = np.argsort(vars)
        # print(vars[-10::])
        X = self.reducer.transform(X)
        return self.clf.score(X, targets)

class SVCWithReject:

    def __init__(self, _lambda, kernel, C, gamma='scale'):
        self.clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self._lambda = _lambda
        self.K = 0

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.K = len(set(y)) # NO

    def predict_proba(self, X):
         return self.clf.predict_proba(X)

    def decision_function(self, X):
        h = self.clf.decision_function(X)
        return h/np.linalg.norm(h, axis=1).reshape(h.shape[0], 1)

    def predict(self, X, tol_rel=0, tol_abs=0, with_second_choice=False):  #0.05, 0.4 /// 0 0.5  // , tol_rel=0, tol_abs=0, with_second_choice=with_second_choice
        if not with_second_choice:
            posteriors = self.predict_proba(X)
            maxs = np.sort(posteriors, axis=1)[:, ::-1][:, 0:2]
            confidence = maxs[:, 0] -  maxs[:, 1]
            preds = np.argmax(posteriors, axis=1)
            print(preds, maxs[0], confidence, posteriors)
            preds[confidence <= tol_rel] = self.K
            preds[np.max(posteriors, axis=1) <= tol_abs] = self.K
            return self.clf.classes_[preds]
        else:
            posteriors = self.predict_proba(X)
            maxs = np.sort(posteriors, axis=1)[:, ::-1][:, 0:2]
            confidence = maxs[:, 0] - maxs[:, 1]
            preds = np.argsort(posteriors, axis=1)[:, ::-1]
            preds[confidence <= tol_rel, 0] = self.K
            preds[np.max(posteriors, axis=1) <= tol_abs, 0] = self.K
            return_list = [(fpred, fprob, spred, sprob) for fpred, fprob, spred, sprob in zip(
                self.clf.classes_[preds[:, 0]], maxs[:, 0], self.clf.classes_[preds[:, 1]], maxs[:, 1])]
            return return_list if len(return_list) > 1 else return_list[0]

    def score(self, X, y):
        preds = self.predict(X)
        diff = np.array(y) - preds
        miss_w_rej = len(diff[diff != 0])  # ratés incluant rejet
        rej = len(preds[preds == self.K])  # rejet
        return 1 - (((miss_w_rej - rej) * 1 + rej * self._lambda) / len(y))


class ZeroClassClassifier:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return [-1]

    def score(self, X, y):
        return 1


class OneClassClassifier(OneClassSVM):

    def __init__(self):
        super().__init__(kernel='linear')
        self.target = None

    def fit(self, X, y=None, sample_weight=None, **params):
        super().fit(X)
        targets = set(y)
        self.target = targets.pop()

    def predict(self, X):
        preds = super().predict(X)
        preds = [self.target if p == 1 else -1 for p in preds]
        return preds

    def score(self, X, y):
        return 1
