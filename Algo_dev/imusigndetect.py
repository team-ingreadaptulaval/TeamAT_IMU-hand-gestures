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
        # self.clf = SVC(gamma='scale', C=10)
        # self.clf = SVC(kernel='linear', C=0.1)
        self.clf = SVCWithReject(_lambda=0.50, kernel='linear', C=0.1)
        self.train_features = np.array([], dtype='float64')
        # self.reducer = NoReductor()
        # self.reducer = SelectorReducer(components=[6, 47, 50, 34, 54, 52, 49, 9, 37, 18, 28, 42]) #, 0, 23, 5, 55, 45, 39, 46, 16])
        # self.reducer = PCAReducer(n_components=20)
        # self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # self.reducer = RandomForestReducer(n_components=25//2)

    def fit(self, signals, targets):
        """

        :type targets: np.array
        """
        self.train_signals = signals
        self.__process_signals(self.train_signals)
        self.train_features = self.__compute_features(self.train_signals, info=targets)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(self.train_features)
        self.train_features = self.scaler.transform(self.train_features)
        self.reducer = PCAReducer(n_components=int(len(targets)))
        self.reducer.fit(self.train_features, targets)
        print('PCA comp out: ', self.reducer.reducer.n_components_)
        reduced_train_feats = self.reducer.transform(self.train_features)
        # clf = SVC(gamma='scale')
        # distribution = {'kernel': ['rbf', 'poly', 'sigmoid'], 'C': np.logspace(-10, 10, 11)}
        # rds = RandomizedSearchCV(clf, distribution, random_state=0, cv=3, n_jobs=-1)
        # search = rds.fit(self.train_features, targets)
        # self.best_params = search.best_params_
        # print(self.best_params)
        # cv_results = search.cv_results_
        # self.clf = SVC(kernel=self.best_params['kernel'], gamma='scale', C=self.best_params['C'])
        if len(np.unique(targets)) != 1:
            self.clf.fit(reduced_train_feats, targets)
        else:
            raise AssertionError('Only one class for svc fit')

    def __process_signals(self, signals):
        # if len(signals[0].shape) == 1:
        #     signals = [s.reshape(1, -1) for s in signals]
        for i, (accel, gyro) in enumerate(zip(signals[0:3], signals[3::])):
            signals[i] = accel  # np.sign(accel - np.mean(accel, axis=1)[:, None]) * accel**2 + np.mean(accel, axis=1)[:, None]  # TODO: maybe a filter also
            signals[i+3] = gyro

    def __compute_features(self, signals, magnitude_only=True, info=None):
        # t0 = time()
        feats = []
        if magnitude_only:
            # np.sqrt(np.sum(np.ma.array(np.dstack(signals[0:3]), mask=np.isnan(np.dstack(signals[0:3]))) ** 2, axis=2))
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
            sensor = ['accel', 'gyro']
            axis = ['mag']
            splits = '1234'
            p = 10
            arcoef_id = [str(i + 1) for i in range(p)]
            cut = len(splits)
            feats += [utils.nancorrelation(acc_cut, gyr_cut).reshape(-1)
                      for acc_cut, gyr_cut in zip(utils.make_cuts(signals[0], cut),utils.make_cuts(signals[1], cut))]
        else:
            sensor = ['accel', 'gyro']
            axis = 'xyz'
            splits = '123'
            cut = len(splits)
            p = 5
            arcoef_id = [str(i + 1) for i in range(p)]
            feats += [scale.mad(s, axis=1) for s in signals]
            feats += [utils.sma(*signals[0:3]), utils.sma(*signals[3::])]
            feats += [utils.correlation(*signals[0:3]).T, utils.correlation(*signals[3::]).T]
        if not self.features_names:
            self.features_names += ['mean' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['median' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['max' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['min' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['std' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['mad' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['sma' + '_' + ss for ss in sensor]
            self.features_names += ['irq' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['energy' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['correlation' + '_' + ss + '_' + aa for ss in sensor for aa in ('xy', 'xz', 'yx')]
            self.features_names += ['mean_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['median_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['max_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['min_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            # self.features_names += ['std_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            # self.features_names += ['mad_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['iqr_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['energy_' + part + '_' + ss + '_' + ax for ss in sensor for ax in axis for part in splits]
            self.features_names += ['fmed' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['fmax' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['fmad' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['kurtosis' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
            self.features_names += ['ar_coeff_' + arid + '_' + ss + '_' + ax for ss in sensor for ax in axis for arid in arcoef_id]


        feats += [np.nanmean(s, axis=1, dtype='float64') for s in signals]
        feats += [np.nanmedian(s, axis=1) for s in signals]
        feats += [np.nanmax(s, axis=1) for s in signals]
        feats += [np.nanmin(s, axis=1) for s in signals]
        feats += [np.nanstd(s, axis=1) for s in signals]

        feats += [stats.iqr(s, axis=1, nan_policy='omit') for s in signals]
        feats += [np.nansum(s ** 2, axis=1) for s in signals]

        cut = len(splits)
        # L = utils.make_cuts(signals[0], cut)
        feats += [np.nanmean(a_cut, axis=1) for s in signals for a_cut in utils.make_cuts(s, cut)]
        feats += [np.nanmedian(a_cut, axis=1) for s in signals for a_cut in utils.make_cuts(s, cut)]
        feats += [np.nanmax(a_cut, axis=1) for s in signals for a_cut in utils.make_cuts(s, cut)]
        feats += [np.nanmin(a_cut, axis=1) for s in signals for a_cut in utils.make_cuts(s, cut)]
        feats += [stats.iqr(a_cut, axis=1, nan_policy='omit') for s in signals for a_cut in utils.make_cuts(s, cut)]
        feats += [np.nansum(a_cut ** 2, axis=1) for s in signals for a_cut in utils.make_cuts(s, cut)]
        # feats += [np.nanmean(s[:, i * (s.shape[1] // cut):(i + 1) * (s.shape[1] // cut)], axis=1) for s in signals for i in range(cut)]
        # feats += [np.nanmedian(s[:, i * (s.shape[1] // cut):(i + 1) * (s.shape[1] // cut)], axis=1) for s in signals for i in range(cut)]
        # feats += [np.nanmax(s[:, i * (s.shape[1] // cut):(i + 1) * (s.shape[1] // cut)], axis=1) for s in signals for i in range(cut)]
        # feats += [np.nanmin(s[:, i * (s.shape[1] // cut):(i + 1) * (s.shape[1] // cut)], axis=1) for s in signals for i in range(cut)]
        # feats.append([np.std(s[:, i * (s.shape[1] // 5):(i + 1) * (s.shape[1] // 5)], axis=1) for s in signals for i in range(5)])
        # feats.append([scale.mad(s[:, i * (s.shape[1] // 5):(i + 1) * (s.shape[1] // 5)], axis=1) for s in signals for i in range(5)])
        # feats += [stats.iqr(s[:, i * (s.shape[1] // cut):(i + 1) * (s.shape[1] // cut)], axis=1, nan_policy='omit') for s in signals for i in range(cut)]
        # feats += [np.nansum(s[:, i * (s.shape[1] // cut):(i + 1) * (s.shape[1] // cut)]**2, axis=1) for s in signals for i in range(cut)]
        # feats.append([np.stld(s[:, i * (s.shape[1] // 5):(i + 1) * (s.shape[1] // 5)], axis=1) for s in signals for i in range(5)])
        # feats.append([scale.mad(s[:, i * (s.shape[1] // 5):(i + 1) * (s.shape[1] // 5)], axis=1) for s in signals for i in range(5)])
        ffts = [utils._fft(s, 0.01 * s.shape[1], handle_nans=True) for s in signals]
        feats += [utils.fft_metric(fft[0], fft[1], np.nanmedian, handle_nans=True) for fft in ffts]
        feats += [utils.fft_metric(fft[0], fft[1], np.nanmax, handle_nans=True) for fft in ffts]
        # feats += [utils.fft_metric(fft[0], fft[1], np.std) for fft in ffts]
        feats += [utils.fft_energy(fft[1]) for fft in ffts]

        feats += [stats.kurtosis(s, axis=1, nan_policy='omit') for s in signals]

        feats += [utils.ar_coeff(s, p).T for s in signals]

        # print('time: ', time() - t0)
        feats = np.vstack(feats).T
        return feats

    def decision_function(self, signals):
        self.test_signals = signals
        self.__process_signals(self.test_signals)
        X = self.__compute_features(self.test_signals)
        X = self.reducer.transform(X)
        h = self.clf.decision_function(X)
        return h/np.linalg.norm(h, axis=1).reshape(h.shape[0], 1)

    def predict(self, signals, with_second_choice=False):
        self.test_signals = signals
        self.__process_signals(self.test_signals)
        X = self.__compute_features(self.test_signals)
        X = self.scaler.transform(X)
        X = self.reducer.transform(X)
        return self.clf.predict(X, tol_rel=0, tol_abs=0, with_second_choice=with_second_choice)

    def score(self, signals, targets):
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
        # self.clf = LinearDiscriminantAnalysis()
        # self.clf = NearestCentroid()
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

    def predict(self, X, tol_rel=0, tol_abs=0, with_second_choice=False):  #0.05, 0.4 /// 0 0.5
        # h = self.decision_function(X)
        # alphas_j = np.argmax(h, axis=1)
        # alphas_j[alphas_j < (1 - self._lambda)] = self.K
        # return alphas_j
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



class ClassifieurAvecRejet:

    def __init__(self, _lambda=10):
        # _lambda est le coût de rejet
        self._lambda = _lambda
        self.m = []
        self.si2 = []
        self.prior = []
        self.K = 0
        self.d = 0

    def fit(self, X, y):
        # TODO Q3C
        # Implémentez ici une fonction permettant d'entraîner votre modèle
        # à partir des données fournies en argument
        self.d = X.shape[1]
        N = X.shape[0]
        self.K = len(set(y))
        self.m = np.zeros([self.d, 1])
        self.si2 = np.zeros([self.K, 1])
        self.prior = np.zeros([self.K, 1])
        self.mu = np.zeros([self.K, self.d])
        for k in range(self.K):
            r_indexes = np.where(y == k)[0]
            for dim in range(self.d):
                self.mu[k, dim] = np.mean(X[r_indexes, dim])  # Moyenne des colones par classe #
            self.prior[k] = np.where(y == k)[0].size / N
            vec_mu_i = self.mu[k, :]
            X_t = X[r_indexes, :]
            self.si2[k] = sum([np.dot(
                (row - vec_mu_i), (row - vec_mu_i)
            ) for row in X_t]) / (self.d * X_t.shape[0])

    def predict_proba(self, X):
        # TODO Q3C
        # Implémentez une fonction retournant la probabilité d'appartenance
        # à chaque classe, pour les données passées en argument. Cette
        # fonction peut supposer que fit() a préalablement été appelé.
        # Indice : calculez les différents termes de l'équation de Bayes
        # séparément
        d = X.shape[1]
        N = X.shape[0]
        K = self.K
        m = self.mu
        si2 = self.si2
        prior = self.prior
        posterior = np.zeros([N, K])
        norm = 0
        const = np.sqrt((2 * np.pi) ** (d))
        for t in range(N):
            x = X[t, :]
            norm = 0
            for i in range(K):
                temp = np.exp(((-0.5 / si2[i]) * np.dot((x - m[i]), (x - m[i]))))
                likes = (1 / (const * np.sqrt(si2[i] ** d))) * temp
                norm += likes * prior[i]
                posterior[t, i] = likes * prior[i]
            posterior[t, :] /= norm
        return posterior

    def predict(self, X):
        # TODO Q3C
        # Implémentez une fonction retournant les prédictions pour les données
        # passées en argument. Cette fonction peut supposer que fit() a
        # préalablement été appelé.
        # Indice : vous pouvez utiliser predict_proba() pour éviter une
        # redondance du code
        d = X.shape[1]
        N = X.shape[0]
        preds = []
        posterior = self.predict_proba(X)
        for row in posterior:
            alpha_j = np.argmax(row)
            if row[alpha_j] < 1 - self._lambda:
                alpha_j = self.K
            preds.append(alpha_j)
        return np.array(preds)

    def score(self, X, y):
        # TODO Q3C
        # Implémentez une fonction retournant le score (tenant compte des données
        # rejetées si lambda < 1.0) pour les données passées en argument.
        # Cette fonction peut supposer que fit() a préalablement été exécuté.
        preds = self.predict(X)
        diff = np.array(y) - preds
        miss_w_rej = len(diff[diff != 0])  # ratés incluant rejet
        rej = len(preds[preds == self.K])  # rejet
        return 1 - (((miss_w_rej - rej) * 1 + rej * self._lambda) / len(y))


class ImuSignDetectTemplateMatcher:

    def __init__(self):
        self.clf = [KNeighborsClassifier(n_neighbors=8) for _ in range(6)]

    def fit_xcorr(self, signals, targets):
        self.__process_signals(signals)
        self.train_signals = signals
        corr_arrays = [np.zeros((sg.shape[0], sg.shape[0])) for sg in signals]
        for i_s, signal in enumerate(signals):
            signal -= np.mean(signal, axis=1).reshape(signal.shape[0], 1)
            signal /= (np.max(signal, axis=1) - np.min(signal, axis=1)).reshape(signal.shape[0], 1)
            for i in range(signal.shape[0]):
                template_inti = signal[i, :]
                for j in range(signal.shape[0]):
                    # corr_arrays[i_s][j, i] = dtw(template_inti.copy(), signal[j, :])
                    corr_arrays[i_s][j, i] = self.__signal_xcorr(template_inti.copy(), signal[j, :])
                # print(i_s, i)
            self.clf[i_s].fit(corr_arrays[i_s], targets)
            # sns.heatmap(pd.DataFrame(corr_arrays[i_s]))
            # plt.show()
        self.train_features = corr_arrays

    def predict(self, signals, with_del_elements=False):
        self.__process_signals(signals)
        nans = []
        corr_arrays = [np.zeros((sg.shape[0], self.train_signals[0].shape[0])) for sg in signals]
        for i_s, signal in enumerate(signals):
            signal -= np.mean(signal, axis=1).reshape(signal.shape[0], 1)
            signal /= (np.max(signal, axis=1) - np.min(signal, axis=1)).reshape(signal.shape[0], 1)
            for i in range(self.train_signals[0].shape[0]):
                template_inti = self.train_signals[i_s][i, :]
                for j in range(signal.shape[0]):
                    # corr_arrays[i_s][j, i] = dtw(template_inti.copy(), signal[j, :])
                    corr_arrays[i_s][j, i] = self.__signal_xcorr(template_inti.copy(), signal[j, :])
                    if np.isnan(corr_arrays[i_s][j, i]):
                        # print('nan pos: ', nans)
                        if j not in nans:
                            nans.append(j)
                            # print('nan pos: ', nans)
            # print(i_s)
                        # self.__signal_xcorr(template_inti.copy(), signal[j, :])
            # sns.heatmap(pd.DataFrame(corr_arrays[i_s]))
            # plt.show()
        preds = self.__decision_function([np.delete(corr_arrays[i_s], np.array(nans), axis=0) for i_s in range(6)])
        if with_del_elements:
            return preds, nans
        else:
            return preds


    def __decision_function(self, correlations):
        preds = []
        # priors = [1.5/12, 1.5/12, 1.5/12, 2.5/12, 2.5/12, 2.5/12]
        priors = [5, 2, 3, 4, 3, 1]
        try:
            probs = np.zeros((correlations[0].shape[0], len(set(self.clf[0]._y))))
        except AttributeError:
            probs = np.zeros((correlations[0].shape[0], len(self.clf[0].classes_)))
        for corr, clf, prior in zip(correlations, self.clf, priors):
            preds.append(clf.predict(corr))
            probs += prior * clf.predict_proba(corr)
        return np.argmax(probs, axis=1)

    def score(self, signals, targets):
        y_preds, dels = self.predict(signals, with_del_elements=True)
        targets = np.delete(targets, dels)
        diff = y_preds - targets
        print(confusion_matrix(targets, y_preds))
        tot = diff.shape[0]
        return (diff[diff == 0].shape[0]) / tot

    def __signal_xcorr(self, template, signal):
        corr, lag = utils.xcorr(template, signal)
        try:
            lagdiff = int(lag[np.where(np.abs(corr) == np.max(np.abs(corr)))[0]])
        except TypeError:
            lagdiff = int(lag[np.where(np.abs(corr) == np.max(np.abs(corr)))[0]][0])
        if lagdiff > 0:
            template = np.delete(template, np.s_[0:lagdiff], axis=0)
        else:
            signal = np.delete(signal, np.s_[0:-lagdiff], axis=0)
        if len(template) < len(signal):
            signal = np.delete(signal, np.s_[len(template):len(signal)])
        else:
            template = np.delete(template, np.s_[len(signal):len(template)])
        return np.abs(np.corrcoef(template.reshape(-1), signal.reshape(-1))[0, 1])

    def __process_signals(self, signals):
        for i, (accel, gyro) in enumerate(zip(signals[0:3], signals[3::])):
            signals[i] = np.sign(accel - np.mean(accel, axis=1)[:, None]) * accel**2 + np.mean(accel, axis=1)[:, None]  # TODO: maybe a filter also
            signals[i+3] = gyro