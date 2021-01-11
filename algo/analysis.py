import numpy as np
import pandas as pd
import pickle as pkl
import os
from scipy import stats, signal, linalg
from statsmodels.robust import scale
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import minmax_scale
from tslearn.metrics import dtw, gak
from . import utils



class MyIMUDataset:

    def __init__(self, format='split', sample=None):
        self.data = {}
        self.accel_x = []
        self.accel_y = []
        self.accel_z = []
        self.gyro_x = []
        self.gyro_y = []
        self.gyro_z = []
        self.targets = []
        self.features = np.array([])
        self.features_names = []
        self.target_names = []
        self.W = np.array([])
        self.load_data(sample=sample)
        self.organize_data(format)
        self.load_features()

    def load_data(self, sample=None):
        with open('../my_data/IMU_data.pkl', 'rb') as f:
            seq_bin = f.read()
            self.data = pkl.loads(seq_bin)
            if sample is not None:
                for key, vals in self.data.items():
                    for i ,s in enumerate(vals[1]):
                        self.data[key][1][i] = s.sample(n=sample, axis=0)


    def organize_data(self, format='split', squared=False):
        for key, value in self.data.items():
            self.target_names.append(key)
            print(key, value[0])
            accelx = value[1][0].values  #  - np.mean(value[1][0].values, axis=1)[:, None]
            accely = value[1][1].values  #  - np.mean(value[1][1].values, axis=1)[:, None]
            accelz = value[1][2].values  #  - np.mean(value[1][2].values, axis=1)[:, None]
            gyrox = value[1][3].values
            gyroy = value[1][4].values
            gyroz = value[1][5].values
            set_len, series_len = value[1][0].values.shape
            label_row = value[0] * np.ones((set_len, 1), dtype=int)
            if format == 'one_array':
                self.accel_x.append(np.hstack([label_row ,accelx]))
                self.accel_y.append(np.hstack([label_row, accely]))
                self.accel_z.append(np.hstack([label_row, accelz]))
                self.gyro_x.append(np.hstack([label_row, gyrox]))
                self.gyro_y.append(np.hstack([label_row, gyroy]))
                self.gyro_z.append(np.hstack([label_row, gyroz]))
            elif format == 'split':
                self.accel_x.append(accelx)
                self.accel_y.append(accely)
                self.accel_z.append(accelz)
                self.gyro_x.append(gyrox)
                self.gyro_y.append(gyroy)
                self.gyro_z.append(gyroz)
            self.targets.append(label_row)
        self.accel_x = np.vstack(self.accel_x)
        self.accel_y = np.vstack(self.accel_y)
        self.accel_z = np.vstack(self.accel_z)
        self.gyro_x = np.vstack(self.gyro_x)
        self.gyro_y = np.vstack(self.gyro_y)
        self.gyro_z = np.vstack(self.gyro_z)
        if squared:
            self.accel_x = np.sign(self.accel_x - np.mean(self.accel_x, axis=1)[:, None]) * self.accel_x**2 + np.mean(self.accel_x, axis=1)[:, None]
            self.accel_y = np.sign(self.accel_y - np.mean(self.accel_y, axis=1)[:, None]) * self.accel_y**2 + np.mean(self.accel_y, axis=1)[:, None]
            self.accel_z = np.sign(self.accel_z - np.mean(self.accel_z, axis=1)[:, None]) * self.accel_z**2 + np.mean(self.accel_z, axis=1)[:, None]
            # self.gyro_x = np.sign(self.gyro_x) * self.gyro_x**2
            # self.gyro_y = np.sign(self.gyro_y) * self.gyro_y**2
            # self.gyro_z = np.sign(self.gyro_z) * self.gyro_z**2
        self.targets = np.vstack(self.targets).reshape(-1)
        # print(self.targets)

    def fit_dtw(self):
        signals = [self.gyro_x] #, self.accel_y, self.accel_z, self.gyro_x, self.gyro_y, self.gyro_z]
        signals_out = [np.zeros((sg.shape[0], sg.shape[0])) for sg in signals]
        for k, sensor in enumerate(signals):
            sensor -= np.mean(sensor, axis=1).reshape(sensor.shape[0], 1)
            sensor /= (np.max(sensor, axis=1) - np.min(sensor, axis=1)).reshape(sensor.shape[0], 1)
            for i, template in enumerate(sensor):
                for j, s in enumerate(sensor):
                    sim = dtw(template, s)
                    # correlation = np.corrcoef(template.reshape(-1), s.reshape(-1))
                    signals_out[k][i, j] = sim
                print(k, i)
        sns.heatmap(pd.DataFrame(signals_out[0]))
        plt.show()

    def fit_xcorr(self):
        signals = [self.accel_x]  #, self.accel_y, self.accel_z, self.gyro_x, self.gyro_y, self.taset.gyro_z]
        signals_out = [np.zeros((sg.shape[0], sg.shape[0])) for sg in signals]
        for k, sensor in enumerate(signals):
            sensor -= np.mean(sensor, axis=1).reshape(sensor.shape[0], 1)
            sensor /= (np.max(sensor, axis=1) - np.min(sensor, axis=1)).reshape(sensor.shape[0], 1)
            for i in range(sensor.shape[0]):
                template_inti = sensor[i, :]
                # signals_to_compare = np.delete(sensor, i, axis=1)
                for j in range(sensor.shape[0]):
                    template = template_inti.copy()
                    s = sensor[j, :]
                    corr, lag = utils.xcorr(template, s)
                    try:
                        lagdiff = int(lag[np.where(np.abs(corr) == np.max(np.abs(corr)))[0]])
                    except TypeError:
                        lagdiff = int(lag[np.where(np.abs(corr) == np.max(np.abs(corr)))[0]][0])
                    if lagdiff > 0:
                        template = np.delete(template, np.s_[0:lagdiff], axis=0)
                    else:
                        s = np.delete(s, np.s_[0:-lagdiff], axis=0)
                    if len(template) < len(s):
                        s = np.delete(s, np.s_[len(template):len(s)])
                    else:
                        template = np.delete(template, np.s_[len(s):len(template)])
                    correlation = np.corrcoef(template.reshape(-1), s.reshape(-1))
                    signals_out[k][i, j] = correlation[0, 1]
                print(k, i, j)
        sns.heatmap(pd.DataFrame(signals_out[0]))
        plt.show()

    def load_features(self):
        if os.path.isfile('../my_data/IMU_features.pkl'):
            with open('../my_data/IMU_features.pkl', 'rb') as f:
                seq_bin = f.read()
                self.features = pkl.loads(seq_bin)
        else:
            self.__extract_features()
        return self.features, self.targets

    def __extract_features(self):
        sensor = ['accel', 'gyro']
        axis = 'xyz'
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
        # self.features_names += ['fmean' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
        # self.features_names += ['fmed' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
        # self.features_names += ['fmax' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
        # self.features_names += ['fmin' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
        # self.features_names += ['std' + '_' + ss + '_' + ax for ss in sensor for ax in axis]
        # self.features_names += ['mad' + '_' + ss + '_' + ax for ss in sensor for ax in axis]

        signals = [self.accel_x, self.accel_y, self.accel_z, self.gyro_x, self.gyro_y, self.gyro_z]
        feats = []
        feats.append([np.mean(s, axis=1) for s in signals])
        feats.append([np.median(s, axis=1) for s in signals])
        feats.append([np.max(s, axis=1) for s in signals])
        feats.append([np.min(s, axis=1) for s in signals])
        feats.append([np.std(s, axis=1) for s in signals])
        feats.append([scale.mad(s, axis=1) for s in signals])
        feats.append([utils.sma(self.accel_x, self.accel_y, self.accel_z), utils.sma(self.gyro_x, self.gyro_y, self.gyro_z)])
        feats.append([stats.iqr(s, axis=1) for s in signals])
        feats.append([np.sum(s**2, axis=1) for s in signals])
        feats.append([utils.correlation(*signals[0:3]).T, utils.correlation(*signals[3::]).T])
        ffts = [utils._fft(s, 0.9) for s in signals]
        # feats.append([utils.fft_metric(fft[0], fft[1], np.mean) for fft in ffts])
        # feats.append([utils.fft_metric(fft[0], fft[1], np.median) for fft in ffts])
        # feats.append([utils.fft_metric(fft[0], fft[1], np.max) for fft in ffts])
        # feats.append([utils.fft_metric(fft[0], fft[1], np.min) for fft in ffts])
        # feats.append([utils.fft_metric(fft[0], fft[1], np.std) for fft in ffts])
        # feats.append([utils.fft_metric(fft[0], fft[1], scale.mad) for fft in ffts])
        self.features = np.hstack([np.vstack(fts).T for fts in feats])

        # to_plot = [9, 10, 11, 12, 13, 14 ,15 ,16 ,17, 30, 31, 32, 33, 34, 35]
        # utils.plot_1d_feature(feat=self.features[:,to_plot].T.tolist(), targets=self.targets, title='features', signal_names=np.array(self.features_names)[to_plot])


class DataAnalyser:

    def __init__(self, dataset):
        self.X = dataset.features
        self.y = dataset.targets
        self.feature_names = dataset.features_names
        self.best_params = {}
        self.best_features = []

    def square_signal(self, series, freq=100, fig_handle=None):
        delta_t = 1/freq
        median = np.median(series)
        mean = np.mean(series)
        max = np.max(series - mean)
        min = np.min(series - mean)
        series = series - mean
        series2 = minmax_scale(series)
        series3 = (np.sign(series)*series**1)
        series4 = utils.lowpass_filter(series3, .015, freq)
        series5 = utils.lowpass_filter(series, .025, freq)
        if fig_handle is None:
            fig, fig_handle = plt.subplot(1, 1)
        fig_handle.plot(series, label='raw')
        # plt.plot(series2, label='mimax')
        fig_handle.plot(series3, label='gauss')
        # plt.plot(series4, label='gauss+filter')
        # plt.plot(series5, label='raw+filter')
        area = minmax_scale((np.cumsum(series) * delta_t), feature_range=(min, max))
        # plt.plot(np.ones(series.shape) * np.std(series), label='mean+std')
        # plt.plot(np.ones(series.shape) * -np.std(series), label='mean-std')
        # plt.plot(area, label='integral')


    def ldaitr_fit_score(self, scale=True):
        if scale:
            self.X = minmax_scale(self.X)
            X = self.X
            y = self.y
        else:
            X = self.X
            y = self.y
        clf = LinearDiscriminantAnalysis()
        clf.fit(X, y)
        n_splits = 10
        kf = KFold(n_splits=n_splits, shuffle=True)
        scr_train = []
        scr_test = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            scr_train.append(clf.score(X_train, y_train))
            scr_test.append(clf.score(X_test, y_test))
        score_test = np.mean(scr_test)
        score_train = np.mean(scr_train)
        return score_train, score_test

    def logitr_fit_score(self, scale=True, optimize=False, printit=True):
        if scale:
            self.X = minmax_scale(self.X)
            X = self.X
            y = self.y
        else:
            X = self.X
            y = self.y
        if optimize:
            clf = LogisticRegression()
            distribution = {'penalty': ['l2'], 'C': np.logspace(-10, 10, 11), 'solver':['newton-cg', 'lbfgs', 'liblinear']}
            rds = RandomizedSearchCV(clf, distribution, random_state=0, cv=3, n_jobs=-1)
            search = rds.fit(X, y)
            self.best_params = search.best_params_
            cv_results = search.cv_results_
            if printit:
                print(self.best_params)
                print(cv_results)
            clf = LogisticRegression(penalty=self.best_params['penalty'], solver=self.best_params['solver'], C=self.best_params['C'])
        else:
            clf = LogisticRegression()
        clf.fit(X, y)
        n_splits = 10
        kf = KFold(n_splits=n_splits, shuffle=True)
        scr_train = []
        scr_test = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            scr_train.append(clf.score(X_train, y_train))
            scr_test.append(clf.score(X_test, y_test))
        score_test = np.mean(scr_test)
        score_train = np.mean(scr_train)
        return score_train, score_test

    def linsvm_fit_score(self, scale=True, optimize=False, printit=True):
        if scale:
            self.X = minmax_scale(self.X)
            X = self.X
            y = self.y
        else:
            X = self.X
            y = self.y
        if optimize:
            clf = LinearSVC(dual=False)
            distribution = {'penalty': ['l1', 'l2'], 'loss': ['squared_hinge'], 'C': np.logspace(-10, 10, 11)}
            rds = RandomizedSearchCV(clf, distribution, random_state=0, cv=3, n_jobs=-1)
            search = rds.fit(X, y)
            self.best_params = search.best_params_
            cv_results = search.cv_results_
            if printit:
                print(self.best_params)
                print(cv_results)
            clf = LinearSVC(dual=False, penalty=self.best_params['penalty'], loss=self.best_params['loss'], C=self.best_params['C'])
        else:
            clf = LinearSVC(dual=False)
        clf.fit(X, y)
        n_splits = 10
        kf = KFold(n_splits=n_splits, shuffle=True)
        scr_train = []
        scr_test = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            scr_train.append(clf.score(X_train, y_train))
            scr_test.append(clf.score(X_test, y_test))
        score_test = np.mean(scr_test)
        score_train = np.mean(scr_train)
        return score_train, score_test

    def svm_fit_score(self, scale=True, optimize=False, printit=True):
        if scale:
            self.X = minmax_scale(self.X)
            X = self.X
            y = self.y
        else:
            X = self.X
            y = self.y
        if optimize:
            clf = SVC(gamma='scale')
            distribution = {'kernel': ['rbf', 'poly', 'sigmoid'], 'C': np.logspace(-10, 10, 11)}
            rds = RandomizedSearchCV(clf, distribution, random_state=0, cv=3, n_jobs=-1)
            search = rds.fit(X, y)
            self.best_params = search.best_params_
            cv_results = search.cv_results_
            if printit:
                print(self.best_params)
                print(cv_results)
            clf = SVC(kernel=self.best_params['kernel'], gamma='scale', C=self.best_params['C'])
        else:
            clf = SVC(gamma='scale')
        clf.fit(X, y)
        n_splits = 10
        kf = KFold(n_splits=n_splits, shuffle=True)
        scr_train = []
        scr_test = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            scr_train.append(clf.score(X_train, y_train))
            scr_test.append(clf.score(X_test, y_test))
        score_test = np.mean(scr_test)
        score_train = np.mean(scr_train)
        return score_train, score_test

    def check_dataset(self, model, n_components=2):
        dim = model(n_components=n_components)
        X_new = dim.fit_transform(self.X)
        self.W = np.linalg.lstsq(self.X, X_new)
        if n_components == 2:
            utils.plot_clustering(X_new, self.y, model.__name__, None)
        elif n_components == 3:
            utils.plot_clustering3D(X_new, self.y, model.__name__, None)

    def feat_importance(self, printit=True, show_hist=False):
        k_best = self.X.shape[1]
        forest = RandomForestClassifier(n_estimators=1, random_state=42, n_jobs=-1)
        forest.fit(self.X, self.y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        best_rf = indices[0:k_best]
        if printit:
            for f in range(self.X.shape[1]):
                #     print(feat_labels[f])
                # print("%2d) %-*s %f" % (f + 1, 30,  self.feature_names[indices[f]], importances[indices[f]]))
                print(f'{f + 1} ({indices[f]}) {self.feature_names[indices[f]]} {importances[indices[f]]}')
        if show_hist:
            plt.title('Feature Importances')
            plt.bar(range(self.X.shape[1]), importances[indices], align='center')
            plt.xticks(range(self.X.shape[1]), self.feature_names, rotation=90)
            plt.xlim([-1, self.X.shape[1]])
            plt.tight_layout()
            plt.show()
        self.best_features = best_rf
        return best_rf

    def features_selection_analysis(self):
        k_best = self.X.shape[1]
        best_rf = self.feat_importance()
        if len(self.best_params) == 0:
            self.svm_fit_score(optimize=True)
        clf = SVC(kernel=self.best_params['kernel'], gamma='scale', C=self.best_params['C'])
        strain = np.zeros((k_best, ))
        stest = np.zeros((k_best, ))
        n_splits = 3
        for i in range(n_splits, k_best):
            print('-------')
            kf = KFold(n_splits=n_splits, shuffle=True)
            scr_train = []
            scr_test = []
            feat_index = best_rf[0:i + 1]
            for train_index, test_index in kf.split(self.X):
                X_train, X_test = self.X[train_index][:, feat_index], self.X[test_index][:, feat_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                clf.fit(X_train, y_train)
                scr_train.append(clf.score(X_train, y_train))
                scr_test.append(clf.score(X_test, y_test))
            strain[i] = np.mean(scr_train)
            stest[i] = np.mean(scr_test)
            print(i, ': ', strain[i], stest[i])
        plt.figure()
        plt.plot(list(range(n_splits, k_best)), stest[n_splits::], label='test')
        plt.plot(list(range(n_splits, k_best)), strain[n_splits::], label='train')
        plt.xlabel('N feats')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

    def dataset_size_analysis(self, classifier='svm'):
        train_sizes = [5, 10, 25, 50, 100, 200, 300, 400, 500]
        stest = []
        strain =[]
        for ts in train_sizes:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=ts, random_state=42)
            if classifier == 'svm':
                self.svm_fit_score(optimize=True)
                clf = SVC(kernel=self.best_params['kernel'], gamma='scale', C=self.best_params['C'])
            elif classifier == 'lr':
                self.logitr_fit_score(optimize=True)
                clf = LogisticRegression(penalty=self.best_params['penalty'], solver=self.best_params['solver'],
                                         C=self.best_params['C'])
            elif classifier == 'lsvm':
                self.linsvm_fit_score(optimize=True)
                clf = LinearSVC(dual=False, penalty=self.best_params['penalty'], loss=self.best_params['loss'],
                                C=self.best_params['C'])
            clf.fit(X_train, y_train)
            strain.append(clf.score(X_train, y_train))
            stest.append(clf.score(X_test, y_test))
        plt.figure()
        plt.plot(train_sizes, stest, label='test')
        plt.plot(train_sizes, strain, label='train')
        plt.xlabel('N sample')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

    def feat_size_analysis(self):
        k_best = self.X.shape[1]
        train_sizes = [10, 25, 50, 100]  #, 25, 50, 100, 200, 300, 400, 500
        if not self.best_features:
            self.feat_importance(printit=False)
        strain = np.zeros((len(train_sizes), k_best))
        stest = np.zeros((len(train_sizes), k_best))
        fig, axs = plt.subplots(1, 2)
        for i, ts in enumerate(train_sizes):
            for fs in range(5, k_best):
                feat_index = self.best_features[0:fs + 1]
                X_train, X_test, y_train, y_test = train_test_split(self.X[:, feat_index], self.y, train_size=ts, random_state=42)
                clf = SVC(gamma='scale')
                distribution = {'kernel': ['rbf', 'poly', 'sigmoid'], 'C': np.logspace(-10, 10, 11)}
                rds = RandomizedSearchCV(clf, distribution, random_state=0, cv=3, n_jobs=-1)
                search = rds.fit(X_train, y_train)
                self.best_params = search.best_params_
                # strain[i, fs], stest[i, fs] = self.svm_fit_score(optimize=True, scale=True, printit=False)
                clf = SVC(kernel=self.best_params['kernel'], gamma='scale', C=self.best_params['C'])
                clf.fit(X_train, y_train)
                strain[i, fs] = clf.score(X_train, y_train)
                stest[i, fs] = clf.score(X_test, y_test)
                print(ts, fs, self.best_params)
            axs[0].plot(strain[i, :], label=str(ts))
            axs[1].plot(stest[i, :], label=str(ts))

        axs[0].set_title('train')
        axs[0].set_xlabel('n_feat')
        axs[1].set_title('test')
        axs[1].set_xlabel('n_feat')
        axs[0].legend()
        axs[1].legend()
        plt.show()















    # def single_fit_xcorr(self, template, signal, plots=False):
    #     t0 = time()
    #     if plots:
    #         plt.figure()
    #         plt.plot(template, label='template')
    #         plt.plot(signal, label='signal')
    #     template -= np.mean(template)
    #     template /= (np.max(template) - np.min(template))
    #     signal -= np.mean(signal)
    #     signal /= (np.max(signal) - np.min(signal))
    #     signal = signal.reshape(-1)
    #     template = template.reshape(-1)
    #     if plots:
    #         plt.figure()
    #         plt.plot(template, label='template')
    #         plt.plot(signal, label='signal')
    #     corr, lag = utils.xcorr(template, signal)
    #     lagdiff = int(lag[np.where(np.abs(corr) == np.max(np.abs(corr)))[0]])
    #     if lagdiff > 0:
    #         template = np.delete(template, np.s_[0:lagdiff], axis=0)
    #     else:
    #         signal = np.delete(signal, np.s_[0:-lagdiff], axis=0)
    #     if len(template) < len(signal):
    #         signal = np.delete(signal, np.s_[len(template):len(signal)])
    #     else:
    #         template = np.delete(template, np.s_[len(signal):len(template)])
    #     if plots:
    #         plt.figure()
    #         plt.plot(template, label='template')
    #         plt.plot(signal, label='signal')
    #         plt.show()
    #     correlation = np.corrcoef(template.reshape(-1), signal.reshape(-1))
    #     print(correlation, time() - t0)