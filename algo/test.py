from imusigndetect import *
from analysis import *
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, SpectralEmbedding,Isomap, LocallyLinearEmbedding
import utils
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.metrics import confusion_matrix
from time import time



if __name__== "__main__":
    strain = []
    stest = []
    cm = np.zeros((6, 6))
    for _ in range(30):
        signals, targets, targets_names = utils.load_imu_data()
        X_train, X_test, y_train, y_test = utils.split_signals(signals, targets, 5)
        sd = ImuSignDetectClassifier()
        sd.fit(X_train, y_train)
        strain.append(sd.score(X_train, y_train))
        stest.append(sd.score(X_test, y_test))
        print(targets_names)
        print(strain[-1], stest[-1])
        this_cm = confusion_matrix(y_test, sd.predict(X_test))
        try:
            cm += this_cm
        except ValueError:
            this_cm = np.hstack([this_cm, np.zeros((5, 1))])
            this_cm = np.vstack([this_cm, np.zeros((1, 6))])
            cm += this_cm
        print(this_cm)
        # h = sd.decision_function(X_test)
        # plt.plot(h)
        # plt.show()
    print('--------------------')
    print(f'Train: {np.mean(strain)}+-{np.std(strain)}, Test: {np.mean(stest)}+-{np.std(stest)}')
    print(cm/30)


    # for _ in range(5):
    #     signals, targets, targets_names = utils.load_FS_IMU_data()
    #     X_train, X_test, y_train, y_test = utils.split_signals(signals, targets, 5)
    #     sd = ImuSignDetectTemplateMatcher()
    #     t0  =time()
    #     sd.fit_xcorr(X_train, y_train)
    #     print('----------------------', time() - t0 , '-------------')
    #     t0 = time()
    #     score = sd.score(X_test, y_test)
    #     print(score)
    #     print('----------------------', time() - t0, '-------------')






    # data = MyIMUDataset()
    # data.load_features()
    # X, y = data.load_features()
    # analyser = DataAnalyser(data)
    # fig, axs = plt.subplots(3, 1)
    # for i, ax in enumerate(axs):
    #     analyser.square_signal(data.gyro_y[7*i, :], fig_handle=ax)
    #     ax.set_title(f'class: {data.targets[7*i]}')
    # plt.legend()
    # plt.show()
    # data.fit_xcorr()
    # score_train, score_test = analyser.svm_fit_score(optimize=True, scale=True)
    # score_train, score_test = analyser.ldaitr_fit_score()
    # print(score_train, score_test)
    # analyser.check_dataset(model=TSNE, n_components=2)
    # analyser.features_selection_analysis()
    # analyser.feat_size_analysis()
    # print(analyser.linsvm_fit_score(optimize=True), analyser.best_params)
    # analyser.dataset_size_analysis(classifier='lsvm')





    # template = data.accel_z[0, :]
    # i = 400
    # signal = data.accel_z[i, :]
    # print(data.targets[0], data.targets[i])
    # template =  pd.read_csv('../ACL_comparaison_xcorr/bon1.csv', header=None).values
    # signal = pd.read_csv('../ACL_comparaison_xcorr/mauvais1.csv', header=None).values
    # analyser.single_fit_xcorr(template, signal, plots=True)
    # import tsfresh
    # signals, targets, targets_names = utils.load_FS_IMU_data()
    # signals_train, signals_test, y_train, y_test = utils.split_signals(signals, targets, 80)
    # t = np.hstack([np.arange(90) for _ in range(signals_train[0].shape[0])])
    # id = np.vstack([np.arange(signals_train[0].shape[0]) for _ in range(90)]).T.reshape(-1)
    # for i, signal in enumerate(signals_train):
    #     signals_train[i] = signal.reshape(-1)
    # signals_train = np.vstack(signals_train)
    # df = pd.DataFrame(np.vstack([id, t, signals_train]).T, columns=['id', 'time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
    # X = tsfresh.extract_features(df, column_id='id', column_sort='time')
    # print(X)
    # X.to_csv('export_dataframe80_test.csv', index = None, header=True)
    # y = np.vstack([np.arange(5) for _ in range(80)]).T.reshape(-1)
    # X = pd.read_csv('export_dataframe80_test.csv')
    # X = X.dropna(axis=1)
    # # df.dropna(axis=1)
    # for i in range(20):
    #     pca = PCA(n_components=25)
    #     X_new = pca.fit_transform(X, y)
    #     X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size=30, random_state=42)
    #     from sklearn.svm import SVC
    #     clf = SVC(kernel='rbf', C=0.1, gamma='scale')
    #     clf.fit(X_train, y_train)
    #     print('-----', i, '-----')
    #     print(clf.score(X_train, y_train))
    #     print(clf.score(X_test, y_test))
    #     print(confusion_matrix(y_test, clf.predict(X_test)))
    # np.savetxt(fname='extfeats.csv', X=X.values, delimiter=',')
    # from tsfresh.examples import load_robot_execution_failures
    # from tsfresh import extract_features
    # df, _ = load_robot_execution_failures()
    # X = extract_features(df, column_id='id', column_sort='time')
    # print(X)
