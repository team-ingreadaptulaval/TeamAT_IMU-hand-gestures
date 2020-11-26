from livesigndetect import LiveSignatureDetection
from sklearn.metrics import confusion_matrix
from xsens_py_interface import clientsocket
import numpy as np
from numpy import linalg
from signgui import Window
from PyQt5.QtWidgets import QApplication
import sys


if __name__== "__main__":
    app = QApplication(sys.argv)
    window = Window()
    app.exec()

    # # init stuff
    # command_string = 'signdetect,0'
    # sd = LiveSignatureDetection(window_len=100, freq=100)
    # data = [0., 0., 0., 0., 0., 0.]
    # data_m1 = data.copy()
    #
    #
    # # Connect to XSens server interface or serial com from cortex data
    # try:
    #     socket = clientsocket.ClientSocket()
    # except ConnectionRefusedError:
    #     print('Client socket unable to connect'.upper())
    #     socket = False
    #
    # # Show GUI
    # # TODO: this

    #
    # count = 0
    # init = True
    # while True:
    #     # Send data to server
    #     if socket:
    #         socket.send(command_string)
    #
    #
    #     # Signdetect algorithm
    #     if init or data != data_m1:
    #         init = False
    #         data_m1 = data.copy()
    #         command_string = sd.handle_new_data(data)  # data = [ax, ay, az, gx, gy, gz]
    #         count += 1
    #         # print(data[3::])
    #
    #
    #     # recieve data form server
    #     if socket:
    #         data = socket.recieve()




    #
    # from Algo_dev.imusigndetect import *
    # from Algo_dev import utils
    #
    # strain = []
    # stest = []
    # cm = np.zeros((6, 6))
    # for _ in range(20):
    #     signals, targets, targets_names = utils.load_FS_IMU_data(file='C:/Users/teamat/Documents/LocalSignDectect/step13 IMU_algo/my_data/IMU_data.pkl')
    #     X_train, X_test, y_train, y_test = utils.split_signals(signals, targets, 5)
    #     sd = ImuSignDetectClassifier()
    #     sd.fit(X_train, y_train)
    #     strain.append(sd.score(X_train, y_train))
    #     stest.append(sd.score(X_test, y_test))
    #     print(targets_names)
    #     print(strain[-1], stest[-1])
    #     this_cm = confusion_matrix(y_test, sd.predict(X_test))
    #     try:
    #         cm += this_cm
    #     except ValueError:
    #         this_cm = np.hstack([this_cm, np.zeros((5, 1))])
    #         this_cm = np.vstack([this_cm, np.zeros((1, 6))])
    #         cm += this_cm
    #     print(this_cm)
    #     # h = sd.decision_function(X_test)
    #     # plt.plot(h)
    #     # plt.show()
    # print('--------------------')
    # print(f'Train: {np.mean(strain)}+-{np.std(strain)}, Test: {np.mean(stest)}+-{np.std(stest)}')
    # print(f'(min, max) Train: ({np.min(strain)}, {np.max(strain)}), Test: ({np.min(stest)}, {np.max(stest)})')
    # print(cm/20)
