from sklearn.metrics import confusion_matrix
from xsens_py_interface import clientsocket
import numpy as np
from numpy import linalg
from PyQt5.QtWidgets import QApplication
import sys
from PyQt5.QtCore import pyqtSignal, QThread
from threading import Lock
import keyboard
from time import sleep


class TCPThread(QThread):
    str_recieved = pyqtSignal(list)
    def __init__(self, parent=None):
        super().__init__(parent)
        # init stuff
        self.__command_string = 'signdetect,0,0'
        self.data = [0., 0., 0., 0., 0., 0.]
        self.data_m1 = self.data.copy()
        # Connect to XSens server interface or serial com from cortex data
        try:
            self.socket = clientsocket.ClientSocket()
        except ConnectionRefusedError:
            print('Client socket unable to connect'.upper())
            self.socket = False
        self.count = 0
        self.init = True
        self.lock = Lock()

    def run(self):
        while True:
            # Send data to server
            if self.socket:
                self.socket.send(self.__command_string)

            # Signdetect algorithm
            if self.init or self.data != self.data_m1:
                self.init = False
                # print(self.data, flush=True)
                self.data_m1 =self.data.copy()
                self.str_recieved.emit(self.data)
                # self.command_string = self.sd.handle_new_data(self.data)  # data = [ax, ay, az, gx, gy, gz]
                self.count += 1
                # print(data[3::])

            # recieve data form server
            if self.socket:
                self.data = self.socket.recieve()

    def set_command_string(self, string):
        with self.lock:
            self.__command_string = string

class KeySwitch(QThread):
    def __init__(self, key, parent=None):
        super().__init__(parent)
        self.key = key
        self.state = 0

    def run(self):
        while True:
            if keyboard.is_pressed(self.key):
                self.state = 1
            else:
                self.state = 0
            sleep(0.01)
