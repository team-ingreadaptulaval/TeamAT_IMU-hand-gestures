import numpy as np
import keyboard
# import pyautogui as pag
from time import time, sleep
import win32api, win32con
from xsens_py_interface import clientsocket
from threading import Thread, Lock
from PyQt5.QtCore import pyqtSignal, QThread


class HIDCom(QThread):

    def __init__(self):
        super().__init__()
        try:
            self.socket = clientsocket.ClientSocket()
        except ConnectionRefusedError:
            print('Client socket unable to connect'.upper())
            self.socket = False
        self.data = [0, 0]
        self.data_m1 = [0, 0]
        self.init = True
        self.mouse = MouseController()
        self.strcmd2numcmd = {'copy': 2, 'paste': 3, 'lclick': 4, 'rclick': 5, 'up': 6, 'down': 7, 'right': 8, 'left': 9,
                             'tab': 10, 'enter': 11}
        self.numcmd2strcmd = {v: k for k, v in self.strcmd2numcmd.copy().items()}
        self.strcmd2fct = {'copy': None, 'paste': None, 'lclick': self.mouse.left_click, 'rclick': self.mouse.right_click, 'up': self.mouse.move_up, 'down': self.mouse.move_down,
                           'right': self.mouse.move_right, 'left': self.mouse.move_left, 'tab': None, 'enter': None}


    def run(self):
        while True:
            # Request
            if self.socket:
                self.socket.send('hid,1')

            if self.init or self.data != self.data_m1:
                self.init = False
                self.mouse.fresh_switch = True
                # print(self.data)
                self.data_m1 =self.data.copy()
                # print(self.data, flush=True)
            if self.data[1]:
                try:
                    pcmd = self.strcmd2fct[self.numcmd2strcmd[self.data[0]]]
                except KeyError:
                    pcmd = None
                if pcmd is not None:
                    # print(pcmd)
                    pcmd()
            if self.socket:
                self.data = self.socket.recieve()
            self.mouse.fresh_switch = False



class MouseController:

    def __init__(self):
        self.speed = 200
        self.pos = win32api.GetCursorPos()
        self.fresh_switch = False

    def move_up(self):
        self.pos = win32api.GetCursorPos()
        win32api.SetCursorPos((self.pos[0], self.pos[1] - 1))
        sleep(1/self.speed)

    def move_down(self):
        self.pos = win32api.GetCursorPos()
        win32api.SetCursorPos((self.pos[0], self.pos[1] + 1))
        sleep(1/self.speed)

    def move_right(self):
        self.pos = win32api.GetCursorPos()
        win32api.SetCursorPos((self.pos[0] + 1, self.pos[1]))
        sleep(1/self.speed)

    def move_left(self):
        self.pos = win32api.GetCursorPos()
        win32api.SetCursorPos((self.pos[0] - 1, self.pos[1]))
        sleep(1/self.speed)

    def move_upright(self):
        self.pos = win32api.GetCursorPos()
        win32api.SetCursorPos((self.pos[0] + 1, self.pos[1] - 1))
        sleep((1 / (np.sqrt(2)/2)) / self.speed)

    def move_upleft(self):
        self.pos = win32api.GetCursorPos()
        win32api.SetCursorPos((self.pos[0] - 1, self.pos[1] - 1))
        sleep((1 / (np.sqrt(2)/2)) / self.speed)

    def move_downright(self):
        self.pos = win32api.GetCursorPos()
        win32api.SetCursorPos((self.pos[0] - 1, self.pos[1] + 1))
        sleep((1 / (np.sqrt(2)/2)) / self.speed)

    def move_down_left(self):
        self.pos = win32api.GetCursorPos()
        win32api.SetCursorPos((self.pos[0] + 1, self.pos[1] - 1))
        sleep((1 / (np.sqrt(2)/2)) / self.speed)

    def left_click(self):
        if self.fresh_switch:
            self.pos = win32api.GetCursorPos()
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, self.pos[0], self.pos[1], 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, self.pos[0], self.pos[1], 0, 0)

    def right_click(self):
        if self.fresh_switch:
            self.pos = win32api.GetCursorPos()
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, self.pos[0], self.pos[1], 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, self.pos[0], self.pos[1], 0, 0)

if __name__=='__main__':
    hid = HIDCom()
    hid.start()
    # mouse = MouseController()
    # for _ in range(500):
    #     mouse.move_up()
    # for _ in range(500):
    #     mouse.move_down()
    # for _ in range(500):
    #     mouse.move_upright()