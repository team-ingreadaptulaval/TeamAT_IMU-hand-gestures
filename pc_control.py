import numpy as np
import keyboard
# import pyautogui as pag
from time import time, sleep
import win32api, win32con
from xsens_py_interface import clientsocket
from threading import Thread, Lock
from PyQt5.QtCore import pyqtSignal, QThread

VK_CODE = {'backspace':0x08,
           'tab':0x09,
           'clear':0x0C,
           'enter':0x0D,
           'shift':0x10,
           'ctrl':0x11,
           'alt':0x12,
           'pause':0x13,
           'caps_lock':0x14,
           'esc':0x1B,
           'spacebar':0x20,
           'page_up':0x21,
           'page_down':0x22,
           'end':0x23,
           'home':0x24,
           'left_arrow':0x25,
           'up_arrow':0x26,
           'right_arrow':0x27,
           'down_arrow':0x28,
           'select':0x29,
           'print':0x2A,
           'execute':0x2B,
           'print_screen':0x2C,
           'ins':0x2D,
           'del':0x2E,
           'help':0x2F,
           '0':0x30,
           '1':0x31,
           '2':0x32,
           '3':0x33,
           '4':0x34,
           '5':0x35,
           '6':0x36,
           '7':0x37,
           '8':0x38,
           '9':0x39,
           'a':0x41,
           'b':0x42,
           'c':0x43,
           'd':0x44,
           'e':0x45,
           'f':0x46,
           'g':0x47,
           'h':0x48,
           'i':0x49,
           'j':0x4A,
           'k':0x4B,
           'l':0x4C,
           'm':0x4D,
           'n':0x4E,
           'o':0x4F,
           'p':0x50,
           'q':0x51,
           'r':0x52,
           's':0x53,
           't':0x54,
           'u':0x55,
           'v':0x56,
           'w':0x57,
           'x':0x58,
           'y':0x59,
           'z':0x5A,
           'numpad_0':0x60,
           'numpad_1':0x61,
           'numpad_2':0x62,
           'numpad_3':0x63,
           'numpad_4':0x64,
           'numpad_5':0x65,
           'numpad_6':0x66,
           'numpad_7':0x67,
           'numpad_8':0x68,
           'numpad_9':0x69,
           'multiply_key':0x6A,
           'add_key':0x6B,
           'separator_key':0x6C,
           'subtract_key':0x6D,
           'decimal_key':0x6E,
           'divide_key':0x6F,
           'F1':0x70,
           'F2':0x71,
           'F3':0x72,
           'F4':0x73,
           'F5':0x74,
           'F6':0x75,
           'F7':0x76,
           'F8':0x77,
           'F9':0x78,
           'F10':0x79,
           'F11':0x7A,
           'F12':0x7B,
           'F13':0x7C,
           'F14':0x7D,
           'F15':0x7E,
           'F16':0x7F,
           'F17':0x80,
           'F18':0x81,
           'F19':0x82,
           'F20':0x83,
           'F21':0x84,
           'F22':0x85,
           'F23':0x86,
           'F24':0x87,
           'num_lock':0x90,
           'scroll_lock':0x91,
           'left_shift':0xA0,
           'right_shift ':0xA1,
           'left_control':0xA2,
           'right_control':0xA3,
           'left_menu':0xA4,
           'right_menu':0xA5,
           'browser_back':0xA6,
           'browser_forward':0xA7,
           'browser_refresh':0xA8,
           'browser_stop':0xA9,
           'browser_search':0xAA,
           'browser_favorites':0xAB,
           'browser_start_and_home':0xAC,
           'volume_mute':0xAD,
           'volume_Down':0xAE,
           'volume_up':0xAF,
           'next_track':0xB0,
           'previous_track':0xB1,
           'stop_media':0xB2,
           'play/pause_media':0xB3,
           'start_mail':0xB4,
           'select_media':0xB5,
           'start_application_1':0xB6,
           'start_application_2':0xB7,
           'attn_key':0xF6,
           'crsel_key':0xF7,
           'exsel_key':0xF8,
           'play_key':0xFA,
           'zoom_key':0xFB,
           'clear_key':0xFE,
           '+':0xBB,
           ',':0xBC,
           '-':0xBD,
           '.':0xBE,
           '/':0xBF,
           ';':0xBA,
           '[':0xDB,
           '\\':0xDC,
           ']':0xDD,
           "'":0xDE,
           '`':0xC0}


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
        self.strcmd2fct = {'copy': self.mouse.copy, 'paste': self.mouse.paste, 'lclick': self.mouse.left_click, 'rclick': self.mouse.right_click, 'up': self.mouse.move_up, 'down': self.mouse.move_down,
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

    def copy(self):
        if self.fresh_switch:
            win32api.keybd_event(0x11, 0, 0, 0)
            win32api.keybd_event(0x43, 0, 0, 0)
            win32api.keybd_event(0x43, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(0x11, 0, win32con.KEYEVENTF_KEYUP, 0)

    def copy(self):
        if self.fresh_switch:
            win32api.keybd_event(VK_CODE['ctrl'], 0, 0, 0)
            win32api.keybd_event(VK_CODE['c'], 0, 0, 0)
            win32api.keybd_event(VK_CODE['c'], 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(VK_CODE['ctrl'], 0, win32con.KEYEVENTF_KEYUP, 0)

    def paste(self):
        if self.fresh_switch:
            win32api.keybd_event(VK_CODE['ctrl'], 0, 0, 0)
            win32api.keybd_event(VK_CODE['v'], 0, 0, 0)
            win32api.keybd_event(VK_CODE['v'], 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(VK_CODE['ctrl'], 0, win32con.KEYEVENTF_KEYUP, 0)

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
    #     mouse.move_upright()
#     mouse.move_upright()#     mouse.move_upright()#     mouse.move_upright()
#     mouse.move_upright()#     mouse.move_upright()#     mouse.move_upright()
#     mouse.move_upright()#     mouse.move_upright()#     mouse.move_upright()
#     mouse.move_upright()#     mouse.move_upright()#     mouse.move_upright()
