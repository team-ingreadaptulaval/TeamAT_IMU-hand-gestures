from signgui import Window
from PyQt5.QtWidgets import QApplication
import sys
import os
import time
import subprocess
from pid import PidFile


if __name__== "__main__":
    if not os.path.exists('signdetectsocket.pid'):
        subprocess.Popen('py xsens_py_interface/multithread_server.py', creationflags=subprocess.DETACHED_PROCESS)
        time.sleep(3)
    app = QApplication(sys.argv)
    window = Window()
    app.exec()

