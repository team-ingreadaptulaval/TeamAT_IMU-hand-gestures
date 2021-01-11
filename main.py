from signgui import Window
from PyQt5.QtWidgets import QApplication
import sys


if __name__== "__main__":
    app = QApplication(sys.argv)
    window = Window()
    app.exec()

