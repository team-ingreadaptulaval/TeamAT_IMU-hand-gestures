# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog_manage1.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(684, 449)
        Dialog.setLayoutDirection(QtCore.Qt.LeftToRight)
        Dialog.setAutoFillBackground(False)
        self.horizontalWidget = QtWidgets.QWidget(Dialog)
        self.horizontalWidget.setGeometry(QtCore.QRect(70, 40, 521, 351))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalWidget.sizePolicy().hasHeightForWidth())
        self.horizontalWidget.setSizePolicy(sizePolicy)
        self.horizontalWidget.setMinimumSize(QtCore.QSize(351, 291))
        self.horizontalWidget.setAutoFillBackground(False)
        self.horizontalWidget.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.Canada))
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.listWidget = QtWidgets.QListWidget(self.horizontalWidget)
        self.listWidget.setObjectName("listWidget")
        self.horizontalLayout_2.addWidget(self.listWidget)
        self.pushButton = QtWidgets.QPushButton(self.horizontalWidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.comboBox = QtWidgets.QComboBox(self.horizontalWidget)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout_2.addWidget(self.comboBox)
        self.tableView = QtWidgets.QTableView(self.horizontalWidget)
        self.tableView.setObjectName("tableView")
        self.horizontalLayout_2.addWidget(self.tableView)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_4 = QtWidgets.QPushButton(self.horizontalWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Manage sequences"))
        self.pushButton.setText(_translate("Dialog", "PushButton"))
        self.pushButton_4.setText(_translate("Dialog", "PushButton"))
        self.pushButton_2.setText(_translate("Dialog", "PushButton"))
        self.pushButton_3.setText(_translate("Dialog", "PushButton"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
