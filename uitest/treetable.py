# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'treetable.ui'
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
        self.treeWidget = QtWidgets.QTreeWidget(Dialog)
        self.treeWidget.setGeometry(QtCore.QRect(60, 70, 401, 192))
        self.treeWidget.setObjectName("treeWidget")
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0.setCheckState(0, QtCore.Qt.Checked)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Manage sequences"))
        self.treeWidget.headerItem().setText(0, _translate("Dialog", "1"))
        self.treeWidget.headerItem().setText(1, _translate("Dialog", "Nouvelle colonne"))
        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        self.treeWidget.topLevelItem(0).setText(0, _translate("Dialog", "Nouvel élément"))
        self.treeWidget.topLevelItem(1).setText(0, _translate("Dialog", "Nouvel élément"))
        self.treeWidget.topLevelItem(2).setText(0, _translate("Dialog", "Nouvel élément"))
        self.treeWidget.topLevelItem(3).setText(0, _translate("Dialog", "Nouvel élément"))
        self.treeWidget.topLevelItem(3).setText(1, _translate("Dialog", "3163"))
        self.treeWidget.topLevelItem(4).setText(0, _translate("Dialog", "Nouvel élément"))
        self.treeWidget.setSortingEnabled(__sortingEnabled)




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
