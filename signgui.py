from PyQt5.QtWidgets import QMainWindow, QWidget, QDialog, QApplication, QPushButton, QListWidget, QAction, qApp,\
    QTextEdit, QLineEdit, QLabel, QMessageBox, QFileDialog, QInputDialog, \
    QListWidgetItem, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, \
    QSizePolicy, QGridLayout, QVBoxLayout, QHBoxLayout, QComboBox, QAbstractItemView
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QColor
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCan
import matplotlib.pyplot as plt
import numpy as np
import time
from comthread import TCPThread, KeySwitch
from livesigndetect import LiveSignatureDetection
import json
import ctypes
import pc_control


class Window(QMainWindow):

    command_recieved = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.window_len = 200
        # TODO: load last used files
        self.sd = LiveSignatureDetection(at='at_pc_dev', sequence_file='IMUsequences.asq', command_mapping_file='dev_cmd_map.acm', window_len=self.window_len, freq=100)
        self.cmd_map_file = self.sd.command_map_file
        self.sign2num = self.sd.target_id
        self.num2sign = {v: k for k, v in self.sign2num.items()}
        title = 'IMU SignDetect'
        self.top = 100
        self.left = 100
        self.width = 1400
        self.colors = [Qt.darkRed, Qt.red, Qt.darkYellow, Qt.blue, Qt.green, Qt.darkMagenta]
        self.height = int((1080 / 1920) * self.width)
        self.axes = None
        self.setWindowTitle(title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowIcon(QIcon('C:/Users/teamat/Pictures/yoda_icon.jpg'))
        myappid = u'TEAMAT.SignDetect.SequenceMatching.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        self.layout = MyGridLayout(self)
        self.setCentralWidget(self.layout)
        self.plot_init()
        self.ui_init()

        self.mySerial = TCPThread()
        self.mySerial.start()
        self.mySerial.str_recieved.connect(self.handle_new_data)
        self.switch = KeySwitch('Num_Lock')
        self.switch.start()
        self.hid = pc_control.HIDCom()
        self.hid.start()

        self.check_table_init()
        self.sign_record_init()

        self.show()

    def sign_record_init(self):
        self.text_sign_instruct = QTextEdit(self)
        self.text_sign_instruct.setReadOnly(True)
        self.text_sign_instruct.setTextColor(Qt.red)
        # self.text_sign_instruct.setStyleSheet("""QTextEdit {border-top: 1px solid black; border-bottom: 1px solid black;}""")
        self.layout.add_widget(self.text_sign_instruct, 0, 0, h=2)
        self.text_sign_instruct.hide()

    def ui_init(self):
        self.menubar_init()
        # button_layout = MyGridLayout()
        # self.button_record = QPushButton('Enregistrer\nsequence', self)
        # self.button_stop = QPushButton('Stop', self)
        # button_layout.add_widget(self.button_record, 0, 0)
        # button_layout.add_widget(self.button_stop, 1, 0)
        # self.layout.add_widget(button_layout, 1, 0)
        self.textedit_log = QTextEdit(self)
        self.textedit_log.setReadOnly(True)
        self.textedit_log.setFixedWidth(400)
        self.lineedit_cs = QLineEdit(self)
        self.lineedit_cs.setReadOnly(True)
        self.lineedit_cs.setFixedWidth(300)
        self.lineedit_cs.setFixedHeight(40)
        self.lineedit_cs.setStyleSheet("""QLineEdit {border: 2px solid black;}""")
        label_cs = QLabel('Current Sequence')
        label_cs.setFixedHeight(40)
        self.layout.add_widget(self.textedit_log, 0, 1, w=2)
        self.layout.add_widget(label_cs, 1, 1)
        self.layout.add_widget(self.lineedit_cs, 1, 2)


    def menubar_init(self):
        exit_act = QAction('&Exit', self)
        exit_act.setShortcut('Ctrl+Q')
        exit_act.setStatusTip('Exit application')
        exit_act.triggered.connect(qApp.quit)

        rec_new_sign_act = QAction('&Record new sign', self)
        rec_new_sign_act.setStatusTip('Record new simple signature')
        rec_new_sign_act.triggered.connect(self.record_new_sign)
        del_sign_act = QAction('&Delete sign', self)
        del_sign_act.setStatusTip('Delete a simple signature')
        del_sign_act.triggered.connect(self.delete_sign)

        manage_seq_act = QAction('&Manage sequence', self)
        manage_seq_act.setStatusTip('Build and delete sequences')
        manage_seq_act.triggered.connect(self.manage_sequence)

        new_mapping_act = QAction('&New command mapping', self)
        new_mapping_act.setStatusTip('Create new command mapping for assistive device')
        new_mapping_act.triggered.connect(self.create_new_command_mapping)
        load_mapping_act = QAction('&Load command mapping', self)
        load_mapping_act.setStatusTip('Load a saved command mapping for assistive device')
        load_mapping_act.triggered.connect(self.load_mapping)
        manage_mapping_act = QAction('&Current command mapping', self)
        manage_mapping_act.setStatusTip('Manage command mapping for the current assistive device')
        manage_mapping_act.triggered.connect(self.manage_mapping)


        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')
        file_menu.addAction(exit_act)
        seq_menu = menubar.addMenu('&Sign and Sequence')
        seq_menu.addAction(rec_new_sign_act)
        seq_menu.addAction(del_sign_act)
        seq_menu.addAction(manage_seq_act)
        at_menu = menubar.addMenu('&Assistive technology')
        at_menu.addAction(new_mapping_act)
        at_menu.addAction(manage_mapping_act)
        at_menu.addAction(load_mapping_act)

    def plot_init(self):
        self.canvas = Canvas(self, width=self.width//100, height=self.height//100//3, dpi=100, window_len=self.window_len, n_lines=3, y_lims=(-20, 20))
        self.layout.add_widget(self.canvas, 2, 0)

    # def check_table_init(self):
    #     self.seq_list = QListWidget(self)
    #     # self.seq_list.setGeometry(800, 200, 600, 600)
    #     self.seq_list.setObjectName("Sequences")
    #     # self.seq_list.show()
    #     i = 0
    #     for key, val in self.sd.known_seq.sequences_init.items():
    #         item = QListWidgetItem()
    #         item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
    #         if val:
    #             item.setCheckState(Qt.Checked)
    #         else:
    #             item.setCheckState(Qt.Unchecked)
    #         self.seq_list.addItem(item)
    #         self.seq_list.item(i).setText(key)
    #         i += 1
    #     self.seq_list.show()
    #
    #     confusion = self.sd.known_seq.evaluate_confusion()
    #     for i in range(self.seq_list.count()):
    #         self.seq_list.item(i).setForeground(Qt.black)
    #     self.shade_confusion(confusion)
    #     self.seq_list.itemChanged.connect(self.checked_sequences)  # Itemselectioncha
    #     self.layout.add_widget(self.seq_list, 1, 1)
    def check_table_init(self):
        self.seq_list = QTreeWidget(self)
        # print(self.seq_list.font().family())
        self.seq_list.setFont(QFont('MS Shell Dlg 2', 18))
        self.seq_list.setStyleSheet("""QTreeWidget::item {border-top: 1px solid black; border-bottom: 1px solid black;}""")
        # self.seq_list.setGeometry(800, 200, 600, 600)
        self.seq_list.setObjectName("Sequences")
        self.seq_list.setHeaderLabels(['Sequence', 'Command'])
        # self.seq_list.show()
        for key, val in self.sd.known_seq.sequences_init.items():
            item = QTreeWidgetItem(self.seq_list)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            if val:
                item.setCheckState(0, Qt.Checked)
            else:
                item.setCheckState(0, Qt.Unchecked)
            item.setText(0, key)
            item.setText(1, self.sd.command_map.get(key))
            # item.setFont()
            # self.seq_list.addItem(item)
        # self.seq_list.setMinimumWidth(self.seq_list.sizeHintForColumn(0))
        # sp = self.seq_list.sizePolicy()
        # sp.setHorizontalPolicy(QSizePolicy.Maximum)
        # self.seq_list.setSizePolicy
        print(self.seq_list.frameWidth())
        self.seq_list.setColumnWidth(0, 500)
        self.seq_list.show()

        confusion = self.sd.known_seq.evaluate_confusion()
        root = self.seq_list.invisibleRootItem()
        for i in range(root.childCount()):  # self.seq_list.count()
            root.child(i).setForeground(0, Qt.black) # self.seq_list.item(i).setForeground(Qt.black)
        self.shade_confusion(confusion)
        self.seq_list.itemChanged.connect(self.checked_sequences)  # Itemselectioncha
        self.layout.add_widget(self.seq_list, 0, 0, h=2)

    def handle_new_data(self, data):
        self.mySerial.set_command_string(self.sd.handle_new_data(data, self.switch.state))
        self.log(True, self.sd.curr_seq, self.sd.last_command, self.sd.get_reversed_at_map().get(self.sd.last_command))
        self.handle_sign_record()
        self.handle_cs_indications()
        self.handle_check_table()
        # print(self.sd.current_window.shape)
        # self.canvas.plot(self.sd.current_window)

    def handle_check_table(self):
        root = self.seq_list.invisibleRootItem()
        for i in range(root.childCount()):
            if root.child(i).text(1) == self.sd.get_reversed_at_map().get(self.sd.last_command):
                if not root.child(i).isSelected():
                    self.seq_list.itemChanged.disconnect()
                    root.child(i).setSelected(True)
                    for c in range(2):
                        root.child(i).setForeground(c, Qt.red)
                        root.child(i).setBackground(c, Qt.black)
                    self.seq_list.itemChanged.connect(self.checked_sequences)
            else:
                if root.child(i).isSelected():
                    self.seq_list.itemChanged.disconnect()
                    for c in range(2):
                        root.child(i).setSelected(False)
                        root.child(i).setForeground(c, Qt.black)
                        root.child(i).setBackground(c, Qt.white)
                    self.seq_list.itemChanged.connect(self.checked_sequences)

    def create_new_command_mapping(self):
        creator = ManageCommand(parent=self, new=True)
        creator.exec()
        filename = QFileDialog.getSaveFileName(self, 'Save new mapping', '', 'AT command map (*acm);;All Files (*)')
        try:
            with open(filename[0], 'w') as fich:
                fich.write(json.dumps([creator.at, creator.mapping]))
        except FileNotFoundError:
            pass

    def manage_sequence(self):
        self.seq_list.itemChanged.disconnect()
        seq_manager = ManageSequenceDialog(self)
        seq_manager.exec()
        for new_seq in seq_manager.new_seq:
            item = QTreeWidgetItem(self.seq_list)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(0, Qt.Checked)
            item.setText(0, new_seq)
            item.setText(1, '')
            # self.seq_list.addItem(item)
        root = self.seq_list.invisibleRootItem()
        for i in seq_manager.idx_to_del[::-1]:
            root.removeChild(root.child(i))
            # self.seq_list.takeItem(self.seq_list.row(item))

        self.seq_list.itemChanged.connect(self.checked_sequences)

    def load_mapping(self):
        filename = QFileDialog.getOpenFileName(self, 'Load mapping', os.getcwd(), 'AT command map (*acm);;All Files (*)')
        try:
            self.sd.command_map = self.sd.load_mapping(filename[0])
            self.reset_table()
            print(self.cmd_map_file)
        except FileNotFoundError:
            pass

    def manage_mapping(self):
        mapper = ManageCommand(parent=self)
        mapper.exec()
        self.sd.command_map = mapper.mapping.copy()
        self.reset_table()

    def record_new_sign(self):
        text, okPressed = QInputDialog.getText(self, "New Sign", "Sign designation:", QLineEdit.Normal, "")
        if okPressed and text != '':
            print(text)  # TODO: msgbox if re_record
            self.sd.new_target_name = text
            self.sd.set_mode(1)

    def handle_sign_record(self):
        if self.sd.get_mode() == 0:
            self.rec_sign_init = True
        elif self.sd.get_mode() == 1:
            if self.rec_sign_init:
                self.rec_sign_init = False
                self.gui_example_count = 0
                self.text_sign_instruct.show()
                self.text_sign_instruct.append('Recording instructions'.upper())
                self.text_sign_instruct.append(f'Recording signature -> {self.sd.new_target_name}:\n'
                                               f'1) Hold switch while placing yourself in the initial position of the new signature\n'
                                               f'2) Release switch\n'
                                               f'3) Execute new signature\n'
                                               f'4) Repeat as instructed (Teach)\n'
                                               f'\n\tTeach')
            if self.sd.new_fit_done:
                print('self.text_sign_instruct.hide()')
                self.sd.set_mode(0)
                self.rec_sign_init = True
                print('record finished')
                self.sign2num = self.sd.target_id
                self.num2sign = {v: k for k, v in self.sign2num.items()}
                self.text_sign_instruct.setText('')
                self.text_sign_instruct.hide()
                self.textedit_log.append(f"Signature '{self.sd.new_target_name}' recorded correctly")
            else:
                if self.gui_example_count != self.sd.example_count:  # new example don
                    self.gui_example_count = self.sd.example_count
                    self.text_sign_instruct.append(f'\tExample {self.gui_example_count} recorded\n\n'
                                                   f'\tTeach\n')

    def handle_cs_indications(self):
        if self.switch.state or self.sd.unavailable_flag:
            self.lineedit_cs.setStyleSheet("""QLineEdit {border: 2px solid red;}""")
        else:
            self.lineedit_cs.setStyleSheet("""QLineEdit {border: 2px solid green;}""")
        try:
            curr_seq = '-'.join([self.num2sign[sign] for sign in self.sd.curr_seq])
        except KeyError:
            self.sign2num = self.sd.target_id
            self.num2sign = {v: k for k, v in self.sign2num.items()}
            curr_seq = '-'.join([self.num2sign[sign] for sign in self.sd.curr_seq])
        if curr_seq != self.lineedit_cs.text():
            self.lineedit_cs.setText(curr_seq)


    def delete_sign(self):
        items = self.sd.targets_names
        item, okPressed = QInputDialog.getItem(self, "Delete sign", "Sign designation:", items, 0, False)
        if okPressed and item:
            self.sd.delete_sign(item)
        self.cmd_map_file = self.sd.command_map_file
        self.sign2num = self.sd.target_id
        self.num2sign = {v: k for k, v in self.sign2num.items()}
        self.seq_list.close()
        self.__delattr__('seq_list')
        self.check_table_init()
        # self.sd.command_map.pop
        # with open(self.parent().cmd_map_file, 'w') as fich:
        #     fich.write(json.dumps([self.box_at_choice.currentText(), self.mapping]))

    def reset_table(self):
        root = self.seq_list.invisibleRootItem()
        for i in range(root.childCount()):  # self.seq_list.count()
            root.child(i).setText(1, self.sd.command_map.get(root.child(i).text(0)))
        print(self.sd.command_map)

    def shade_confusion(self, confusion):
        root = self.seq_list.invisibleRootItem()
        j = 0
        for confused in confusion:
            field_to_shade = self.item2num(confused)
            for i in field_to_shade:
                try:
                    root.child(i).setForeground(0, self.colors[j])
                except IndexError:
                    j -= 1
                    root.child(i).setForeground(0, self.colors[j])
            j += 1

    def item2num(self, item_list):
        # num_list = []
        # for item_str in item_list:
        #     for i in range(self.seq_list.count()):
        #         if self.seq_list.item(i).text() == item_str:
        #             num_list.append(i)
        #             break
        # return num_list
        num_list = []
        root = self.seq_list.invisibleRootItem()
        for item_str in item_list:
            for i in range(root.childCount()):
                if root.child(i).text(0) == item_str:
                    num_list.append(i)
                    break
        return num_list

    def checked_sequences(self):
        ls = []
        print('-------')
        root = self.seq_list.invisibleRootItem()
        self.seq_list.itemChanged.disconnect()
        for i in range(root.childCount()):
            root.child(i).setForeground(0, Qt.black)
        for i in range(root.childCount()):
            if root.child(i).checkState(0) == Qt.Checked and self.sd.command_map.get(root.child(i).text(0)) is None:
                root.child(i).setCheckState(0, Qt.Unchecked)
            elif root.child(i).checkState(0) == Qt.Checked:
                ls.append(root.child(i).text(0))
        print(ls)
        confusion = self.sd.known_seq.set_sequence(ls)
        print('sign comp', self.sd.known_seq.enabeled_sign, self.sd.known_seq.available_sign)
        # if self.sd.known_seq.enabeled_sign != self.sd.known_seq.available_sign:
        if self.sd.known_seq.sign_is_changed():
            self.sd.refit()
        self.shade_confusion(confusion)
        self.seq_list.itemChanged.connect(self.checked_sequences)
        # ls = []
        # print('-------')
        # self.seq_list.itemChanged.disconnect()
        # for i in range(self.seq_list.count()):
        #     self.seq_list.item(i).setForeground(Qt.black)
        # for i in range(self.seq_list.count()):
        #     if self.seq_list.item(i).checkState() == Qt.Checked:
        #         ls.append(self.seq_list.item(i).text())
        # confusion = self.sd.known_seq.set_sequence(ls)
        # self.shade_confusion(confusion)
        # self.seq_list.itemChanged.connect(self.checked_sequences)

    def log(self, must_differ=False, *kargs):
        max_lines = 10
        # print(kargs)
        text = ', '.join([str(piece) for piece in kargs])
        if must_differ:
            if text != self.textedit_log.toPlainText().split(sep='\n')[-1]:
                self.textedit_log.append(text)
        else:
            self.textedit_log.append(text)
        # self.line_edit_log.setText(self.line_edit_log.text() + '\n\r' + text)

    def closeEvent(self, event):
        # here you can terminate your threads and do other stuff
        self.mySerial.terminate()
        self.switch.terminate()
        print('treads closing')
        self.sd.known_seq.save_init_modif()
        # and afterwards call the closeEvent of the super-class
        print('close event')
        super(QMainWindow, self).closeEvent(event)

class ManageCommand(QDialog):

    def __init__(self, parent=None, new=False):
        super().__init__(parent=parent)
        self.mapping = parent.sd.command_map.copy()
        self.new = new
        self.at_map = self.parent().sd.get_at_map()
        self.setWindowTitle('Command Manger')
        self.setGeometry(200, 200, 1000, 500)
        self.layout = QHBoxLayout(self)
        self.setLayout(self.layout)
        self.association_table_init()
        self.cmd_list_init()
        self.ui_init()
        self.layout.addWidget(self.table, 0)
        self.layout.addWidget(self.cmds, 1)
        self.layout.addWidget(self.confirm_layout, 2)

    def ui_init(self):
        self.confirm_layout = MyGridLayout()
        # self.line_edit_file = QLineEdit
        self.at_label = QLabel('Choose AT')
        self.box_at_choice = QComboBox(self)
        self.box_at_choice.addItem('at_pc_dev')
        self.at = self.box_at_choice.currentText()
        self.box_at_choice.currentTextChanged.connect(self.set_at)
        self.button_cancel = QPushButton('Cancel', self)
        self.button_cancel.clicked.connect(self.close)
        self.button_save = QPushButton('Save', self)
        self.button_save.clicked.connect(self.save_cmd)
        self.confirm_layout.add_widget(self.at_label, 0, 0)
        self.confirm_layout.add_widget(self.box_at_choice, 1, 0)
        self.confirm_layout.add_widget(self.button_save, 2, 0)
        self.confirm_layout.add_widget(self.button_cancel, 3, 0)

    def association_table_init(self):
        self.table = QTableWidget(self)
        seqs = self.parent().sd.known_seq.sequences_init
        n_rows = len(seqs)
        self.n_seqs = n_rows
        self.table.setRowCount(n_rows)
        self.table.setColumnCount(2)
        self.table.setAcceptDrops(True)
        self.table.setDragEnabled(True)
        self.table.setDefaultDropAction(Qt.MoveAction)
        # self.table.setDragDropMode(QAbstractItemView.InternalMove)
        for i, seq in enumerate(seqs.keys()):
            item = QTableWidgetItem()
            item.setText(seq)
            item.setFlags(Qt.ItemIsSelectable)
            self.table.setItem(i, 0, item)
            item = QTableWidgetItem()
            if not self.new:
                try:
                    item.setText(self.parent().sd.command_map[seq])
                except KeyError:
                    pass
            # item.setFlags(Qt.ItemIsEditable | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled | Qt.ItemIsEnabled)
            self.table.setItem(i, 1, item)

    def cmd_list_init(self):
        self.cmds = QListWidget(self)
        for cmd in self.at_map.keys():
            item = QListWidgetItem()
                # item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setText(cmd)
            self.cmds.addItem(item)
        self.cmds.setAcceptDrops(False)
        self.cmds.setDragEnabled(True)

    def save_cmd(self):
        self.mapping = {}
        for i in range(self.n_seqs):
            key, val = self.table.item(i, 0).text(), self.table.item(i, 1)
            if val is not None:
                if val.text() != '' and not val.text().isspace():
                    self.mapping[key] = val.text()
        if not self.new:
            print('save_cmd:', self.mapping)
            with open(self.parent().cmd_map_file, 'w') as fich:
                fich.write(json.dumps([self.box_at_choice.currentText(), self.mapping]))
        self.close()

    def set_at(self):
        self.at = self.box_at_choice.currentText()
        print(self.at)

class ManageSequenceDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # self.layout = MyGridLayout(self)
        # self.setCentralWidget(self.layout)
        self.setWindowTitle('Sequence Manger')
        self.setGeometry(200, 200, 1000, 500)
        self.new_seq = []
        self.idx_to_del = []
        self.layout = QHBoxLayout(self)
        self.setLayout(self.layout)
        self.ui_init()
        self.seq_table_init()
        self.combo_init()
        self.layout.addWidget(self.combo_layout, 0)
        self.layout.addWidget(self.button_add, 1)
        self.layout.addWidget(self.seq_list, 2)
        self.layout.addWidget(self.confirm_layout, 3)

    def ui_init(self):
        self.button_add = QPushButton('Add >>', self)
        self.button_add.clicked.connect(self.add_to_seqs)
        self.confirm_layout = MyGridLayout()
        self.button_del = QPushButton('Delete selected sequences', self)
        self.button_del.clicked.connect(self.delete_sequences)
        self.button_ok = QPushButton('Done', self)
        self.button_ok.clicked.connect(self.close)
        # self.button_cancel = QPushButton('Cancel', self)
        self.confirm_layout.add_widget(self.button_del, 0, 0)
        self.confirm_layout.add_widget(self.button_ok, 1, 0)
        # self.confirm_layout.add_widget(self.button_cancel, 2, 0)

    def seq_table_init(self):
        self.seq_list = QListWidget(self)
        i = 0
        for key, val in self.parent().sd.known_seq.sequences_init.items():
            item = QListWidgetItem()
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            self.seq_list.addItem(item)
            self.seq_list.item(i).setText(key)
            i += 1
        self.seq_list.show()

    def combo_init(self):
        self.combo_layout = MyGridLayout()
        self.signs = [QComboBox(), QComboBox(), QComboBox(), QComboBox()]
        for i, cb in enumerate(self.signs):
            self.combo_layout.add_widget(cb, i, 0)
            cb.addItem('-')
            for sign in self.parent().sd.known_seq.target_id.keys():
                cb.addItem(sign)

    def add_to_seqs(self):
        sign_choose = [cb.currentText() for cb in self.signs]
        sign_choose = [value for value in sign_choose if value != '-']
        if len(sign_choose) > 0:
            self.new_seq.append('-'.join(sign_choose))
            user_confirm = QMessageBox(parent=self)
            user_confirm.setIcon(QMessageBox.Question)
            user_confirm.setWindowTitle("Save this sequence?")
            user_confirm.setText(self.new_seq[-1])
            user_confirm.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            user_confirm.buttonClicked.connect(self.user_confirm_decision)
            user_confirm.exec()

    def user_confirm_decision(self, val):
        # print(user_confirm.text())
        if val.text() == 'OK':
            try:
                self.parent().sd.known_seq.save_new_sequence(self.new_seq[-1])
                item = QListWidgetItem()
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked)
                item.setText(self.new_seq[-1])
                self.seq_list.addItem(item)
            except IndexError:
                self.new_seq = self.new_seq[0:-1]
        else:
            print(val.text())

    def delete_sequences(self):
        seq_to_del = []
        idx_to_del = []
        for i in range(self.seq_list.count()):
            item = self.seq_list.item(i)
            if item.checkState() == Qt.Checked:
                seq_to_del.append(item.text())
                idx_to_del.append(i)
        for i in idx_to_del[::-1]:
            item = self.seq_list.item(i)
            self.seq_list.takeItem(self.seq_list.row(item))
        self.parent().sd.known_seq.delete_sequence(seq_to_del)
        self.idx_to_del = idx_to_del
        self.close()


class MyGridLayout(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.grid = QGridLayout(parent)
        self.setLayout(self.grid)

    def add_widget(self, widget, row, col, h=1, w=1):
        self.grid.addWidget(widget, row, col, h, w)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class Canvas(FigCan):

    def __init__(self, parent=None, width=5, height=5, dpi=100, window_len=100, n_lines=6, y_lims=(-10, 15)):
        fig, self.axes = plt.subplots(2, 1, figsize=(width, height), dpi=dpi)
        # fig = plt.figure(figsize=(width, height), dpi=dpi)
        # self.axes = [fig.add_subplot(121), fig.add_subplot(122)]
        self.axes[0].set_ylim(*y_lims)
        self.axes[1].set_ylim(-10, 10)
        self.plotter = [RealtimePlot(self.axes[0], window_len=window_len, n_lines=n_lines),
                        RealtimePlot(self.axes[1], window_len=window_len, n_lines=n_lines)]
        FigCan.__init__(self, fig)
        self.setParent(parent)
        plt.draw()
        self.axes[0].set_title('Accelerometer readings')
        self.axes[1].set_title('Gyroscope readings')
        self.axes[0].get_xaxis().set_visible(False)
        self.axes[1].get_xaxis().set_visible(False)
        plt.ion()
        self.count = 0


    def plot(self, data):
        # self.plotter.plot(data)
        self.count += 1
        if self.count >= 1:
            self.plotter[0].plot(data[0:3, :])
            self.plotter[1].plot(data[3::, :])
            self.count = 0
            # plt.draw()
            # self.axes[0].draw_artist(self.plotter.lineplot[0][0])



class RealtimePlot:
    def __init__(self, axes, window_len, n_lines):
        self.axes = axes
        self.n_lines = n_lines
        colors = 'rgbcmk'
        self.lineplot = [axes.plot(np.arange(window_len), np.zeros(window_len), c) for c in colors[0:n_lines]]
        # self.lineplot = [axes.plot(np.arange(window_len), np.zeros(window_len), "r-"),
        #                  axes.plot(np.arange(window_len), np.zeros(window_len), "g-"),
        #                  axes.plot(np.arange(window_len), np.zeros(window_len), "b-"),
        #                  axes.plot(np.arange(window_len), np.zeros(window_len), "c-"),
        #                  axes.plot(np.arange(window_len), np.zeros(window_len), "m-"),
        #                  axes.plot(np.arange(window_len), np.zeros(window_len), "k-")]
        self.window_len = window_len
        # self.lineplot, = axes.plot(np.arange(100), np.zeros(100), "r-")
        # self.lineplot2, = axes.plot(np.arange(100), np.zeros(100), "b-")
        # self.lineplot3, = axes.plot(np.arange(100), np.zeros(100), "g-")


    def plot(self, dataPlot):
        x = np.arange(dataPlot.shape[1])
        # self.lineplot[0][0].set_data(x, dataPlot[0, :])
        # self.lineplot[1][0].set_data(x, dataPlot[1, :])
        # self.lineplot[2][0].set_data(x, dataPlot[2, :])

        # print(dataPlot[0, :])
        for lp, dp in zip(self.lineplot, dataPlot):
            lp[0].set_data(x, dp)

        # self.axes.relim()