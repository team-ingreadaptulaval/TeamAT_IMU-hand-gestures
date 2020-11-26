/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QGroupBox *connectedMtwListGroupBox;
    QListWidget *connectedMtwList;
    QCheckBox *pitchToSelectCheckBox;
    QGroupBox *mtwPropertiesGroupBox;
    QLabel *batteryLevelCaptionLabel;
    QLabel *effUpdateRateCaptionLabel;
    QLabel *rssiCaptionLabel;
    QLabel *rssiLabel;
    QLabel *batteryLevelLabel;
    QLabel *effUpdateRateLabel;
    QLabel *rollLabel;
    QLabel *rollCaptionLabel;
    QLabel *yawCaptionLabel;
    QLabel *pitchLabel;
    QLabel *pitchCaptionLabel;
    QLabel *yawLabel;
    QGroupBox *stationPropertiesGroupBox;
    QLabel *updateRateCaptionLabel;
    QComboBox *allowedUpdateRatesComboBox;
    QLabel *channelCaptionLabel;
    QComboBox *channelComboBox;
    QLabel *stationIdCaptionLabel;
    QLabel *stationIdLabel;
    QPushButton *enableButton;
    QPushButton *startMeasurementButton;
    QPushButton *recordingButton;
    QLineEdit *logFilenameEdit;
    QLabel *logFilenameCaptionLabel;
    QProgressBar *flushingProgressBar;
    QLabel *flushingCaptionLabel;
    QGroupBox *loggingGroupBox;
    QTextBrowser *logWindow;
    QPushButton *clearLogPushButton;
    QGroupBox *stateMachineImageGroupBox;
    QLabel *stateDiagramLabel;
    QGroupBox *dockedMtwListGroupBox;
    QListWidget *dockedMtwList;
    QLabel *logoLabel;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(824, 348);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        connectedMtwListGroupBox = new QGroupBox(centralWidget);
        connectedMtwListGroupBox->setObjectName(QStringLiteral("connectedMtwListGroupBox"));
        connectedMtwListGroupBox->setGeometry(QRect(170, 170, 151, 171));
        connectedMtwList = new QListWidget(connectedMtwListGroupBox);
        connectedMtwList->setObjectName(QStringLiteral("connectedMtwList"));
        connectedMtwList->setGeometry(QRect(10, 20, 131, 111));
        connectedMtwList->setSortingEnabled(true);
        pitchToSelectCheckBox = new QCheckBox(connectedMtwListGroupBox);
        pitchToSelectCheckBox->setObjectName(QStringLiteral("pitchToSelectCheckBox"));
        pitchToSelectCheckBox->setGeometry(QRect(10, 140, 101, 20));
        mtwPropertiesGroupBox = new QGroupBox(centralWidget);
        mtwPropertiesGroupBox->setObjectName(QStringLiteral("mtwPropertiesGroupBox"));
        mtwPropertiesGroupBox->setGeometry(QRect(330, 170, 161, 171));
        batteryLevelCaptionLabel = new QLabel(mtwPropertiesGroupBox);
        batteryLevelCaptionLabel->setObjectName(QStringLiteral("batteryLevelCaptionLabel"));
        batteryLevelCaptionLabel->setGeometry(QRect(20, 30, 71, 16));
        batteryLevelCaptionLabel->setScaledContents(true);
        effUpdateRateCaptionLabel = new QLabel(mtwPropertiesGroupBox);
        effUpdateRateCaptionLabel->setObjectName(QStringLiteral("effUpdateRateCaptionLabel"));
        effUpdateRateCaptionLabel->setGeometry(QRect(20, 78, 91, 16));
        rssiCaptionLabel = new QLabel(mtwPropertiesGroupBox);
        rssiCaptionLabel->setObjectName(QStringLiteral("rssiCaptionLabel"));
        rssiCaptionLabel->setEnabled(true);
        rssiCaptionLabel->setGeometry(QRect(20, 54, 46, 16));
        rssiLabel = new QLabel(mtwPropertiesGroupBox);
        rssiLabel->setObjectName(QStringLiteral("rssiLabel"));
        rssiLabel->setGeometry(QRect(100, 54, 50, 13));
        rssiLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        batteryLevelLabel = new QLabel(mtwPropertiesGroupBox);
        batteryLevelLabel->setObjectName(QStringLiteral("batteryLevelLabel"));
        batteryLevelLabel->setGeometry(QRect(100, 30, 50, 16));
        batteryLevelLabel->setLayoutDirection(Qt::LeftToRight);
        batteryLevelLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        effUpdateRateLabel = new QLabel(mtwPropertiesGroupBox);
        effUpdateRateLabel->setObjectName(QStringLiteral("effUpdateRateLabel"));
        effUpdateRateLabel->setGeometry(QRect(100, 78, 50, 16));
        effUpdateRateLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        rollLabel = new QLabel(mtwPropertiesGroupBox);
        rollLabel->setObjectName(QStringLiteral("rollLabel"));
        rollLabel->setGeometry(QRect(79, 102, 71, 20));
        rollLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        rollCaptionLabel = new QLabel(mtwPropertiesGroupBox);
        rollCaptionLabel->setObjectName(QStringLiteral("rollCaptionLabel"));
        rollCaptionLabel->setGeometry(QRect(20, 102, 46, 16));
        yawCaptionLabel = new QLabel(mtwPropertiesGroupBox);
        yawCaptionLabel->setObjectName(QStringLiteral("yawCaptionLabel"));
        yawCaptionLabel->setGeometry(QRect(20, 150, 46, 16));
        pitchLabel = new QLabel(mtwPropertiesGroupBox);
        pitchLabel->setObjectName(QStringLiteral("pitchLabel"));
        pitchLabel->setGeometry(QRect(79, 126, 71, 20));
        pitchLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        pitchCaptionLabel = new QLabel(mtwPropertiesGroupBox);
        pitchCaptionLabel->setObjectName(QStringLiteral("pitchCaptionLabel"));
        pitchCaptionLabel->setGeometry(QRect(20, 126, 46, 16));
        yawLabel = new QLabel(mtwPropertiesGroupBox);
        yawLabel->setObjectName(QStringLiteral("yawLabel"));
        yawLabel->setGeometry(QRect(79, 150, 71, 20));
        yawLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        stationPropertiesGroupBox = new QGroupBox(centralWidget);
        stationPropertiesGroupBox->setObjectName(QStringLiteral("stationPropertiesGroupBox"));
        stationPropertiesGroupBox->setGeometry(QRect(10, 10, 151, 331));
        updateRateCaptionLabel = new QLabel(stationPropertiesGroupBox);
        updateRateCaptionLabel->setObjectName(QStringLiteral("updateRateCaptionLabel"));
        updateRateCaptionLabel->setEnabled(false);
        updateRateCaptionLabel->setGeometry(QRect(10, 120, 61, 20));
        allowedUpdateRatesComboBox = new QComboBox(stationPropertiesGroupBox);
        allowedUpdateRatesComboBox->setObjectName(QStringLiteral("allowedUpdateRatesComboBox"));
        allowedUpdateRatesComboBox->setEnabled(false);
        allowedUpdateRatesComboBox->setGeometry(QRect(80, 120, 61, 22));
        channelCaptionLabel = new QLabel(stationPropertiesGroupBox);
        channelCaptionLabel->setObjectName(QStringLiteral("channelCaptionLabel"));
        channelCaptionLabel->setEnabled(false);
        channelCaptionLabel->setGeometry(QRect(10, 50, 46, 13));
        channelComboBox = new QComboBox(stationPropertiesGroupBox);
        channelComboBox->setObjectName(QStringLiteral("channelComboBox"));
        channelComboBox->setEnabled(false);
        channelComboBox->setGeometry(QRect(90, 41, 51, 31));
        stationIdCaptionLabel = new QLabel(stationPropertiesGroupBox);
        stationIdCaptionLabel->setObjectName(QStringLiteral("stationIdCaptionLabel"));
        stationIdCaptionLabel->setGeometry(QRect(10, 20, 46, 13));
        stationIdLabel = new QLabel(stationPropertiesGroupBox);
        stationIdLabel->setObjectName(QStringLiteral("stationIdLabel"));
        stationIdLabel->setGeometry(QRect(30, 20, 61, 16));
        stationIdLabel->setScaledContents(false);
        enableButton = new QPushButton(stationPropertiesGroupBox);
        enableButton->setObjectName(QStringLiteral("enableButton"));
        enableButton->setEnabled(false);
        enableButton->setGeometry(QRect(10, 80, 131, 31));
        startMeasurementButton = new QPushButton(stationPropertiesGroupBox);
        startMeasurementButton->setObjectName(QStringLiteral("startMeasurementButton"));
        startMeasurementButton->setEnabled(false);
        startMeasurementButton->setGeometry(QRect(10, 150, 131, 31));
        recordingButton = new QPushButton(stationPropertiesGroupBox);
        recordingButton->setObjectName(QStringLiteral("recordingButton"));
        recordingButton->setEnabled(false);
        recordingButton->setGeometry(QRect(10, 220, 131, 31));
        logFilenameEdit = new QLineEdit(stationPropertiesGroupBox);
        logFilenameEdit->setObjectName(QStringLiteral("logFilenameEdit"));
        logFilenameEdit->setEnabled(false);
        logFilenameEdit->setGeometry(QRect(60, 190, 81, 21));
        logFilenameCaptionLabel = new QLabel(stationPropertiesGroupBox);
        logFilenameCaptionLabel->setObjectName(QStringLiteral("logFilenameCaptionLabel"));
        logFilenameCaptionLabel->setEnabled(false);
        logFilenameCaptionLabel->setGeometry(QRect(10, 190, 46, 21));
        flushingProgressBar = new QProgressBar(stationPropertiesGroupBox);
        flushingProgressBar->setObjectName(QStringLiteral("flushingProgressBar"));
        flushingProgressBar->setEnabled(false);
        flushingProgressBar->setGeometry(QRect(60, 260, 81, 23));
        flushingProgressBar->setValue(0);
        flushingProgressBar->setTextVisible(false);
        flushingProgressBar->setInvertedAppearance(true);
        flushingCaptionLabel = new QLabel(stationPropertiesGroupBox);
        flushingCaptionLabel->setObjectName(QStringLiteral("flushingCaptionLabel"));
        flushingCaptionLabel->setEnabled(false);
        flushingCaptionLabel->setGeometry(QRect(10, 260, 46, 20));
        loggingGroupBox = new QGroupBox(centralWidget);
        loggingGroupBox->setObjectName(QStringLiteral("loggingGroupBox"));
        loggingGroupBox->setGeometry(QRect(330, 10, 341, 141));
        logWindow = new QTextBrowser(loggingGroupBox);
        logWindow->setObjectName(QStringLiteral("logWindow"));
        logWindow->setGeometry(QRect(10, 20, 321, 101));
        clearLogPushButton = new QPushButton(loggingGroupBox);
        clearLogPushButton->setObjectName(QStringLiteral("clearLogPushButton"));
        clearLogPushButton->setGeometry(QRect(10, 125, 321, 10));
        stateMachineImageGroupBox = new QGroupBox(centralWidget);
        stateMachineImageGroupBox->setObjectName(QStringLiteral("stateMachineImageGroupBox"));
        stateMachineImageGroupBox->setGeometry(QRect(500, 170, 311, 171));
        stateDiagramLabel = new QLabel(stateMachineImageGroupBox);
        stateDiagramLabel->setObjectName(QStringLiteral("stateDiagramLabel"));
        stateDiagramLabel->setGeometry(QRect(10, 20, 291, 141));
        dockedMtwListGroupBox = new QGroupBox(centralWidget);
        dockedMtwListGroupBox->setObjectName(QStringLiteral("dockedMtwListGroupBox"));
        dockedMtwListGroupBox->setGeometry(QRect(170, 10, 151, 141));
        dockedMtwList = new QListWidget(dockedMtwListGroupBox);
        dockedMtwList->setObjectName(QStringLiteral("dockedMtwList"));
        dockedMtwList->setGeometry(QRect(10, 20, 131, 111));
        dockedMtwList->setSortingEnabled(true);
        logoLabel = new QLabel(centralWidget);
        logoLabel->setObjectName(QStringLiteral("logoLabel"));
        logoLabel->setGeometry(QRect(690, 20, 121, 131));
        MainWindow->setCentralWidget(centralWidget);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Awinda monitor", Q_NULLPTR));
        connectedMtwListGroupBox->setTitle(QApplication::translate("MainWindow", "Connected MTw list (0):", Q_NULLPTR));
        pitchToSelectCheckBox->setText(QApplication::translate("MainWindow", "Pitch to select", Q_NULLPTR));
        mtwPropertiesGroupBox->setTitle(QApplication::translate("MainWindow", "Selected MTw properties:", Q_NULLPTR));
        batteryLevelCaptionLabel->setText(QApplication::translate("MainWindow", "Battery level:", Q_NULLPTR));
        effUpdateRateCaptionLabel->setText(QApplication::translate("MainWindow", "Eff. update rate:", Q_NULLPTR));
        rssiCaptionLabel->setText(QApplication::translate("MainWindow", "RSSI:", Q_NULLPTR));
        rssiLabel->setText(QApplication::translate("MainWindow", "-", Q_NULLPTR));
        batteryLevelLabel->setText(QApplication::translate("MainWindow", "-", Q_NULLPTR));
        effUpdateRateLabel->setText(QApplication::translate("MainWindow", "-", Q_NULLPTR));
        rollLabel->setText(QApplication::translate("MainWindow", "-", Q_NULLPTR));
        rollCaptionLabel->setText(QApplication::translate("MainWindow", "accel_x", Q_NULLPTR));
        yawCaptionLabel->setText(QApplication::translate("MainWindow", "accel_z", Q_NULLPTR));
        pitchLabel->setText(QApplication::translate("MainWindow", "-", Q_NULLPTR));
        pitchCaptionLabel->setText(QApplication::translate("MainWindow", "accel_y", Q_NULLPTR));
        yawLabel->setText(QApplication::translate("MainWindow", "-", Q_NULLPTR));
        stationPropertiesGroupBox->setTitle(QApplication::translate("MainWindow", "Wireless master properties:", Q_NULLPTR));
        updateRateCaptionLabel->setText(QApplication::translate("MainWindow", "Update rate:", Q_NULLPTR));
        channelCaptionLabel->setText(QApplication::translate("MainWindow", "Channel:", Q_NULLPTR));
        channelComboBox->clear();
        channelComboBox->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "11", Q_NULLPTR)
         << QApplication::translate("MainWindow", "12", Q_NULLPTR)
         << QApplication::translate("MainWindow", "13", Q_NULLPTR)
         << QApplication::translate("MainWindow", "14", Q_NULLPTR)
         << QApplication::translate("MainWindow", "15", Q_NULLPTR)
         << QApplication::translate("MainWindow", "16", Q_NULLPTR)
         << QApplication::translate("MainWindow", "17", Q_NULLPTR)
         << QApplication::translate("MainWindow", "18", Q_NULLPTR)
         << QApplication::translate("MainWindow", "19", Q_NULLPTR)
         << QApplication::translate("MainWindow", "20", Q_NULLPTR)
         << QApplication::translate("MainWindow", "21", Q_NULLPTR)
         << QApplication::translate("MainWindow", "22", Q_NULLPTR)
         << QApplication::translate("MainWindow", "23", Q_NULLPTR)
         << QApplication::translate("MainWindow", "24", Q_NULLPTR)
         << QApplication::translate("MainWindow", "25", Q_NULLPTR)
        );
        stationIdCaptionLabel->setText(QApplication::translate("MainWindow", "ID:", Q_NULLPTR));
        stationIdLabel->setText(QApplication::translate("MainWindow", "-", Q_NULLPTR));
        enableButton->setText(QApplication::translate("MainWindow", "Enable", Q_NULLPTR));
        startMeasurementButton->setText(QApplication::translate("MainWindow", "Start Measurement", Q_NULLPTR));
        recordingButton->setText(QApplication::translate("MainWindow", "Start recording", Q_NULLPTR));
        logFilenameEdit->setText(QApplication::translate("MainWindow", "logfile.mtb", Q_NULLPTR));
        logFilenameCaptionLabel->setText(QApplication::translate("MainWindow", "Filename:", Q_NULLPTR));
        flushingCaptionLabel->setText(QApplication::translate("MainWindow", "Flushing:", Q_NULLPTR));
        loggingGroupBox->setTitle(QApplication::translate("MainWindow", "What's going on:", Q_NULLPTR));
        clearLogPushButton->setText(QString());
        stateMachineImageGroupBox->setTitle(QApplication::translate("MainWindow", "State diagram:", Q_NULLPTR));
        stateDiagramLabel->setText(QApplication::translate("MainWindow", "<State diagram>", Q_NULLPTR));
        dockedMtwListGroupBox->setTitle(QApplication::translate("MainWindow", "Docked MTw list (0):", Q_NULLPTR));
        logoLabel->setText(QApplication::translate("MainWindow", "<Logo label>", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
