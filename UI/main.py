import skrf as rf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from function import COM_function
import numpy as np


matplotlib.use("Qt5Agg")


class MyWindow:
    def __init__(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("Channel Evaluation")
        # MainWindow.resize(1500, 1200)
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()
        MainWindow.resize(int(self.screenwidth*0.9), int(self.screenheight*0.9))


        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setGeometry(QtCore.QRect(0, 0,int(self.screenwidth*0.7), int(self.screenheight*0.7)))
        self.centralwidget.setSizePolicy(sizePolicy)

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0,int(self.screenwidth*0.7), int(self.screenheight*0.7)))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setGeometry(QtCore.QRect(0, 0,int(self.screenwidth*0.7), int(self.screenheight*0.7)))
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setContentsMargins(20,20,20,20)
        #self.gridLayout.setSpacing(10)
        self.tableView_input = QtWidgets.QTableView(self.gridLayoutWidget)
        self.tableView_input.setObjectName("tableView1")
        self.tableView_input.setMaximumWidth(int(self.screenwidth*0.2))
        self.tableView_input.setMaximumHeight(int(self.screenwidth*0.1))

        # self.tableView_input.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # self.tableView_input.setSizePolicy(sizePolicy)
        self.gridLayout.addWidget(self.tableView_input, 0, 0, 1, 1)

        self.tableView_output = QtWidgets.QTableView(self.gridLayoutWidget)
        self.tableView_output.setObjectName("tableView2")
        self.tableView_output.setMaximumWidth(int(self.screenwidth*0.2))  
        self.tableView_output.setMaximumHeight(int(self.screenwidth*0.1))
      
        # self.tableView_output.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tableView_output.setEditTriggers(
            QtWidgets.QTableView.NoEditTriggers)
        # self.tableView_output.setSizePolicy(sizePolicy)
        self.gridLayout.addWidget(self.tableView_output, 1, 0, 1, 1)

        self.pushButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setFixedSize(500,150)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setWeight(30)
        self.pushButton.setFont(font)
        self.gridLayout.addWidget(self.pushButton, 2, 0, 1, 1)
        self.pushButton.setText("Compute Channel Params")

        self.txt = QtWidgets.QLabel(self.gridLayoutWidget)
        self.txt.setObjectName("label1")
        self.gridLayout.addWidget(self.txt,3,0,1,1)
        self.txt.setText("--目前仅支持thru/FEXT/NEXT在同一个s参数里的模式"+"\n"+"--暂时无异常处理."+'\n'+"--port order：p1,n1,p2,n2"+ "\n"+"--Fext,Next 填写在s参数中对应的port number，四个一组以‘;’间隔")
        self.txt.setFont(font)




        self.figure_init()
        self.tableView_init()

        self.pushButton.clicked.connect(self.compute_channel_params)

        mainWindow.show()

    def figure_init(self):
                # add figure for ILD
        self.figure1 = plt.figure()
        self.canvas_ILfitted = self.figure1.add_subplot(111)

        self.figure2 = plt.figure()
        self.canvas_ICR = self.figure2.add_subplot(111)

        self.figure3 = plt.figure()        
        self.canvas_ILD = self.figure3.add_subplot(111)

        self.figure4 = plt.figure()        
        self.canvas_impulse = self.figure4.add_subplot(111)

        self.canvas1 = FigureCanvas(self.figure1)
        self.gridLayout.addWidget(self.canvas1,0,1,1,1)
        self.canvas2 = FigureCanvas(self.figure2)
        self.gridLayout.addWidget(self.canvas2,1,1,1,1)
        self.canvas3 = FigureCanvas(self.figure3)
        self.gridLayout.addWidget(self.canvas3,2,1,1,1)
        self.canvas4 = FigureCanvas(self.figure4)
        self.gridLayout.addWidget(self.canvas4,3,1,1,1)

    def tableView_init(self):

        self.model_input = QStandardItemModel(7, 2)
        self.model_input.setHorizontalHeaderLabels(['Input Item', 'Value'])

        self.model_output = QStandardItemModel(3, 2)
        self.model_output.setHorizontalHeaderLabels(['Output Item', 'Value'])

        self.items_input = [
            'Data Rate(GBand)', 'A_nt', 'A_ft', 'Thru', 'Fext', 'Next', 'F/N type']
        self.items_output = ['Fom ILD(dB)', 'ICN(mV)', 'COM']
        #test_com = '2.3'

        self.input_defaultValue = ['53.125', '3', '1.5', '1, 5, 31, 32', '31, 32, 2, 6;31, 32,3, 7;31, 32, 4, 8',
                                   '31, 32, 30, 29;31, 32, 22, 21;31, 32, 28, 27;31, 32, 20, 19', '1']
        for i in range(len(self.items_input)):
            self.model_input.setItem(i, 0, QStandardItem(self.items_input[i]))
            self.model_input.setItem(
                i, 1, QStandardItem(self.input_defaultValue[i]))

        for j in range(len(self.items_output)):
            self.model_output.setItem(
                j, 0, QStandardItem(self.items_output[j]))
        #self.model_output.setItem(2,1,QStandardItem(test_com))
        self.tableView_input.setModel(self.model_input)
        self.tableView_output.setModel(self.model_output)

    def tableView_output_init(self):

        self.model_output = QStandardItemModel(3, 2)
        self.model_output.setHorizontalHeaderLabels(['Output Item', 'Value'])

        self.items_output = ['Fom ILD(dB)', 'ICN(mV)', 'COM']
        #test_com = '2.3'

        for j in range(len(self.items_output)):
            self.model_output.setItem(
                j, 0, QStandardItem(self.items_output[j]))
        #self.model_output.setItem(2,1,QStandardItem(test_com))
        self.tableView_output.setModel(self.model_output)

    def compute_channel_params(self):
        self.figure_init()
        self.tableView_output_init()
        self.Data_Rate = float(self.model_input.item(0, 1).text())
        self.Ant = float(self.model_input.item(1, 1).text())
        self.Aft = float(self.model_input.item(2, 1).text())
        if self.model_input.item(6, 1).text() == '0':
            fext_file = np.array([])
            next_file = np.array([])
            num_fext = int(self.model_input.item(4, 1).text())
            num_next = int(self.model_input.item(5, 1).text())
            thru_file = QFileDialog.getOpenFileName(
                None, 'import Thru file', 'C:\\', "s4p files(*.s4p)")
            for f in range(num_fext):
                fext_file = np.append(fext_file, QFileDialog.getOpenFileName(
                    None, 'import fext'+str(f+1), 'C:\\', "s4p files(*.s4p)"))
            for n in range(num_next):
                next_file = np.append(next_file, QFileDialog.getOpenFileName(
                    None, 'import next'+str(n+1), 'C:\\', "s4p files(*.s4p)"))
        else:
            Thru_port_str = self.model_input.item(3, 1).text()
            self.Thru_port = list(map(int, Thru_port_str.split(',')))
            self.Thru_port = [i-1 for i in self.Thru_port]
            Fext_port_str = self.model_input.item(4, 1).text()
            Next_port_str = self.model_input.item(5, 1).text()
            self.Fext_port = list(
                map(int, Fext_port_str.replace(';', ',').split(',')))
            self.Fext_port = [i-1 for i in self.Fext_port]
            
            self.Fext_port = [self.Fext_port[i:i+4]
                              for i in range(0, len(self.Fext_port), 4)]
            
            self.Next_port = list(
                map(int, Next_port_str.replace(';', ',').split(',')))
            self.Next_port = [i-1 for i in self.Next_port]
            self.Next_port = [self.Next_port[i:i+4]
                              for i in range(0, len(self.Next_port), 4)]
            self.init_data_file = QFileDialog.getOpenFileName(
                None, 'import snp files', '..\\data\\', "snp files(*.s*p)")
            init_data = rf.Network(self.init_data_file[0])

            Thru = rf.subnetwork(init_data, self.Thru_port)
            FEXT = [rf.subnetwork(init_data, i) for i in self.Fext_port]
            NEXT = [rf.subnetwork(init_data, i) for i in self.Next_port]

            [self.ILD, self.FOM_ILD,self.IL_fitted, self.ICN, self.ICR,f_num,IL,self.or_impulse,self.impulse] = COM_function.compute_multi(
                Thru, FEXT, NEXT, self.Data_Rate, self.Aft, self.Ant)
            
            self.model_output.setItem(
                0, 1, QStandardItem(str(np.round(self.FOM_ILD, 3))))
            self.model_output.setItem(1, 1, QStandardItem(
                str(np.round(self.ICN*1000, 3))))
    ### plot figure IL_fitted
            self.canvas_ILfitted.plot(Thru.f[0:f_num] / 1e9, 20 * np.log10(abs(self.IL_fitted)), label='IL_fitted')
            self.canvas_ILfitted.plot(Thru.f[0:f_num] / 1e9, 20 * np.log10(abs(IL[0:f_num])), label='IL')
            self.canvas_ILfitted.set_title('FOM_ILD=' + str(np.around(self.FOM_ILD, 3)) + 'dB')
            self.canvas_ILfitted.set_xlim(0, self.Data_Rate)
            self.canvas_ILfitted.set_ylim(np.min(20 * np.log10(abs(IL))), 0)
            self.canvas_ILfitted.set_xlabel('Freq(GHz)')
            self.canvas_ILfitted.set_ylabel('dB')
            self.canvas_ILfitted.legend()
            self.canvas_ILfitted.grid()
            self.canvas1.draw()
    
    ### plot figure ILD
            self.canvas_ILD.plot(Thru.f[0:f_num] / 1e9, self.ILD)
            self.canvas_ILD.set_title('ILD')
            self.canvas_ILD.set_xlim(0, self.Data_Rate)
            self.canvas_ILD.set_ylim(-3,3)
            self.canvas_ILD.set_xlabel('Freq(GHz)')
            self.canvas_ILD.set_ylabel('dB')
            self.canvas_ILD.grid()
            self.canvas3.draw()

    ### plot figure ICR
            ix_start = np.where(np.round(self.ICR,1)==80.0)
            ix_ny = np.where(np.round(Thru.f/1e9,1)==26.6)
            self.canvas_ICR.semilogx(Thru.f/1e9,self.ICR)
            self.canvas_ICR.annotate("ny freq", xy = (Thru.f[ix_ny[0][1]]/1e9,self.ICR[ix_ny[0][1]]), xytext= (Thru.f[ix_ny[0][1]]/1e9,self.ICR[ix_ny[0][1]]+5), arrowprops = dict(arrowstyle="->"))

            self.canvas_ICR.stem(Thru.f[ix_ny[0][1]]/1e9,self.ICR[ix_ny[0][1]],'g',label ='ny freq')
            self.canvas_ICR.set_ylim(0, 80)
            self.canvas_ICR.set_xlim(Thru.f[ix_start]/1e9,100)
            self.canvas_ICR.set_ylabel('dB')
            #plt.xscale('log')
            self.canvas_ICR.set_title("ICR")
            self.canvas_ICR.set_xlabel('Freq')
            self.canvas_ICR.grid()
            self.canvas2.draw()

    ## plot figure impulse
            fmax = Thru.frequency.stop
            time_step = 1 / fmax / 2
            delay = int(1*1e-9/time_step)
            UI = 1 / self.Data_Rate/1e9
            x_time = np.arange(np.size(self.or_impulse)+delay)*time_step*1e9
            self.canvas_impulse.plot(x_time,np.append(np.zeros(delay),self.or_impulse), label='or impulse response')
            self.canvas_impulse.plot(x_time,np.append(np.zeros(delay),self.impulse), label='modified impulse')
            self.canvas_impulse.set_title('Impulse Response(delay 1ns)')
            self.canvas_impulse.set_xlabel('time(ns)')
            self.canvas_impulse.legend()
            self.canvas_impulse.grid()
            self.canvas4.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = QMainWindow()
    #mainWindow.resize(1500, 1200)
    a = MyWindow(mainWindow)

    sys.exit(app.exec_())
