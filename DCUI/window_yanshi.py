# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\new.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


# import ui_source
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


class CommonHelper:
    def __init__(self):
        pass

    @staticmethod
    def readQss(style):
        with open(style, 'r') as f:
            return f.read()


class UI_Fatigue(object):
    def setupUi(self, UI_Fatigue):
        self.setStyleSheet("background-color:rgb(46, 46, 98)")
        UI_Fatigue.setObjectName("UI_Fatigue")
        UI_Fatigue.setFixedSize(1107, 868)
        # UI_Fatigue.resize(1107, 868)
        # UI_Fatigue.setStyleSheet("background-color:rgb(46, 46, 98)")
        self.frame = QtWidgets.QFrame(UI_Fatigue)
        self.frame.setGeometry(QtCore.QRect(880, 140, 201, 701))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        # self.frame.setStyleSheet("background-color:rgb(255, 0, 0)")

        

        # ---------------------------风格文件读取--------------------------
        funbtn_stylefile = './DCUI/test.qss'
        funbtn_style = CommonHelper.readQss(funbtn_stylefile)

        modebtn_stylefile = './DCUI/b.qss'
        modebtn_style = CommonHelper.readQss(modebtn_stylefile)

        label_stylefile = './DCUI/a.qss'
        label_style = CommonHelper.readQss(label_stylefile)


        labelf_stylefile = './DCUI/c.qss'
        labelf_style = CommonHelper.readQss(labelf_stylefile)

        # ---------------------------风格文件读取--------------------------


        # ---------------------------数据日志输入框--------------------------
        font = QtGui.QFont()
        font.setWeight(12)
        font.setBold(True)
        font.setFamily("Microsoft YaHei")
        font.setPointSize(10)
        self.label_wink = QtWidgets.QLabel(self.frame)
        self.label_wink.setGeometry(QtCore.QRect(5, 280, 190, 40))
        self.label_wink.setFont(font)
        self.label_wink.setText(" 眨眼次数: 0")
        self.label_wink.setStyleSheet(labelf_style)


        self.label_yawn = QtWidgets.QLabel(self.frame)
        self.label_yawn.setGeometry(QtCore.QRect(5, 350, 190, 40))
        self.label_yawn.setFont(font)
        self.label_yawn.setText(" 哈欠次数: 0")
        self.label_yawn.setStyleSheet(labelf_style)


        self.label_yawn_time = QtWidgets.QLabel(self.frame)
        self.label_yawn_time.setGeometry(QtCore.QRect(5, 420, 190, 40))
        self.label_yawn_time.setFont(font)
        self.label_yawn_time.setText(" 哈欠时间: 0")
        self.label_yawn_time.setStyleSheet(labelf_style)


        self.label_nod = QtWidgets.QLabel(self.frame)
        self.label_nod.setGeometry(QtCore.QRect(5, 490, 190, 40))
        self.label_nod.setFont(font)
        self.label_nod.setText(" 点头次数: 0")
        self.label_nod.setStyleSheet(labelf_style)

        self.label_nap_time = QtWidgets.QLabel(self.frame)
        self.label_nap_time.setGeometry(QtCore.QRect(5, 560, 190, 40))
        self.label_nap_time.setFont(font)
        self.label_nap_time.setText(" 打盹时间: 0")
        self.label_nap_time.setStyleSheet(labelf_style)


        
        
        # ---------------------------数据日志输入框--------------------------

        # ---------------------------图片 视频 摄像头模式按钮--------------------------
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(0, 0, 200, 50))
        font = QtGui.QFont()
        font.setBold(True)
        # font.setUnderline(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setStyleSheet(modebtn_style)

        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(0, 100, 200, 50))
        font = QtGui.QFont()
        font.setBold(True)
        # font.setUnderline(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet(modebtn_style)
        self.pushButton_2.setObjectName("pushButton_2")

        self.clsButton = QtWidgets.QPushButton(self.frame)
        # self.clsButton.setGeometry(QtCore.QRect(5, 615, 190, 50))
        self.clsButton.setGeometry(QtCore.QRect(0, 200, 200, 50))

        font = QtGui.QFont()
        font.setBold(True)
        # font.setUnderline(True)
        font.setWeight(75)
        self.clsButton.setFont(font)
        self.clsButton.setStyleSheet("QPushButton{height: 50px;background-color: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5, stop:0 #ff3432, stop: 1 #ff3432);\
                                     color: white;\
                                     border-radius: 15px;\
                                     font-family:Microsoft YaHei;\
                                     font-size: 20px;\
                                     font-weight:bold;}\
                                     QPushButton:hover{background-color: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5, stop:0 #ff67b2, stop: 1 #ff67b2);}\
                                     QPushButton:pressed{background-color: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5, stop:0 #ff3432, stop: 1 #ff67b2);}")
        self.clsButton.setObjectName("clsButton")


        # self.pushButton3.setStyleSheet(qssStyle1)

        #self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        #self.pushButton_3.setGeometry(QtCore.QRect(0, 160, 200, 50))

        font = QtGui.QFont()
        font.setBold(True)
        # font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        # self.pushButton_3.setFont(font)
        # self.pushButton_3.setStyleSheet(modebtn_style)

        # self.pushButton_3.setObjectName("pushButton_3")

        # ---------------------------图片 视频 摄像头模式按钮--------------------------

        # ---------------------------切换页面--------------------------

        self.frame_2 = QtWidgets.QFrame(UI_Fatigue)
        self.frame_2.setGeometry(QtCore.QRect(10, 110, 900, 750))
        self.frame_2.setStyleSheet("")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        # self.frame_2.setStyleSheet("background-color:rgb(0, 255, 0)")

        self.show_picture_page = QtWidgets.QStackedWidget(self.frame_2)
        self.show_picture_page.setGeometry(QtCore.QRect(0, 0, 860, 720))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.show_picture_page.setFont(font)
        self.show_picture_page.setObjectName("show_picture_page")

        # ---------------------------切换页面--------------------------

        # ---------------------------视频模式页面--------------------------
        self.photo = QtWidgets.QWidget()
        self.photo.setObjectName("photo")

        self.label_vid = QtWidgets.QLabel(self.photo)
        self.label_vid.setGeometry(QtCore.QRect(50, 30, 800, 600))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(36)
        # font.setBold(True)
        self.label_vid.setFont(font)
        self.label_vid.setText("Video Here")
        # self.label_vid.setObjectName("label")
        self.label_vid.setStyleSheet("border: 4px dashed #FFFFFF;color:white")
        self.label_vid.setAlignment(Qt.AlignCenter)

        self.label_file = QtWidgets.QLabel(self.photo)
        self.label_file.setGeometry(QtCore.QRect(50, 650, 500, 40))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_file.setFont(font)
        self.label_file.setText(" 文件目录: ")
        self.label_file.setObjectName("label")
        # self.label_file.setStyleSheet("border: 2px dashed #FFFFff;color:white")
        self.label_file.setStyleSheet(label_style)

        self.lineEdit_vid = QtWidgets.QLineEdit(self.photo)
        self.lineEdit_vid.setGeometry(QtCore.QRect(120, 655, 420, 30))
        self.lineEdit_vid.setObjectName("lineEdit")
        # self.lineEdit_vid.setStyleSheet("background-color:rgb(60, 60, 60);color:white")
        self.lineEdit_vid.setStyleSheet(label_style)

        self.open_vid_btn = QtWidgets.QPushButton(self.photo)
        self.open_vid_btn.setGeometry(QtCore.QRect(580, 650, 120, 40))
        font = QtGui.QFont()
        font.setBold(True)
        # font.setUnderline(True)
        font.setWeight(75)
        self.open_vid_btn.setFont(font)
        # self.open_vid_btn.setStyleSheet("QPushButton{color:#ffffff;border: 2px solid yellow;border-radius:4px;}")
        self.stop_vid_btn = QtWidgets.QPushButton(self.photo)
        self.stop_vid_btn.setGeometry(QtCore.QRect(730, 650, 120, 40))
        font = QtGui.QFont()
        font.setBold(True)
        # font.setUnderline(True)
        self.stop_vid_btn.setFont(font)
        # self.stop_vid_btn.setStyleSheet("QPushButton{color:#ffffff;border: 2px solid yellow;border-radius:4px;}")

        self.open_vid_btn.setStyleSheet(funbtn_style)
        self.stop_vid_btn.setStyleSheet(funbtn_style)

        self.show_picture_page.addWidget(self.photo)

        # ---------------------------视频模式页面--------------------------

        # ---------------------------摄像头模式页面--------------------------
        self.camera = QtWidgets.QWidget()
        self.camera.setObjectName("videos")
        self.label_cam = QtWidgets.QLabel(self.camera)
        self.label_cam.setGeometry(QtCore.QRect(50, 30, 800, 600))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(36)
        self.label_cam.setFont(font)
        self.label_cam.setText("Camera Here")
        self.label_cam.setObjectName("label_cam")
        self.label_cam.setStyleSheet("border: 4px dashed #FFFFFF;color:white")
        self.label_cam.setAlignment(Qt.AlignCenter)

        self.open_cam_btn = QtWidgets.QPushButton(self.camera)
        self.open_cam_btn.setGeometry(QtCore.QRect(580, 650, 120, 40))
        font = QtGui.QFont()
        font.setBold(True)
        # font.setUnderline(True)
        font.setWeight(75)
        self.open_cam_btn.setFont(font)

        self.stop_cam_btn = QtWidgets.QPushButton(self.camera)
        self.stop_cam_btn.setGeometry(QtCore.QRect(730, 650, 120, 40))
        font = QtGui.QFont()
        font.setBold(True)
        # font.setUnderline(True)
        font.setWeight(75)
        self.stop_cam_btn.setFont(font)

        self.open_cam_btn.setStyleSheet(funbtn_style)
        self.stop_cam_btn.setStyleSheet(funbtn_style)
        self.show_picture_page.addWidget(self.camera)
        # ---------------------------摄像头模式页面--------------------------

        self.label_title = QtWidgets.QLabel(UI_Fatigue)
        self.label_title.setGeometry(QtCore.QRect(50, 40, 1030, 71))
        font.setFamily("STXihei")
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(True)
        font.setItalic(False)
        # font.setUnderline(True)
        font.setWeight(75)
        self.label_title.setFont(font)

        # self.label_title.setStyleSheet("color:#FFFFFF")
        self.label_title.setStyleSheet(label_style)
        self.label_title.setAlignment(Qt.AlignCenter)

        # self.label_title.setStyleSheet("Font{background-color:rgb(255, 255, 255);}")
        # self.listView = QtWidgets.QListView(UI_Fatigue)
        # self.listView.setGeometry(QtCore.QRect(-5, 1, 1121, 871))
        # self.listView.setStyleSheet(" \n""background-image: url(:/bg.png);")background-color:blue;
        # self.listView.setStyleSheet("background-color:#1E1E1E")

        # self.listView.setObjectName("listView")
        # self.listView.raise_()
        self.frame.raise_()
        # self.frame_2.raise_()
        self.label_title.raise_()

        self.retranslateUi(UI_Fatigue)
        self.show_picture_page.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(UI_Fatigue)

    def retranslateUi(self, UI_Fatigue):
        _translate = QtCore.QCoreApplication.translate
        UI_Fatigue.setWindowTitle(_translate("UI_Fatigue", "Form"))
        self.pushButton.setText(_translate("UI_Fatigue", "视频模式"))
        self.pushButton_2.setText(_translate("UI_Fatigue", "摄像头实时识别"))
        #self.pushButton_3.setText(_translate("UI_Fatigue", "摄像头实时识别"))
        self.clsButton.setText(_translate("UI_Fatigue", "解除警告"))

        self.open_vid_btn.setText(_translate("UI_Fatigue", "视频检测"))
        self.stop_vid_btn.setText(_translate("UI_Fatigue", "停止检测"))

        self.open_cam_btn.setText(_translate("UI_Fatigue", "打开摄像头"))
        self.stop_cam_btn.setText(_translate("UI_Fatigue", "停止检测"))
        self.label_title.setText(_translate("UI_Fatigue", "疲劳检测"))