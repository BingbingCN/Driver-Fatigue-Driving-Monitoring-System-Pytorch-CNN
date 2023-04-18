from pathlib import Path
import threading
import time
import sys
import os
import cv2

import PyQt5.QtCore

from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import random
import os
import dlib
from DCUI.window_yanshi import UI_Fatigue
from stopThreading import stop_thread
import interface
import pygame

pygame.mixer.init()


class MainWindow(UI_Fatigue, QMainWindow):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()

        self.setupUi(self)
        self.show_picture_page.setCurrentIndex(0)

        # self.model = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
        self.vid_source = "0"
        # 打盹时间
        self.nap_time = 0
        # 眨眼次数
        self.wink_count = 0
        # 哈欠次数
        self.yawn_count = 0
        # 哈欠时间
        self.yawn_time = 0
        # 瞌睡点头次数
        self.nod_count = 0

        self.stopEvent = threading.Event()
        self.stopEvent.clear()

        self.open_vid_btn.clicked.connect(self.open_mp4)
        self.stop_vid_btn.clicked.connect(self.close_vid)
        self.open_cam_btn.clicked.connect(self.open_camera)
        self.stop_cam_btn.clicked.connect(self.close_vid)

        self.pushButton.clicked.connect(self.show_video)
        self.pushButton_2.clicked.connect(self.show_camera)
        self.clsButton.clicked.connect(interface.take_snapshot)

        self.th3 = threading.Thread(target=self.voice_prompt)
        self.th3.start()

    def show_video(self):
        self.show_picture_page.setCurrentIndex(0)

    def show_camera(self):
        self.show_picture_page.setCurrentIndex(1)

    def reset_vid(self):
        self.stop_vid_btn.setEnabled(True)
        self.stop_cam_btn.setEnabled(True)
        self.label_clear()

        self.lineEdit_vid.clear()
        self.label_vid.setPixmap(QPixmap(""))
        self.label_vid.setText("Video Here")
        self.label_cam.setPixmap(QPixmap(""))
        self.label_cam.setText("Camera Here")
        self.vid_source = 0

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()

    # 关闭页面
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close_vid()
            self.close()
            stop_thread(self.th3)
            event.accept()
        else:
            event.ignore()

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(
            self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.lineEdit_vid.clear()
            self.lineEdit_vid.insert(fileName)
            self.open_vid_btn.setEnabled(False)
            self.stop_vid_btn.setEnabled(True)
            self.vid_source = fileName
            th = threading.Thread(target=self.infer_fatigue)
            th.start()

    def open_camera(self):
        # self.lineEdit_vid.clear()
        # self.lineEdit_vid.insert(fileName)
        self.open_vid_btn.setEnabled(False)
        self.stop_vid_btn.setEnabled(True)
        self.vid_source = 0
        th = threading.Thread(target=self.infer_fatigue)
        th.start()

    def infer_fatigue(self):
        self.label_clear()
        interface.take_snapshot()
        if self.vid_source == 0:
            cap = cv2.VideoCapture(self.vid_source)
            if not cap.isOpened():
                raise ValueError("Camera open failed.")
                return
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    img, self.nap_time, self.wink_count, self.yawn_count, self.yawn_time, self.nod_count = interface.dete_tired(
                        frame)
                    self.show_frame(img, self.label_cam)
                    self.show_label(self.wink_count, self.yawn_count,
                                    self.yawn_time, self.nod_count, self.nap_time)
                if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                    self.stopEvent.clear()
                    self.open_vid_btn.setEnabled(True)
                    self.reset_vid()
                    break
        else:
            cap = cv2.VideoCapture(self.vid_source)
            if not cap.isOpened():
                raise ValueError("Video open failed.")
                return
            while True:
                ret, frame = cap.read()
                if ret:
                    img, self.nap_time, self.wink_count, self.yawn_count, self.yawn_time, self.nod_count = interface.dete_tired(
                        frame)
                    self.show_frame(img, self.label_vid)
                    self.show_label(self.wink_count, self.yawn_count,
                                    self.yawn_time, self.nod_count, self.nap_time)

                if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                    self.stopEvent.clear()
                    self.open_vid_btn.setEnabled(True)
                    self.open_cam_btn.setEnabled(True)

                    self.reset_vid()
                    break

    def voice_prompt(self):
        while True:
            if (self.wink_count >= 50 or self.yawn_count >= 10 or self.nod_count >= 5):
                # print("running")
                pygame.mixer.music.load('DCUI/warning.mp3')
                pygame.mixer.music.play(1)
                # while pygame.mixer.music.get_busy():  # 在音频播放为完成之前不退出程序
                #     pass
            time.sleep(2)

    def show_label(self, wink, yawn, yawn_t, nod, nap_t):
        
        self.label_wink.setText(" 眨眼次数: " + str(wink))
        self.label_yawn.setText(" 哈欠次数: " + str(yawn))
        self.label_yawn_time.setText(" 哈欠时间: " + str(yawn_t))
        self.label_nod.setText(" 点头次数: " + str(nod))
        self.label_nap_time.setText(" 打盹时间: " + str(nap_t))

    def label_clear(self):
        self.wink_count = 0
        self.yawn_count = 0
        self.nod_count = 0
        self.label_wink.setText(" 眨眼次数: 0")
        self.label_yawn.setText(" 哈欠次数: 0")
        self.label_yawn_time.setText(" 哈欠时间: 0")
        self.label_nod.setText(" 点头次数: 0")
        self.label_nap_time.setText(" 打盹时间: 0")

    def show_frame(self, frame, show_area):
        frame = self.myframe_resize(frame)
        # opencv读取的bgr格式图片转换成rgb格式
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _image = QtGui.QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)  # pyqt5转换成自己能放的图片格式
        jpg_out = QtGui.QPixmap(_image)  # 转换成QPixmap
        show_area.setPixmap(jpg_out)  # 设置图片显示

    def myframe_resize(self, frame):
        if frame.shape[1] > frame.shape[0]:
            resize_scale = 800 / frame.shape[1]
            frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        else:
            resize_scale = 600 / frame.shape[0]
            frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        return frame


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
