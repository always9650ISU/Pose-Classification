import sys
import configparser
import os
import time
import numpy as np
import threading
from multiprocessing import Process, Manager, Queue
from ctypes import c_char_p
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QMediaPlaylist
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QImage, QPixmap
# from queue import Queue
from mainWindow import Ui_MainWindow
from func.utils import RealSense_get_frame, classification, show_image_process, Demo_frame, keypoints_detection


class Q_get(QThread):
    out = pyqtSignal(tuple)

    def __init__(self, q, parent=None):
        super(Q_get, self).__init__(parent)
        self.q = q

    def run(self):
        while True:
            data = self.q.get()
            self.out.emit(data)


class Change_Model(QThread):
    out = pyqtSignal(str)

    def __init__(self, mediaPlayer, parent=None):
        super(Change_Model, self).__init__(parent)
        self.mediaPlayer = mediaPlayer

    def run(self):
        filepath = self.mediaPlayer.currentMedia().canonicalUrl().toString()
        filename = os.path.split(filepath)[-1]
        exercise_idx = filename.split('-')[0]
        while True:
            time.sleep(1)
            filepath = self.mediaPlayer.currentMedia().canonicalUrl().toString()
            filename = os.path.split(filepath)[-1]
            if exercise_idx != filename.split('-')[0]:
                exercise_idx = filename.split('-')[0]
                self.out.emit(exercise_idx)


class Listen_Playlist(QThread):
    out = pyqtSignal(int)

    def __init__(self, PlayerList, parent=None):
        super(Listen_Playlist, self).__init__(parent)
        self.PlayerList = PlayerList

    def run(self):
        while True:
            idx = self.PlayerList.currentIndex()
            self.out.emit(idx)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        # uic.loadUi("mainwindow.ui", self)

        self.exercise = ['201',]
        self.DemoScale = 0.8

        self._loadconfig()
        self.get_windows_size()

        # # create windows
        self.setupUi(self)
        self.setupMedia()

        self.resize(self.window_width, self.window_height)
        self.showMaximized()

        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(
            0, 0, self.GUIDemo_width, self.GUIDemo_height))
        self.verticalLayoutWidget.setGeometry(
            QtCore.QRect(self.GUIDemo_width, 0, self.GUICamera_width, self.window_height - 100 - self.GUICamera_height))
        self.horizontalLayoutWidget_3.setGeometry(
            QtCore.QRect(self.GUIDemo_width, self.window_height - 100 - self.GUICamera_height, self.GUICamera_width, self.GUICamera_height))

        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(
            5, self.GUIDemo_height - 75, self.GUIDemo_width - 10, int(self.window_height*0.05)))

        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(5, int(
            self.window_height*0.95 - 100), self.GUIDemo_width - 10, int(self.window_height*0.05)))

        font = QtGui.QFont()
        font.setPointSize(20)
        self.Text.setFont(font)
        self.Next_Target_Motion.setFont(font)
        self.Next_Target_Motion.setStyleSheet("background-color: yellow")

        self.timeSlider.setValue(0)
        self.timeSlider.setMinimum(0)
        self.mediaPlayer.positionChanged.connect(self.get_time)

        self.start_button.clicked.connect(self.mediaPlayer.play)
        self.pause_button.clicked.connect(self.mediaPlayer.pause)
        self.next_button.clicked.connect(self.next_video)
        self.previous_button.clicked.connect(self.previous_video)

        self.start_process()

        self.Camera_frame = Q_get(self.q_show)
        self.predict_text = Q_get(self.q_prediction)
        self.get_cexercise_dix = Change_Model(self.mediaPlayer)
        self.list_Playlist = Listen_Playlist(self.MediaPlaylist)
        self.Camera_frame.start()
        self.predict_text.start()
        self.get_cexercise_dix.start()
        self.list_Playlist.start()
        self.Camera_frame.out.connect(self.update_image)
        self.predict_text.out.connect(self.update_text)
        self.get_cexercise_dix.out.connect(self.Pass_ExerciseIdx)
        self.list_Playlist.out.connect(self.Change_PlayList_Order)

        self.makeConnections()

    def get_windows_size(self):

        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)

        self.window_width = int(sizeObject.width())
        self.window_height = int(sizeObject.height())

        self.GUIDemo_height = int(self.window_height * 0.9)
        self.GUIDemo_width = int(self.window_width * self.DemoScale)
        self.GUICamera_width = int(self.window_width - self.GUIDemo_width)

        self.GUICamera_height = int(
            self.GUICamera_width * self.Camera_height / self.Camera_width)

    def _loadconfig(self):
        config = configparser.ConfigParser()
        config.read('./config.ini')
        self.Camera_height = int(config['Main']["Camera_Height"])
        self.Camera_width = int(config['Main']["Camera_Width"])
        self.Camera_fps = int(config['Main']["Camera_fps"])

    def get_time(self, idx):
        self.timeSlider.setMaximum(self.mediaPlayer.duration())
        self.timeSlider.setValue(idx)

    def setupMedia(self):

        self.videoOutput = self.makeVideoWidget()
        self.mediaPlayer = self.makeMediaPlayer()
        self.MediaPlaylist = QMediaPlaylist()
        self.mediaPlayer.setPlaylist(self.MediaPlaylist)

    def next_video(self):
        self.MediaPlaylist.next()
        self.mediaPlayer.play()

    def previous_video(self):
        self.MediaPlaylist.previous()

        self.mediaPlayer.play()

    @pyqtSlot(int)
    def Change_PlayList_Order(self, idx):
        if idx == -1:
            idx = 0
        self.Demolist.setCurrentRow(idx)

    @pyqtSlot(str)
    def Pass_ExerciseIdx(self, idx):
        self.exercise_idx.value = idx

    @pyqtSlot(tuple)
    def update_text(self, data):
        _, text = data
        next_motion = ''
        if "Next" in text.keys():
            next_motion = f"Next:{text.pop('Next')} \n"

        t = ''
        for key, value in text.items():
            t += f"{key}:{str(value)} \n"

        self.Text.setWordWrap(True)
        # self.Text.setText(next_motion) # Next motion
        self.Text.setText(t)
        self.Next_Target_Motion.setText(next_motion)

    @pyqtSlot(tuple)
    def update_image(self, data):

        _, img = data
        qt_img = self.convert_cv_qt(img)
        self.Camera.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""

        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        q = QtGui.QImage(
            cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        # p = convert_to_Qt_format.scaled(
        #     self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(q)

    def makeMediaPlayer(self):
        mediaPlayer = QMediaPlayer(self)
        mediaPlayer.setVideoOutput(self.videoOutput)
        return mediaPlayer

    def makeVideoWidget(self):
        DemoOutWidget = QVideoWidget(self)
        vbox = QVBoxLayout()
        vbox.addWidget(DemoOutWidget)
        self.Demowidget.setLayout(vbox)
        return DemoOutWidget

    def makeConnections(self):
        self.actionExit.triggered.connect(self.end_process)
        self.actionSelect_file.triggered.connect(self.Select_file)

    def Select_file(self):

        self.filepaths, _ = QFileDialog.getOpenFileNames(
            self,  "Open Video", 'C:/Users/User/Documents/Video_2_frame/Input_Video')

        if self.filepaths == '':
            return

        for filepath in self.filepaths:
            filename = os.path.split(filepath)[-1]
            self.Demolist.addItem(filename)
            content = QMediaContent(QtCore.QUrl(filepath))
            self.MediaPlaylist.addMedia(content)

        idx = self.MediaPlaylist.mediaCount() - len(self.filepaths)
        self.Demolist.setCurrentRow(idx)
        self.MediaPlaylist.setCurrentIndex(idx)
        self.MediaPlaylist.setPlaybackMode(QMediaPlaylist.Loop)
        self.mediaPlayer.play()

    def start_process(self):

        q_camera = Queue()
        self.q_show = Queue()
        self.q_keypoints = Queue()
        self.q_prediction = Queue()
        manager = Manager()
        self.exercise_idx = manager.Value(c_char_p, " ")

        self.p_realsense = Process(target=RealSense_get_frame,
                                   args=(q_camera, ),
                                   kwargs={"resize": (self.GUICamera_width, self.GUICamera_height)})
        self.p_detect = Process(target=keypoints_detection,
                                args=(q_camera, self.q_show, self.q_keypoints))
        self.p_classification = Process(target=classification,
                                        args=(self.q_keypoints,
                                              self.q_prediction, self.exercise),
                                        kwargs={"exercise_idx": self.exercise_idx})

        self.p_realsense.start()
        self.p_detect.start()
        self.p_classification.start()

    def end_process(self):
        self.close()
        self.p_realsense.terminate()
        self.p_classification.terminate()
        self.p_detect.terminate()


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


# ref:https://www.youtube.com/watch?v=mJgR98v3CYE&ab_channel=DuarteCorporationTutoriales
# ref:https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
# QMedia player ref https://blog.csdn.net/m0_46913647/article/details/122078826
