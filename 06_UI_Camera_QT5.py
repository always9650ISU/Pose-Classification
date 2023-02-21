import sys
import configparser
import os
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
from mainWindow import Ui_MainWindow
from func.utils import classification, show_image_process, Demo_frame, keypoints_detection
from func.Camera import Camera_get_frame
from func.QT5Util import Q_get, Change_Model, Listen_Playlist


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    Load Gui and start process. 
    """

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        # Default Demo Scale which small then window size.
        self.DemoScale = 0.8

        # Loading init config.
        self._loadconfig()
        # Default size.
        self.get_windows_size()
        # Load UI layout.
        self.setupUi(self)
        # Resize app.
        self.resize(self.window_width, self.window_height)
        # full all window.
        self.showMaximized()
        # Set Video Wedget and MediaPlayer.
        self.setupMedia()

        # Set text layout.
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
        # Set text size about pose probability.
        font = QtGui.QFont()
        font.setPointSize(20)
        self.Text.setFont(font)
        # Set text size about next pose.
        font = QtGui.QFont()
        font.setPointSize(40)
        self.Next_Target_Motion.setFont(font)
        self.Next_Target_Motion.setStyleSheet("background-color: yellow")

        # Set time Slider layout and display.
        self.timeSlider.setValue(0)
        self.timeSlider.setMinimum(0)
        self.mediaPlayer.positionChanged.connect(self.get_time)

        # Set bootton event.
        self.start_button.clicked.connect(self.mediaPlayer.play)
        self.pause_button.clicked.connect(self.mediaPlayer.pause)
        self.next_button.clicked.connect(self.next_video)
        self.previous_button.clicked.connect(self.previous_video)

        # Start Camera, detection, classification process and run in the background.
        self.start_process()

        # Set an auto update event. update camera, text model idx, playlist index.
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

        # Set munu bar.
        self.makeConnections()

    def get_windows_size(self):
        """
        Get the screen size and default App, Demo and Camera window size. 
        """
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)

        self.window_width = int(sizeObject.width())
        self.window_height = int(sizeObject.height())

        self.GUIDemo_height = int(self.window_height * 0.9)
        self.GUIDemo_width = int(self.window_width * self.DemoScale)
        self.GUICamera_width = int(self.window_width - self.GUIDemo_width)

        self.GUICamera_height = int(
            self.GUICamera_width * self.Camera_height / self.Camera_width)

    def _loadconfig(self):
        """
        Config the default value.
        """
        # Init config parser.
        config = configparser.ConfigParser()
        config.read('./config.ini')

        # Get Camera params.
        self.Camera_height = int(config['Main']["Camera_Height"])
        self.Camera_width = int(config['Main']["Camera_Width"])
        self.Camera_fps = int(config['Main']["Camera_fps"])

    def get_time(self, idx):
        """
        The Event change time slider.
        """
        self.timeSlider.setMaximum(self.mediaPlayer.duration())
        self.timeSlider.setValue(idx)

    def setupMedia(self):
        """
        Set Video Wedget and MediaPlayer.
        """
        self.videoOutput = self.makeVideoWidget()
        self.mediaPlayer = self.makeMediaPlayer()
        # Set MediaPlaylist.
        self.MediaPlaylist = QMediaPlaylist()
        self.mediaPlayer.setPlaylist(self.MediaPlaylist)

    def makeMediaPlayer(self):
        """
        Set MediaPlayer.
        """
        mediaPlayer = QMediaPlayer(self)
        mediaPlayer.setVideoOutput(self.videoOutput)
        return mediaPlayer

    def makeVideoWidget(self):
        '''
        Set Video Wedget.
        '''
        DemoOutWidget = QVideoWidget(self)
        vbox = QVBoxLayout()
        vbox.addWidget(DemoOutWidget)
        self.Demowidget.setLayout(vbox)
        return DemoOutWidget

    def next_video(self):
        """
        The Event play next video.
        """
        self.MediaPlaylist.next()
        self.mediaPlayer.play()

    def previous_video(self):
        """
        The Event play previous video.
        """
        self.MediaPlaylist.previous()
        self.mediaPlayer.play()

    @pyqtSlot(int)
    def Change_PlayList_Order(self, idx):
        """
        Processing event that set Playlist  idx.
        """
        if idx == -1:
            idx = 0
        self.Demolist.setCurrentRow(idx)

    @pyqtSlot(str)
    def Pass_ExerciseIdx(self, idx):
        """
        Processing event that set exercise idx.
        """
        self.exercise_idx.value = idx

    @pyqtSlot(tuple)
    def update_text(self, data):
        """
        Processing event that show text.
        """
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
        """
        Processing event that show Camera frame.
        """
        _, img = data
        qt_img = self.convert_cv_qt(img)
        self.Camera.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        q = QtGui.QImage(
            cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(q)

    def makeConnections(self):
        """
        Set menu clock event.
        """
        self.actionExit.triggered.connect(self.end_process)
        self.actionSelect_file.triggered.connect(self.Select_file)

    def Select_file(self):
        """
        Select Only Mp4 file files, add into Playlist and play first video.  
        """
        self.filepaths, _ = QFileDialog.getOpenFileNames(
            self,  "Open Video", 'C:/Users/User/Documents/Video_2_frame/Input_Video')

        # If no file select, return empty.
        if self.filepaths == '':
            return

        # Check files endswith "Mp4", then add Playlist.
        for filepath in self.filepaths:
            filename = os.path.split(filepath)[-1]
            if not filename.endswith("MP4"):
                print("It can not support other than mp4 file.")
                continue
            self.Demolist.addItem(filename)
            content = QMediaContent(QtCore.QUrl(filepath))
            self.MediaPlaylist.addMedia(content)

        # Get current idx and play. It support add file more than first times.
        idx = self.MediaPlaylist.mediaCount() - len(self.filepaths)
        self.Demolist.setCurrentRow(idx)
        self.MediaPlaylist.setCurrentIndex(idx)
        # Set Play mode.
        self.MediaPlaylist.setPlaybackMode(QMediaPlaylist.Loop)
        self.mediaPlayer.play()

    def start_process(self):
        '''
        Create mutilporcess in the background. 
        And Create Queue connect each process passing data.
        '''
        q_camera = Queue()
        self.q_show = Queue()
        self.q_keypoints = Queue()
        self.q_prediction = Queue()
        manager = Manager()
        self.exercise_idx = manager.Value(c_char_p, " ")

        self.p_realsense = Process(target=Camera_get_frame,
                                   args=(q_camera, ),
                                   kwargs={"resize": (self.GUICamera_width, self.GUICamera_height)})
        self.p_detect = Process(target=keypoints_detection,
                                args=(q_camera, self.q_show, self.q_keypoints))
        self.p_classification = Process(target=classification,
                                        args=(self.q_keypoints,
                                              self.q_prediction,),
                                        kwargs={"exercise_idx": self.exercise_idx})

        self.p_realsense.start()
        self.p_detect.start()
        self.p_classification.start()

    def end_process(self):
        """
        End process.
        """
        self.close()
        self.p_realsense.terminate()
        self.p_classification.terminate()
        self.p_detect.terminate()


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
