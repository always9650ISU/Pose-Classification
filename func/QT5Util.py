from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import os
import time


class Q_get(QThread):
    """
    Get tuple from QThead in UI background. 
    """
    out = pyqtSignal(tuple)

    def __init__(self, q, parent=None):
        super(Q_get, self).__init__(parent)
        self.q = q

    def run(self):
        while True:
            # Get data in queue.
            data = self.q.get()
            # Output data be an event and auto processing.
            self.out.emit(data)


class Change_Model(QThread):
    """
    Detect current playing file in background and return exercise_idx.
    It will change model by exercise_idx.
    """
    out = pyqtSignal(str)

    def __init__(self, mediaPlayer, parent=None):
        super(Change_Model, self).__init__(parent)
        self.mediaPlayer = mediaPlayer

    def run(self):
        # Initial playing file path and label.
        filepath = self.mediaPlayer.currentMedia().canonicalUrl().toString()
        filename = os.path.split(filepath)[-1]
        exercise_idx = filename.split('-')[0]
        while True:
            time.sleep(1)
            # Get playing file path.
            filepath = self.mediaPlayer.currentMedia().canonicalUrl().toString()
            # Get exercise label from path.
            filename = os.path.split(filepath)[-1]
            if exercise_idx != filename.split('-')[0]:
                exercise_idx = filename.split('-')[0]
                # Output exercise_idx to chang model.
                self.out.emit(exercise_idx)


class Listen_Playlist(QThread):
    """
    Detect current playing file in background.
    It will change playList menu index.
    """
    out = pyqtSignal(int)

    def __init__(self, PlayerList, parent=None):
        super(Listen_Playlist, self).__init__(parent)
        self.PlayerList = PlayerList

    def run(self):
        while True:
            # Get current index.
            idx = self.PlayerList.currentIndex()
            self.out.emit(idx)
