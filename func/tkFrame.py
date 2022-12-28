import tkinter as tk
import cv2
import PIL.ImageTk
import PIL.Image
import time


class tkFrame(tk.Frame):
    def __init__(self, window, q, player=None, width=None, height=None, fps=30, name=None):
        super().__init__(window)
        self.window = window
        self.width = width
        self.height = height
        self.name = name
        self.player = player
        self.q = q
        self.fps = fps
        self.start_time = time.time()
        self.canvas = tk.Canvas(
            self, width=self.width, height=self.height)
        self.canvas.pack()
        self.delay = int(1000 / fps)

        self.update_frame()

    def update_frame(self):

        success, play_time, self.frame = self.q.get()
        if success:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.frame = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(self.frame))
            self.canvas.create_image(0, 0, image=self.frame, anchor=tk.NW)

        if self.player:
            audio_frame, val = self.player.get_frame()
        else:
            pass
        if self.name == "Demo":
            elapsed = (time.time() - self.start_time) * 1000
            self.delay = max(1, int(play_time - elapsed))
        if self.name == "Camera":
            print(self.name, self.q.qsize(), self.delay)

        self.after(self.delay, self.update_frame)
