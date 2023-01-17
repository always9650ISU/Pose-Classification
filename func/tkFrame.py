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
        self.during_time = 0
        # self.success, self.play_time, self.frame = self.q.get()
        self.idx = 0

        self.update_frame()

    def update_frame(self):
        a_time = time.time()
        self.success, self.play_time, self.frame = self.q.get()
        b_time = time.time()
        if self.success:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.frame = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(self.frame))
            self.canvas.create_image(0, 0, image=self.frame, anchor=tk.NW)
        c_time = time.time()
        current = time.time()
        # if self.name == "Demo":
        #     # if self.idx <= 180:
        #     #     print(self.idx, self.name, 1/(current - self.during_time), current -
        #     #           self.during_time, self.q.qsize(), b_time - a_time, c_time - b_time)
        #     elapsed = (current - self.start_time) * 1000
        #     self.delay = max(1, int(self.play_time - elapsed))

        # if self.name == "Camera":

        #     elapsed = (time.time() - self.start_time) * 1000
        #     self.delay = max(15, int(self.play_time - elapsed))
        #     print("Camera", self.name, self.q.qsize(), self.delay)
        if self.player:
            audio_frame, val = self.player.get_frame()

        self.during_time = current

        self.idx += 1
        self.after(self.delay, self.update_frame)
