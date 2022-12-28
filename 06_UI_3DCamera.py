import tkinter as tk
import configparser
import time
import os
import PIL.Image
import PIL.ImageTk
import cv2
from ffpyplayer.player import MediaPlayer
from multiprocessing import Process, Pipe, Queue, Value, SimpleQueue
from func.utils import RealSense_get_frame, classification, show_image_process, Demo_frame, keypoints_detection
from func.tkFrame import tkFrame


class App(object):
    def __init__(self, root):

        # DemoPath = r"./Input_Video/101-頸部活動.MP4"
        # CameraPath = r"./Input_Video/102-踏步抓握.MP4"
        # self.Demo_cap = cv2.VideoCapture(DemoPath)
        # self.FPS = int(self.Demo_cap.get(cv2.CAP_PROP_FPS))
        # self.Camera_cap = cv2.VideoCapture(CameraPath)
        self.exercise = ['201',]
        self.DemoScale = 0.8
        self._loadconfig()
        self.get_widows_size()

        self.root = root
        self.root.title("Motion Classification")
        self.root.attributes('-fullscreen', True)
        self.fullScreenState = False
        # self.root.attributes('-fullscreen', True)
        # self.root.geometry(f"{self.window_width}x{self.window_height}")

        self._loading_DisplayGUI()

        self.root.bind("<F11>", self.toggleFullScreen)
        self.root.bind("<Escape>", self.quitFullScreen)
        self.start_process()
        self.Get_Demo_fps()

        demo = tkFrame(self.root, self.q_demo, player=self.player, width=self.GUIDemo_width,
                       height=self.GUIDemo_height, fps=self.Demo_fps, name="Demo")
        demo.place(x=0, y=0, height=self.GUIDemo_height,
                   width=self.GUIDemo_width)

        Camera = tkFrame(self.root, self.q_show,  width=self.Camera_height,
                         height=self.Camera_width, fps=self.Camera_fps, name="Camera")
        Camera.place(x=self.GUIDemo_width, y=self.window_height - self.GUICamera_height,
                     height=self.GUICamera_height,
                     width=self.GUICamera_width)
        # tkFrame(self.root, self.q_show, height=self.Camera_height, width=self.Camera_width,)
        # self.frameloop()

    def Get_Demo_fps(self):
        for filename in os.listdir("Input_Video"):
            if filename.split("-")[0] == self.exercise[0]:
                demo_src = os.path.join('./Input_Video', filename)
        self.player = MediaPlayer(demo_src)
        cap = cv2.VideoCapture(demo_src)
        self.Demo_fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

    def frameloop(self):
        # audio_frame, val = self.player.get_frame()
        # ret, self.demo = self.Demo_cap.read()
        self.demo = self.q_demo.get()
        self.demo = cv2.resize(
            self.demo, (self.GUIDemo_width, self.GUIDemo_height))
        self.demo = cv2.cvtColor(self.demo, cv2.COLOR_BGR2RGB)

        self.demo = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(self.demo))

        # ret, self.camera = self.Camera_cap.read()
        self.camera = self.q_show.get()
        self.camera = cv2.resize(
            self.camera, (self.GUICamera_width, self.GUICamera_height))
        self.camera = cv2.cvtColor(self.camera, cv2.COLOR_BGR2RGB)
        self.camera = cv2.flip(self.camera, 1)
        self.camera = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(self.camera))

        # self.Demo_canvas.create_image(0, 0, image=self.demo, anchor=tk.NW)
        self.Camera_canvas.create_image(0, 0, image=self.camera, anchor=tk.NW)
        elapsed = (time.time() - self.start_time) * 1000
        # play_time = int(self.Demo_cap.get(cv2.CAP_PROP_POS_MSEC))
        play_time = 30
        sleep = max(1, int(play_time - elapsed))
        self.root.after(sleep, self.frameloop)

    def start_process(self):
        q_camera = Queue()
        self.q_demo = Queue()
        self.q_show = Queue()
        self.q_keypoints = Queue()
        self.q_prediction = Queue()
        stop_sign = Value('i', 0)
        exercise_idx = Value("i", 0)

        self.p_realsense = Process(target=RealSense_get_frame,
                                   args=(q_camera, ),
                                   kwargs={"resize": (self.GUICamera_width, self.GUICamera_height)})
        self.p_demo = Process(target=Demo_frame,
                              args=(self.q_demo, self.exercise, stop_sign,),
                              kwargs={"exercise_idx": exercise_idx, "resize": (self.GUIDemo_width, self.GUIDemo_height)})
        self.p_detect = Process(target=keypoints_detection,
                                args=(q_camera, self.q_show, self.q_keypoints))
        self.p_classification = Process(target=classification,
                                        args=(self.q_keypoints,
                                              self.q_prediction, self.exercise),
                                        kwargs={"exercise_idx": exercise_idx})

        # self.p_show = Process(target=show_image_process,
        #   args=(q_show, stop_sign, ), )

        self.p_realsense.start()
        self.p_demo.start()
        self.p_detect.start()
        self.p_classification.start()
        # self.p_show.start()

    def end_process(self):

        self.p_realsense.terminate()
        self.p_classification.terminate()
        self.p_detect.terminate()
        self.p_demo.terminate()
        # self.p_show.terminate()
        self.root.destroy()

    def get_widows_size(self):
        root = tk.Tk()
        root.update_idletasks()
        root.attributes('-fullscreen', True)
        root.state('iconic')
        self.window_width = int(root.winfo_width())
        self.window_height = int(root.winfo_height())
        root.destroy()

        # self.Demo_height = int(self.window_height * self.DemoScale)
        self.GUIDemo_height = int(self.window_height)
        self.GUIDemo_width = int(self.window_width * self.DemoScale)
        self.GUICamera_width = int(self.window_width - self.GUIDemo_width)

        self.GUICamera_height = int(
            self.GUICamera_width * self.Camera_height / self.Camera_width)
        # self.Camera_height = int(self.window_height - self.Demo_height)

    def _loadconfig(self):

        config = configparser.ConfigParser()
        config.read('./config.ini')
        self.Camera_height = int(config['Main']["Camera_Height"])
        self.Camera_width = int(config['Main']["Camera_Width"])
        self.Camera_fps = int(config['Main']["Camera_fps"])

    def _loading_DisplayGUI(self):

        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar)
        filemenu.add_command(label="Exit", command=self.end_process)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        # self.Demo_canvas = tk.Canvas(
        #     self.root, width=self.GUIDemo_width, height=self.GUIDemo_height)
        # self.Demo_canvas.place(x=0, y=0,
        #                        height=self.GUIDemo_height,
        #                        width=self.GUIDemo_width)
        # self.Camera_canvas = tk.Canvas(
        #     self.root, height=self.Camera_height, width=self.Camera_width)
        # self.Camera_canvas.place(height=self.Camera_height, width=self.Camera_width,
        #                          x=self.GUIDemo_width, y=self.window_height - self.GUICamera_height - 5)

    def toggleFullScreen(self, event):
        self.fullScreenState = not self.fullScreenState
        self.root.attributes("-fullscreen", self.fullScreenState)

    def quitFullScreen(self, event):
        self.fullScreenState = False
        self.root.attributes("-fullscreen", self.fullScreenState)


if __name__ == '__main__':
    root = tk.Tk()
    gui = App(root)
    gui.root.mainloop()
