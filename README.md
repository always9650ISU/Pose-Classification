==============
Pose Clasfficiation
==============

A Pyhton 3.7 app for Camera/RealSense using Mediapipe pose SDK.


------------
Installation
------------

The following instructions have been tested on windows 10.

1. Install Python.
2. Downloads the repository into some folder:
3. Change direct to the folder and Install Pyhton modules.
   ```
     cd /{folder}/{folder}/Pose-Classification-main
     pip install -r requestion.txt
   ```
------------
Training 
------------
1. Get frame from video. In command line, Change direct to Pose Classifiction folder and execute 01_Video2frame.py.It will get video to image from Input_Video to Output_frame
```
python 01_Video2frame.py
```
2. Labeling images in Label folder. One label create one folder, the same pose move to the folder.
3.  Data processing. In command line, execute 02_data_process.py.
```
python 02_data_process.py
```
4. Train model. execute 03_train.py.
```
python 03_train.py
```

------------
Usage
------------
1. Double click "runUI.bat"(default use Camera.)
2. Click "File" and "Select file" on the menubar and Select video only support mp4 or mov file.
3. Do sports like Demo Video, App will show pose probability and  count pose.

It can stop by ckicking stop button, next button play next video, previous button play previous video.

------------
References
------------
[mediapipe pose](https://google.github.io/mediapipe/solutions/pose.html)
