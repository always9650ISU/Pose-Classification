import cv2
from ffpyplayer.player import MediaPlayer
import time
import tkinter as tk


filename = r"./Input_Video/101-頸部活動.MP4"

video = cv2.VideoCapture(filename)
FPS = int(video.get(cv2.CAP_PROP_FPS))  # Frames per Sec
cv2.namedWindow('video',cv2.WINDOW_KEEPRATIO) 
cv2.resizeWindow('video', 500,300) 
cv2.moveWindow('video',300,200)


player = MediaPlayer(filename)

val = ''
start_time = time.time()
while val != 'eof':
    

    audio_frame, val = player.get_frame()
    
    if val != 'eof' and audio_frame is not None:
        img, t = audio_frame

    ret, frame = video.read()
    

    if not ret:
        print("End of video")
        break
    elapsed = (time.time() - start_time) * 1000
    play_time = int(video.get(cv2.CAP_PROP_POS_MSEC))
    sleep = max(1, int(play_time - elapsed))
    print('sleep:', sleep)

    if cv2.waitKey(sleep) == 27:
        break
    
    cv2.imshow('video', frame)

video.release()
cv2.destroyAllWindows()
player.close_player()