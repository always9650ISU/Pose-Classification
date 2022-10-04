import cv2
import os
import tqdm

Input_Dir = './Input_Video' 
Save_Dir = './Output_frame' 

if not os.path.exists(Input_Dir):
    os.makedirs(Input_Dir)
if not os.path.exists(Save_Dir):
    os.makedirs(Save_Dir)

for filename in os.listdir(Input_Dir):
    
    if not filename.endswith('MP4') and not filename.endswith('MOV'):
        continue
    
    video_path = os.path.join(Input_Dir, filename)
    SaveDir = os.path.join(Save_Dir, filename.split('-')[0])

    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
    else:
        print("*",filename)
        continue

    print(filename)
    cap = cv2.VideoCapture(video_path)
    video_n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_counter = 0
    with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
        
        while True:
            ret, frame = cap.read()
            
            SavePath = os.path.join(SaveDir, str(frame_counter) + '.jpg')
            # print(SavePath)

            if not ret:
                print('end')
                break
            
            cv2.imwrite(SavePath ,frame)
            frame_counter += 1
            pbar.update()
            