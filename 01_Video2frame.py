import cv2
import os
import tqdm


def main():
    """
    Slice video from mp4 or Mov to jpg files.
    """
    # Default Input and output path.
    Input_Dir = './Input_Video'
    Save_Dir = './Output_frame'

    # Auto create directory if not exists.
    if not os.path.exists(Input_Dir):
        os.makedirs(Input_Dir)
    if not os.path.exists(Save_Dir):
        os.makedirs(Save_Dir)

    # Slice video files to image. Video file must be endwith "MP4" or "Mov"
    for filename in os.listdir(Input_Dir):

        # Video files support "MP4" or "Mov" files.
        if not filename.endswith('MP4') and not filename.endswith('MOV'):
            continue

        # Create image saveing directory.
        video_path = os.path.join(Input_Dir, filename)
        SaveDir = os.path.join(Save_Dir, filename.split('-')[0])

        if not os.path.exists(SaveDir):
            os.makedirs(SaveDir)

        # Cv2.VideoCapture read video.
        cap = cv2.VideoCapture(video_path)
        video_n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Image name use 'frame counter'.
        frame_counter = 0
        with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:

            while True:
                ret, frame = cap.read()
                # Create image save path
                SavePath = os.path.join(SaveDir, str(frame_counter) + '.jpg')

                # if Video EoF start next one.
                if not ret:
                    print('end')
                    break
                # Save image.
                cv2.imwrite(SavePath, frame)
                frame_counter += 1
                pbar.update()


if __name__ == "__main__":
    main()
