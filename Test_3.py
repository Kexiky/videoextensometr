import cv2
import numpy as np

video_path = r'Video\м1 _ жёлтыйТ_4.avi'
video = cv2.VideoCapture(video_path)

if not video.isOpened(): print("Hi")

fl, frame_pred = video.read()
fl, frame = video.read()
while video.isOpened() and (frame is not None):


    f_frame_pred = np.fft.fft2(frame_pred)
    f_frame_pred = np.fft.ifftshift(f_frame_pred)


    cv2.imshow("original", frame_pred)
    cv2.imshow("fourier", np.abs(f_frame_pred))

    if cv2.waitKey(25) == ord('q'):
        break
    frame_pred = frame
    fl, frame = video.read()

video.release()
cv2.destroyAllWindows()