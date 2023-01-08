import cv2
import numpy as np

video_path = r'Video\м1 _ жёлтыйТ_4.avi'
video = cv2.VideoCapture(video_path)

if not video.isOpened(): print("Hi")

_, frame_pred = video.read()
fl, frame = video.read()
while video.isOpened() and (frame is not None):

    canny_orig = cv2.Canny(frame, 40, 75)
    blured = cv2.GaussianBlur(frame, (21, 21), 0)
    blured_med = cv2.medianBlur(frame, 15)
    edges_gauss = cv2.Canny(blured, 45, 75)
    edges_median = cv2.Canny(blured_med, 45, 75)
    combined = cv2.GaussianBlur(cv2.medianBlur(frame, 13), (5, 5), 0)
    edges_combined = cv2.Canny(combined, 40, 75)
    f_frame = np.fft.fft2(frame)
    f_frame = np.fft.ifftshift(f_frame)


    cv2.imshow("original", frame)
    cv2.imshow("canny_orig", canny_orig)
    cv2.imshow("fourier", np.abs(f_frame))


    cv2.imshow("Gaussian", blured)
    cv2.imshow("Median", blured_med)
    cv2.imshow("Combined", combined)
    cv2.imshow("Gaussian_canny", edges_gauss)
    cv2.imshow("Median_canny", edges_median)
    cv2.imshow("Combined_canny", edges_combined)


    if cv2.waitKey(25) == ord('q'):
        break
    frame_pred = frame
    fl, frame = video.read()

video.release()
cv2.destroyAllWindows()

#Самый простой метод dic
#Написать функцию визуализуализации
#найти другие метрики