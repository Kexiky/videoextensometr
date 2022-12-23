import cv2
import numpy as np
from scipy import signal

def optical_flow(I1g, I2g, window_size, tau=1e-2):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    w = window_size // 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)


    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0] - w):
        for j in range(w, I1g.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
            # b = ... # get b here
            # A = ... # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            nu = [0, 1] # get velocity here
            u[i, j] = nu[0]
            v[i, j] = nu[1]

    return (u, v)

video_path = r'Video\м1 _ жёлтыйТ_4.avi'
video = cv2.VideoCapture(video_path)

if not video.isOpened(): print("Hi")

fl, frame_pred = video.read()
fl, frame = video.read()
while video.isOpened() and (frame is not None):
    cv2.imshow("original", frame)
    cv2.imshow(optical_flow(cv2.cvtColor(frame_pred, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 20))


    if cv2.waitKey(25) == ord('q'):
        break
    frame_pred = frame
    fl, frame = video.read()

video.release()
cv2.destroyAllWindows()