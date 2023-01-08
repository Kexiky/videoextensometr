import cv2
import numpy as np

def comparison2d(img1, img2, n_step=64):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    correl_M = []
    for i in range(0, len(img1)-n_step, n_step):
        row = []
        for j in range(0, len(img1)-n_step, n_step):
            eps = 10e5
            indexes =[]
            for i1 in range(0, len(img2)-n_step, n_step):
                for j1 in range(0, len(img2)-n_step, n_step):
                    k = np.abs(np.matrix.sum(img1[i:i+n_step, j:j+n_step]) - np.matrix.sum(img2[i1:i1+n_step, j1:j1+n_step]))
                    if (k < eps):
                        eps = k
                        indexes = [i1, j1]
            row.append(indexes)
        correl_M.append(row)

    return correl_M


video_path = r'Video\м1 _ жёлтыйТ_4.avi'
video = cv2.VideoCapture(video_path)
n_step = 64

f1, pred_frame = video.read()
f1, frame = video.read()
while video.isOpened():
    comp_m = comparison2d(pred_frame, frame, n_step)
    cv2.imwrite(comp_m[0])
