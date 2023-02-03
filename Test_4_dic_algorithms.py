import cv2
import numpy as np

def comparison2d(img1, img2, n_step=64):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    correl_M = []
    for i in np.arange(n_step, len(img1)- 2 * n_step, n_step):
        row = []
        for j in np.arange(n_step, len(img1)- 2 * n_step, n_step):
            eps = 10e5
            indexes =[]
            for i1 in np.arange(i - n_step, i + 2 * n_step, 2):
                for j1 in np.arange(j - n_step, j + 2 * n_step, 2):
                    sum1 = np.sum(img1[i:i+n_step, j:j+n_step])
                    sum2 = np.sum(img2[i1:i1+n_step, j1:j1 + n_step])
                    k = np.abs(sum1 - sum2)
                    if (k < eps):
                        eps = k
                        indexes = [i1, j1]
            row.append(indexes)
        correl_M.append(row)

    return np.array(correl_M)

def draw_vector_displ(img, vec, n_step):
    new_img = img
    for i in np.arange(vec.shape[0]):
        for j in np.arange(vec.shape[1]):
            start = (i*n_step + n_step//2, j * n_step + n_step//2)
            end = (vec[i, j, 0], vec[i, j, 1])
            color = (0, 0, 255)
            #print(start, end)
            new_img = cv2.arrowedLine(new_img, start, end, color, 4)

    return new_img



video_path = r'Video\м1 _ жёлтыйТ_4.avi'
video = cv2.VideoCapture(video_path)
n_step = 32

f1, pred_frame = video.read()
f1, frame = video.read()
while video.isOpened():
    comp_m = comparison2d(pred_frame, frame, n_step)
    print(comp_m.shape)
    result_frame = draw_vector_displ(pred_frame, comp_m, n_step)
    #print(pred_frame)
    #print(result_frame)
    cv2.imshow("with_vectors", result_frame)
    if cv2.waitKey(25) == ord('q'):
        break
    pred_frame = frame
    f1, frame = video.read()

