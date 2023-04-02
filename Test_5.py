import cv2
import numpy as np


# загрузка изображений
img1 = cv2.imread(r'images\image1.jpg')
img2 = cv2.imread(r'images\image2.jpg')

images = [img1, img2]
# выполнение калибровки камер
camera_matrix = 10
dist_coeff = np.zeros((4,1))
img_size = (img1.shape[1], img1.shape[0])
world_points = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]]
img_points = None

ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
    [world_points]*len(images),
    img_points,
    img_size,
    camera_matrix,
    dist_coeff
)

print('test')

# поиск ключевых точек на каждом изображении
detector = cv2.SIFT_create()
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# сопоставление ключевых точек на двух изображениях
matcher = cv2.FlannBasedMatcher_create()
matches = matcher.match(des1, des2)

# вычисление фундаментальной матрицы
fundamental_matrix, _ = cv2.findFundamentalMat(
    [kp1[m.queryIdx].pt for m in matches],
    [kp2[m.trainIdx].pt for m in matches],
    cv2.FM_RANSAC)
print(fundamental_matrix)
