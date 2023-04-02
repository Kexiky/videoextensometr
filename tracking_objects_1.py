import cv2

# загрузка видео файлов
path_video1 = r'Video\video_2023-02-04_10-06-12.mp4'
path_video2 = r'Video\м1 _ жёлтыйТ_4.avi'
video1 = cv2.VideoCapture(path_video2)
#video2 = cv2.VideoCapture(path_video2)

# извлечение кадров
frames1 = []
frames2 = []
while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video1.read()
    if not ret1:# or not ret2:
        break
    frames1.append(frame1)
    frames2.append(frame2)


    # инициализация объектов для отслеживания
    p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(frames1[0], cv2.COLOR_BGR2GRAY), maxCorners=100, qualityLevel=0.3, minDistance=7)

    # параметры алгоритма Lucas-Kanade
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # отслеживание объектов на видео
    for i in range(len(frames1)):
        # преобразование кадров в оттенки серого
        prev_frame = cv2.cvtColor(frames1[i], cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(frames2[i], cv2.COLOR_BGR2GRAY)

        # вычисление оптического потока с помощью алгоритма Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, p0, None, **lk_params)

        # отображение точек, отслеживаемых алгоритмом
        for j in range(len(p1)):
            if st[j]:
                cv2.circle(frames2[i], (int(p1[j, 0, 0]), int(p1[j, 0, 1])), 5, (0, 0, 255), -1)

        # обновление точек для отслеживания
        p0 = p1.reshape(-1, 1, 2)

        # отображение кадров с отслеживаемыми точками

        cv2.imshow('frame', frames2[i])
        if cv2.waitKey(1) == ord('q'):
            break

# освобождение ресурсов
video1.release()
#video2.release()
cv2.destroyAllWindows()
