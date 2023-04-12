import cv2

path_video1 = r'Video\video_2023-02-04_10-06-12.mp4'
path_video2 = r'Video\м1 _ жёлтыйТ_4.avi'
path_video1_result = r'Video\video_2023-02-04_10-06-12_rez.mp4'
path_video2_result = r'Video\м1 _ жёлтыйТ_4_rez.avi'

# Загрузка видеофайла
cap = cv2.VideoCapture(path_video2)

# получаем разрешение, fps и кодировку исходного видео-файла
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
codec = cv2.VideoWriter_fourcc(*'XVID')
#codec = "".join([chr((int(codec) >> 8 * i) & 0xFF) for i in range(4)])
print(codec)

rez = cv2.VideoWriter(path_video2_result, codec, fps, (width, height))

# Определение алгоритма трекинга
feature_params = dict(maxCorners=10000, qualityLevel=0.2, minDistance=1, blockSize=1)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Чтение первого кадра
ret, prev_frame = cap.read()

# Определение начальных точек для трекинга
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Цикл обработки видео
while True:
    # Чтение следующего кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Расчет оптического потока методом Лукаса-Канаде
    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

    # Отбор точек, для которых трекинг удался
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # Отрисовка треков
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)

    # Отображение результата
    '''cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
'''
    rez.write(frame)

    # Обновление переменных
    prev_gray = gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

# Освобождение ресурсов
cap.release()
rez.release()
cv2.destroyAllWindows()
