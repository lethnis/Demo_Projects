import cv2
import os
import mediapipe as mp
from utils import process_image

# создаём папку для результатов обработки
os.makedirs('output', exist_ok=True)

# выберем, что будем обрабатывать (image, video, webcam)
mode = 'image'
# указываем путь до изображения или видео
path = 'data/sample image.jpg'

# создаём объект face detection
mp_face_detection = mp.solutions.face_detection

# запускаем обнаружение в контекстном менеджере
with mp_face_detection.FaceDetection(
        min_detection_confidence=0.8,       # минимальный показатель уверенности, что лицо найдено
        model_selection=0                   # выбор модели: 0 для лиц близко к камере, 1 для удалённых лиц
        ) as face_detection:

    # если обрабатываем изображение
    if mode == 'image':

        # считываем изображение
        img = cv2.imread(path)

        # обрабатываем с помощью написанной функции
        img = process_image(img, face_detection)

        # отображаем в окне и ждём нажатия клавиши для продолжения
        cv2.imshow('', img)
        cv2.waitKey(0)

        # сохраняем изображение
        cv2.imwrite('output/blurred_image.jpg', img)

    # если обрабатываем видео
    elif mode == 'video':

        # считываем видео
        cap = cv2.VideoCapture(path)

        # сохраняем частоту кадров в переменную
        fps = cap.get(cv2.CAP_PROP_FPS)

        # считываем кадр
        ret, frame = cap.read()

        # создаём объект для сохранения видео
        output_video = cv2.VideoWriter('output/blurred_video.mp4',          # путь файла
                                       cv2.VideoWriter_fourcc(*'MP4V'),     # кодек
                                       fps,                                 # частота кадров
                                       (frame.shape[1], frame.shape[0]))    # размер видео

        # пока видео не закончилось
        while ret:

            # обрабатываем кадр
            frame = process_image(frame, face_detection)

            # сохраняем кадр в новое видео
            output_video.write(frame)

            # считываем следующий кадр
            ret, frame = cap.read()

        # освобождаем память
        cap.release()
        output_video.release()

    # если обрабатываем видео с вебкамеры
    elif mode == 'webcam':

        # подключаемся к камере
        cap = cv2.VideoCapture(0)

        # считываем кадр
        ret, frame = cap.read()

        while ret:

            # обрабатываем изображение
            frame = process_image(frame, face_detection)

            # отображаем в окне
            cv2.imshow('', frame)

            # считываем следующий кадр
            ret, frame = cap.read()

            # если нажата клавиша 'q' выходим из цикла
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # освобождаем память
        cap.release()
