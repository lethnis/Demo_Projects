import cv2


def process_image(img, face_detection):

    """
    Функция принимает на вход изображение и детектор лиц
    и возвращает размытое изображение лица
    """

    # находим абсолютные величины высоты и ширины
    H, W, _ = img.shape

    # конвертируем палитру в RGB, в которой работает детектор лиц
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ищем лица
    out = face_detection.process(img_rgb)

    # если лица найдены
    if out.detections is not None:

        # для каждого найденного лица
        for detection in out.detections:

            # в объекте детекции находим relative bounding box
            bbox = detection.location_data.relative_bounding_box

            # получаем относительные координаты и измерения
            x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            # преобразуем в абсолютные величины
            x, y, w, h = int(x * W), int(y * H), int(w * W), int(h * H)

            # на часть изображения с лицом применяем размытие
            img[y: y + h, x: x + w] = cv2.blur(img[y: y + h, x: x + w], (30, 30))

    # возвращаем всё изображение
    return img