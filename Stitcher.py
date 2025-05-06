import cv2
import numpy as np
import glob
import uuid
from tqdm import tqdm


def detect_and_describe(image):
    """ detect_and_describe(image) - возвращает ключевые точки и дескрипторы

            1. Создание дескриптора
                    Описание локальной области вокруг ключевой точки.

            2. Разделение на подрегионы:
                    Окрестность 16×16 пикселей делится на 4×4 субрегиона (всего 16).

            3. Гистограммы градиентов:
                    Для каждого субрегиона строится 8-бинная гистограмма ориентаций (всего 16×8=128 элементов).

            4. Нормализация:
                    Вектор дескриптора нормализуется для устойчивости к изменению освещения. Значения обрезаются до 0.2 для снижения влияния шума.

            Возвращаемые значения

                Kp (KeyPoints):
                    Список объектов cv2. KeyPoint с полями:
                    pt (x, y) — координаты.
                    Size Диаметр значимой области.
                    Angle Ориентация в градусах.
                    Response Сила отклика (чем выше, тем значимее точка).
                    Octave Номер октавы, где обнаружена точка.

                Des (Descriptors):
                    Массив NumPy формы (N, 128), где N — число ключевых точек.
                    Тип данных: np.float32.
                    Пример дескриптора: [0.1, 0.05, ..., 0.2] (нормированные значения). """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp, des = sift.detectAndCompute(gray, None)

    return kp, des


def match_descriptors(des1, des2):
    """ Функция использует метод Brute-Force Matcher
            На вход поступают два дескриптора (локальная область точки) : первый дескриптор с первой картинки, второй - со второй.
            Перебираются все возможные пары дескрипторов, но выбирается только с наименьшим расстоянием между ними.
            """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []

    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)
    return good


def find_homography(kp1, kp2, matches):
    """ find_homography ищет матрицу гомографии ( матрицу 3х3, которая отображает точки
    одного изображения на соответствующие точки другого изображения,
    учитывая поворот, масштаб и т.д. ) """

    if len(matches) < 4:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0)
    return H


def combine_images(img1, img2, H):
    """ Функция объединяет два изображения (img1 и img2) в панораму с
         использованием матрицы гомографии H, которая описывает преобразование между ними. """

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)

    all_corners = np.concatenate((warped_corners,
                                  np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]]).reshape(-1, 1, 2)),
                                 axis=0)

    [x_min, y_min] = np.int32(np.min(all_corners, axis=(0, 1)) - 1)
    [x_max, y_max] = np.int32(np.max(all_corners, axis=(0, 1)) + 1)

    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]])

    warped_img2 = cv2.warpPerspective(img2, translation.dot(H),
                                      (x_max - x_min, y_max - y_min))

    panorama = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    panorama[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1

    mask = warped_img2 > 0
    panorama[mask] = warped_img2[mask]

    return panorama


def stitch_images(images):
    if len(images) < 1:
        return None

    result = images[0]
    for i in tqdm(range(1, len(images)), desc="Stitching images"):
        kp1, des1 = detect_and_describe(result)
        kp2, des2 = detect_and_describe(images[i])

        matches = match_descriptors(des1, des2)
        if len(matches) < 4:
            print(f"Not enough matches for image {i + 1}")
            continue

        H = find_homography(kp1, kp2, matches)
        if H is None:
            print(f"Homography failed for image {i + 1}")
            continue

        H_inv = np.linalg.inv(H)
        result = combine_images(result, images[i], H_inv)

    return result


def stitch_merge(images, group_size=2):
    """Сшивает изображения иерархически, объединяя их группами.

    Параметры:
        images (list): список изображений (numpy arrays)
        group_size (int): размер группы для сшивания (2, 4, 8...)

    Возвращает:
        numpy array: финальная панорама
    """
    if len(images) == 0:
        return None

    # Рекурсивно обрабатываем группы
    while len(images) > 1:
        next_level = []

        # Обрабатываем группы изображений
        for i in tqdm(range(0, len(images), group_size), desc="Merging groups"):
            group = images[i:i + group_size]

            if len(group) == 1:
                next_level.append(group[0])
                continue

            # Сшиваем группу изображений
            panorama = stitch_images(group)

            if panorama is not None:
                next_level.append(panorama)
            else:
                # Если сшивание не удалось, сохраняем исходные изображения
                next_level.extend(group)

        images = next_level

    return images[0] if images else None


if __name__ == "__main__":
    images = []
    for img_path in sorted(glob.glob('images/*.JPG')):
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading: {img_path}")

    if len(images) < 2:
        print("Need at least 2 images")
    else:
        # Выбираем режим сшивания: 2, 4 или 8 изображений за шаг
        group_size = 2  # можно изменить на 4/8

        # Генерируем уникальное имя файла
        unique_filename = f'panorama_merge_{group_size}_{uuid.uuid4().hex}.jpg'

        # Сшиваем с выбранным размером группы
        panorama = stitch_merge(images, group_size)

        if panorama is not None:
            cv2.imwrite(unique_filename, panorama)
            print(f"Panorama saved as {unique_filename}")
        else:
            print("Panorama creation failed")