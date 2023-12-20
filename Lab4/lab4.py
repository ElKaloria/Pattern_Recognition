"""Цель лабораторной работы: решение задач распознавания и
обработки изображений с помощью одной из открытых библиотек
распознавания.
С помощью выбранной открытой библиотеки распознавания образов
(OpenCV, VXL, AForge.NET, Camellia или любой другой) необходимо
решить следующие задачи:
1. Считать цветное изображение,
2. Конвертировать его в черно-белое,
3. Выполнить пороговое преобразование (все точки с интенсивностью
ниже заданного значения занулить),
4. Придумать собственный фильтр и применить его к изображению,
5. Реализовать поиск границ с помощью одного из известных операторов
(Кенни, Лапласа и т.д.),
6. Выполнить аффинное преобразование изображения (сжатие и поворот
изображения),
7. Выполнить поиск шаблона изображения в другом изображении,
8. На выбор (поиск геометрического объекта, морфология, фильтрация
или что-то свое)."""


import cv2
import numpy as np
import matplotlib.pyplot as plt


def loading_displaying_img(img_name, flag):
    if not flag:
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_name)

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    return img


def threshold_img(img_param, thresh, maxval):
    img = cv2.medianBlur(img_param, 5)
    ret, th1 = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, maxval, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img_param, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def filter_img(img, kernel):
    dst = cv2.filter2D(img, -1, kernel)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()


def canny_edge_detection(img, min_val, max_val):
    edges = cv2.Canny(img, min_val, max_val)
    cv2.imshow("", edges)
    cv2.waitKey(0)


def scaling_and_rotation(img):
    (h, w) = img.shape[:2]
    res = cv2.resize(img, (int(0.5 * w), int(0.5 * h)), interpolation=cv2.INTER_CUBIC)
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, -45, 0.6)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    plt.subplot(121), plt.imshow(res), plt.title('scaling image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(rotated), plt.title('rotated image')
    plt.xticks([]), plt.yticks([])
    plt.show()
    # cv2.imshow("", res)
    # cv2.waitKey(0)


def find_pattern(img, template_name):
    img2 = img.copy()
    template = cv2.imread(template_name, cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    # Apply template Matching
    res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # else take maximum
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img2, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle("cv2.TM_CCOEFF_NORMED")
    plt.show()


def morph_gradient(img):
    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('original image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gradient, cmap='gray')
    plt.title('gradient image'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    # Scan and print images
    img_original = loading_displaying_img('ej.jpg', 1)
    img_gray = loading_displaying_img('ej.jpg', 0)
    # Threshold images
    threshold_img(img_gray, 127, 255)
    # Apply a filter to the image
    array = [[0, 0, 0, 1, 0],
             [1, 1, 0, 0, 1],
             [0, 1, 0, 1, 0],
             [0, 0, 0, 1, 1],
             [1, 1, 1, 1, 1]]
    filter = np.array(array)
    filter_img(img_original, filter)
    # Canny Edge Detection algorithm
    canny_edge_detection(img_gray, 150, 200)
    scaling_and_rotation(img_gray)
    find_pattern(img_gray, "ej_pattern.jpg")
    morph_gradient(img_gray)
