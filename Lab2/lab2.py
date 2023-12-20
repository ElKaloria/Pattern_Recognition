import sys

from PIL import Image
import numpy as np


def dict_ideal_image() -> dict:
    """
    Function return dict with ideal images for pattern recognition
    :rtype: dict
    :return image_dict
    """
    image_dict = {}
    for i in range(10):
        with Image.open(f"C:/Users/maksd/PycharmProjects/Pattern_Recognition/Lab2/Big_number/{i}.png") as j:
            j.load()
        image_dict["array_ideal_{0}".format(i)] = np.asarray(j.convert("1"))
        j.close()

    return image_dict


def numbers_recognition(image_dict: dict, image) -> tuple:
    """
    Function return the percent of recognition and the number that was recognized
    :param image_dict:
    :param image:
    :return:
    """
    count_black = 0
    count_same = 0
    res_array = [[], []]
    for k, v in image_dict.items():
        if len(v[0]) != len(image[0]):
            print("Image array are not equal")
            sys.exit()

        for i in range(len(v[0])):
            for j in range(len(v[0])):
                if not image[i][j]:
                    count_black += 1
                if not v[i][j] and not image[i][j]:
                    count_same += 1
        percent = (count_same / count_black) * 100
        res_array[0].append(k)
        res_array[1].append(percent)
        count_same, count_black = 0, 0
    res_percent = max(res_array[1])
    res_index = res_array[1].index(res_percent)

    return res_percent, res_array[0][res_index]


def scan_image(filename: str) -> np.array:
    with Image.open(f"C:/Users/maksd/PycharmProjects/Pattern_Recognition/Lab2/Lab2_numbers/{filename}.png") as img:
        img.load()

    some_number = np.asarray(img.convert("1"))
    return some_number


if __name__ == "__main__":
    image_dict = dict_ideal_image()
    some_number = scan_image("a")

    percent_recognition, number = numbers_recognition(image_dict, some_number)
    if percent_recognition < 55:
        print("This is not a number")
    else:
        print(f"This is the number {number[-1]} with percent of recognition {percent_recognition}")
    # img.show()
