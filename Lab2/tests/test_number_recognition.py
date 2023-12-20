from Lab2 import lab2
import pytest


@pytest.mark.parametrize(
    "number_test",
    [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
)
def test_number_recognition(number_test):
    image_dict = lab2.dict_ideal_image()
    some_number = lab2.scan_image(number_test)
    _, recognized_number = lab2.numbers_recognition(image_dict, some_number)
    assert recognized_number[-1] == number_test
