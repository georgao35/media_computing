import unittest
from PIL import Image

from palette import build_palettes


class MyTestCase(unittest.TestCase):
    def test_something(self):
        path = 'imgs/img2.jpg'
        image = Image.open(path)
        palettes = build_palettes(image)
        print(palettes)


if __name__ == '__main__':
    unittest.main()
