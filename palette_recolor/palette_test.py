import unittest
from PIL import Image

from palette import build_palettes


class MyTestCase(unittest.TestCase):
    def test_something(self):
        path = 'imgs/img2.png'
        image = Image.open(path)
        palettes = build_palettes(image)
        print(palettes)

    def test_recolor(self):
        pass


if __name__ == '__main__':
    unittest.main()
