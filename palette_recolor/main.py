import sys

from PIL import Image, ImageQt

from PyQt5.QtWidgets import (QFileDialog, QWidget, QLabel, QApplication, QDesktopWidget, QPushButton, QHBoxLayout,
                             QVBoxLayout, QColorDialog)
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtCore import Qt, QCoreApplication

from palette import build_palettes
from utils import RGBtoLAB, LABtoRGB


def get_draw_color(color):
    # convert the color space and represent in hex
    color = LABtoRGB(color)
    return f'{round(color[0]):02x}{round(color[1]):02x}{round(color[2]):02x}'


class PaletteOp(QLabel):

    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super(PaletteOp, self).__init__(parent, flags)
        self.color = 'ffffff'
        self.setStyleSheet("background-color: #ffffff")
        self.setMaximumSize(100, 100)
        self.setMinimumSize(100, 100)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        color = QColorDialog.getColor()
        print(color)
        if not color.isValid():
            return
        self.setColor(color)


    def setColor(self, color):
        self.color = color
        self.setStyleSheet(f'background-color: #{get_draw_color(color)}')
        # self.setPixmap(QPixmap.fromImage(ImageQt.ImageQt()))


class PaletteRecoloring(QWidget):

    def __init__(self, k=5, bins=16):
        super(PaletteRecoloring, self).__init__()
        self.image = None
        self.k = k
        self.bins = bins

        self._initUI()
        self.show()

    def _initUI(self):
        self.setGeometry(300, 300, 1600, 600)
        self.setWindowTitle('Palette')
        qr = self.frameGeometry()
        qr.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(qr.topLeft())
        # labels for images
        self.img_label = QLabel(self)
        self.orig_img_label = QLabel(self)
        # labels for palettes
        self.palettes = []
        palettes_layout = QVBoxLayout()
        for i in range(self.k):
            p = PaletteOp(self)
            self.palettes.append(p)
            palettes_layout.addWidget(p)
        # buttons
        load_img_btn = QPushButton('Load Image', self)
        load_img_btn.clicked.connect(self.load_image)
        load_img_btn.resize(load_img_btn.sizeHint())

        save_img_btn = QPushButton('Save Image', self)
        save_img_btn.clicked.connect(self.save_image)
        save_img_btn.resize(save_img_btn.sizeHint())

        quit_btn = QPushButton('Quit', self)
        quit_btn.clicked.connect(QCoreApplication.instance().quit)
        quit_btn.resize(quit_btn.sizeHint())

        # set layouts
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.orig_img_label)
        img_layout.addWidget(self.img_label)
        img_layout.addLayout(palettes_layout)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(load_img_btn)
        btn_layout.addWidget(save_img_btn)
        btn_layout.addWidget(quit_btn)
        app_layout = QVBoxLayout()
        app_layout.addLayout(img_layout)
        app_layout.addLayout(btn_layout)
        self.setLayout(app_layout)

    def load_image(self):
        file_name = QFileDialog.getOpenFileName()[0]
        print(file_name)
        if file_name == '':
            return
        im = Image.open(file_name)
        # im = im.resize((round(im.size[0] * 450 / im.size[1]), round(im.size[1] * 450 / im.size[1])),
        #                       Image.ANTIALIAS)
        img_rgb = im.convert("RGBA")
        self.image = im.convert("RGB")
        self.img_label.setPixmap(QPixmap.fromImage(ImageQt.ImageQt(img_rgb)).scaledToHeight(500))
        # self.orig_img_label.setPixmap(QPixmap.fromImage(ImageQt.ImageQt(img_rgb)).scaledToHeight(500))
        # get the palettes based on knn
        palettes_colors = build_palettes(self.image, self.k, self.bins)
        for palette, color in zip(self.palettes, palettes_colors):
            palette.setColor(color)
        print(palettes_colors)

    def save_image(self):
        file_name = QFileDialog.getSaveFileName()[0]
        print(file_name)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = PaletteRecoloring()

    sys.exit(app.exec_())
