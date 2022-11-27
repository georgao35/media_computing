import sys

from PIL import Image, ImageQt

from PyQt5.QtWidgets import (QFileDialog, QWidget, QLabel, QApplication, QDesktopWidget, QPushButton, QHBoxLayout,
                             QVBoxLayout, QColorDialog)
from PyQt5.QtGui import QMouseEvent, QPixmap, QColor
from PyQt5.QtCore import Qt, QCoreApplication

from palette import build_palettes
from recolor import modify_lumin, image_recolor
from utils import RGBtoLAB, LABtoRGB


def get_draw_color(color):
    # convert the color space and represent in hex
    return f'{round(color[0]):02x}{round(color[1]):02x}{round(color[2]):02x}'


class PaletteOp(QLabel):

    def __init__(self, idx=-1, parent=None, flags=Qt.WindowFlags()):
        # the label class for palettes. colors are stored in RGB tuples
        super(PaletteOp, self).__init__(parent, flags)
        self.color = [255, 255, 255]
        self.palette_idx = idx
        self.setStyleSheet("background-color: #ffffff")
        self.setMaximumSize(100, 100)
        self.setMinimumSize(100, 100)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        color = QColorDialog.getColor(initial=QColor(*self.color))
        if not color.isValid():
            return
        self.setColor(color.getRgb()[:3])
        if isinstance(self.parent(), PaletteRecoloring):
            self.parent().recolor(self.palette_idx, self.color)

    def setColor(self, color):
        self.color = color
        print(self.color)
        self.setStyleSheet(f'background-color: #{get_draw_color(self.color)}')


class PaletteRecoloring(QWidget):

    def __init__(self, k=5, bins=16):
        super(PaletteRecoloring, self).__init__()
        self.modified_img = None
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
        # self.img_label = QLabel(self)
        self.orig_img_label = QLabel(self)
        # labels for palettes
        self.palettes = []
        self.palettes_colors = []
        self.palettes_colors_cur = []
        palettes_layout = QVBoxLayout()
        for i in range(self.k):
            p = PaletteOp(i, self)
            self.palettes_colors.append(p.color)
            self.palettes.append(p)
            palettes_layout.addWidget(p)
        self.palettes_colors_cur[:] = self.palettes_colors[:]
        # buttons
        load_img_btn = QPushButton('Load Image', self)
        load_img_btn.clicked.connect(self.load_image)
        load_img_btn.resize(load_img_btn.sizeHint())

        save_img_btn = QPushButton('Save Image', self)
        save_img_btn.clicked.connect(self.save_image)
        save_img_btn.resize(save_img_btn.sizeHint())

        recolor_img_btn = QPushButton('Recolor', self)
        recolor_img_btn.clicked.connect(self.transfer)
        recolor_img_btn.resize(recolor_img_btn.sizeHint())

        quit_btn = QPushButton('Quit', self)
        quit_btn.clicked.connect(QCoreApplication.instance().quit)
        quit_btn.resize(quit_btn.sizeHint())

        # set layouts
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.orig_img_label)
        # img_layout.addWidget(self.img_label)
        img_layout.addLayout(palettes_layout)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(load_img_btn)
        btn_layout.addWidget(save_img_btn)
        btn_layout.addWidget(recolor_img_btn)
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
        img_rgb = im.resize((round(im.size[0] * 500 / im.size[1]), round(im.size[1] * 500 / im.size[1])),
                            Image.ANTIALIAS)
        img_rgb = img_rgb.convert("RGBA")
        self.image = im.convert("RGB")
        # self.img_label.setPixmap(QPixmap.fromImage(ImageQt.ImageQt(img_rgb)).scaledToHeight(500))
        # self.orig_img_label.setPixmap(QPixmap.fromImage(ImageQt.ImageQt(img_rgb)).scaledToHeight(500))
        # get the palettes based on knn
        palettes_colors = build_palettes(self.image, self.k, self.bins)
        for idx, (palette, color) in enumerate(zip(self.palettes, palettes_colors)):
            palette.setColor(LABtoRGB(color))
            self.palettes_colors[idx] = tuple(color)
            self.palettes_colors_cur[idx] = tuple(color)

    def save_image(self):
        file_name = QFileDialog.getSaveFileName()[0]
        print(file_name)

    def recolor(self, palette_idx, palette_color):
        print(f'recolor palette:{palette_idx} to {palette_color}')
        # change the lumin
        palettes_colors_lab = modify_lumin(self.palettes_colors_cur, palette_idx, palette_color)
        # change the modified palettes' colors
        for palette, color in zip(self.palettes, palettes_colors_lab):
            palette.setColor(LABtoRGB(color))
        self.palettes_colors_cur[:] = palettes_colors_lab[:]

    def transfer(self):
        im = image_recolor(self.image, self.palettes_colors, self.palettes_colors_cur)
        modified_img = im.resize((round(im.size[0] * 500 / im.size[1]), round(im.size[1] * 500 / im.size[1])),
                                 Image.ANTIALIAS)  # change the displayed picture's resolution
        self.modified_img = im
        self.orig_img_label.setPixmap(QPixmap.fromImage(ImageQt.ImageQt(modified_img)).scaledToHeight(500))
        im.save('1.jpg')
        # self.palettes_colors[:] = self.palettes_colors_cur[:]


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = PaletteRecoloring()

    sys.exit(app.exec_())
