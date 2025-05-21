import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QMessageBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize


class ImageViewer(QMainWindow):
    def __init__(self, image_folder):
        super().__init__()
        self.image_folder = image_folder
        self.image_files = []
        self.current_index = 0
        self.initUI()
        self.load_images()
        if self.image_files:
            self.display_image()
            self.update_status()
        else:
            QMessageBox.warning(self, "警告", f"文件夹 {image_folder} 中没有找到图片文件")

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('图片查看器')
        self.setGeometry(100, 100, 800, 600)

        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建图片显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setText(f"加载文件夹 {self.image_folder} 中的图片...")
        main_layout.addWidget(self.image_label)

        # 创建按钮布局
        button_layout = QHBoxLayout()

        # 上一张按钮
        self.prev_btn = QPushButton('上一张')
        self.prev_btn.clicked.connect(self.prev_image)
        self.prev_btn.setEnabled(False)
        button_layout.addWidget(self.prev_btn)

        # 下一张按钮
        self.next_btn = QPushButton('下一张')
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)
        button_layout.addWidget(self.next_btn)

        # 添加按钮布局到主布局
        main_layout.addLayout(button_layout)

        # 创建状态标签
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label)

    def load_images(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        self.image_files = []
        try:
            for file in os.listdir(self.image_folder):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    self.image_files.append(os.path.join(self.image_folder, file))
            # 按文件名排序
            self.image_files.sort(key=lambda x: x.lower())

            if self.image_files:
                self.prev_btn.setEnabled(True)
                self.next_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片时出错: {str(e)}")

    def display_image(self):
        if 0 <= self.current_index < len(self.image_files):
            try:
                image_path = self.image_files[self.current_index]
                pixmap = QPixmap(image_path)

                # 调整图片大小以适应窗口
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        self.image_label.size() * 0.95,  # 留出一些边距
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.image_label.setPixmap(scaled_pixmap)
                    self.setWindowTitle(f'车牌识别 - {os.path.basename(image_path)}')
                else:
                    self.image_label.setText(f"无法加载图片: {image_path}")
            except Exception as e:
                self.image_label.setText(f"显示图片时出错: {str(e)}")

    def resizeEvent(self, event):
        # 窗口大小改变时重新显示图片
        if self.image_files:
            self.display_image()
        super().resizeEvent(event)

    def prev_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.display_image()
            self.update_status()

    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_image()
            self.update_status()

    def update_status(self):
        if self.image_files:
            self.status_label.setText(f"图片 {self.current_index + 1}/{len(self.image_files)}")


def run_previous_processes():
    """模拟前面需要运行的所有程序"""
    print("开始运行前置处理程序...")
    # 这里应该是所有需要先运行的代码
    # 为了演示，我们只打印一些信息并等待一段时间
    for i in range(5):
        print(f"处理步骤 {i + 1}/5...")
        time.sleep(1)  # 模拟每个步骤需要1秒
    print("前置处理程序全部完成！")


if __name__ == '__main__':
    # 请修改为你要展示图片的文件夹路径
    IMAGE_FOLDER = r"D:\datasets\images\result"

    # 先运行所有前置程序
    run_previous_processes()

    # 然后启动图片查看器
    app = QApplication(sys.argv)

    # 检查文件夹是否存在
    if not os.path.exists(IMAGE_FOLDER):
        print(f"错误: 指定的文件夹 {IMAGE_FOLDER} 不存在")
        sys.exit(1)

    viewer = ImageViewer(IMAGE_FOLDER)
    viewer.show()
    sys.exit(app.exec_())