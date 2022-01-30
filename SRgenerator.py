# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox,QApplication,QFileDialog
from PyQt5.QtCore import QTimer,QDateTime
from torch import device,FloatTensor,cuda,matmul,load,no_grad,nn#使打包文件体积小一些
import torchvision.transforms.functional as FT
import math
import time
from PIL import Image
import os
from threading import Thread
# 常量
device = device("cuda" if cuda.is_available() else "cpu")
rgb_weights = FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = FloatTensor([0.485, 0.456,
                                   0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = FloatTensor([0.229, 0.224,
                                  0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = FloatTensor(
    [0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = FloatTensor(
    [0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
def convert_image(img, source, target):
    """
    转换图像格式.

    :参数 img: 输入图像
    :参数 source: 数据源格式, 共有3种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
    :参数 target: 数据目标格式, 共5种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
                   (4) 'imagenet-norm' (由imagenet数据集的平均值和方差进行标准化)
                   (5) 'y-channel' (亮度通道Y，采用YCbCr颜色空间, 用于计算PSNR 和 SSIM)
    :返回: 转换后的图像
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'
                      }, "无法转换图像源格式 %s!" % source
    assert target in {
        'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'
    }, "无法转换图像目标格式t %s!" % target

    # 转换图像数据至 [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)   #把一个取值范围是[0,255]的PIL.Image 转换成形状为[C,H,W]的Tensor，取值范围是[0,1.0]

    elif source == '[0, 1]':
        pass  # 已经在[0, 1]范围内无需处理

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # 从 [0, 1] 转换至目标格式
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # 无需处理

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                           rgb_weights) / 255. + 16.

    return img

class ConvolutionalBlock(nn.Module):
    """
    卷积模块,由卷积层, BN归一化层, 激活层构成.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :参数 in_channels: 输入通道数
        :参数 out_channels: 输出通道数
        :参数 kernel_size: 核大小
        :参数 stride: 步长
        :参数 batch_norm: 是否包含BN层
        :参数 activation: 激活层类型; 如果没有则为None
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # 层列表
        layers = list()

        # 1个卷积层
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # 1个BN归一化层
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # 1个激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # 合并层
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        前向传播

        :参数 input: 输入图像集，张量表示，大小为 (N, in_channels, w, h)
        :返回: 输出图像集，张量表示，大小为(N, out_channels, w, h)
        """
        output = self.conv_block(input)

        return output

class ResidualBlock(nn.Module):
    """
    残差模块, 包含两个卷积模块和一个跳连.
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :参数 kernel_size: 核大小
        :参数 n_channels: 输入和输出通道数（由于是ResNet网络，需要做跳连，因此输入和输出通道数是一致的）
        """
        super(ResidualBlock, self).__init__()

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        前向传播.

        :参数 input: 输入图像集，张量表示，大小为 (N, n_channels, w, h)
        :返回: 输出图像集，张量表示，大小为 (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    子像素卷积模块, 包含卷积, 像素清洗和激活层.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :参数 kernel_size: 卷积核大小
        :参数 n_channels: 输入和输出通道数
        :参数 scaling_factor: 放大比例
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # 首先通过卷积将通道数扩展为 scaling factor^2 倍
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # 进行像素清洗，合并相关通道数据
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        # 最后添加激活层
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        前向传播.

        :参数 input: 输入图像数据集，张量表示，大小为(N, n_channels, w, h)
        :返回: 输出图像数据集，张量表示，大小为 (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class SRResNet(nn.Module):
    """
    SRResNet模型
    """
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :参数 large_kernel_size: 第一层卷积和最后一层卷积核大小
        :参数 small_kernel_size: 中间层卷积核大小
        :参数 n_channels: 中间层通道数
        :参数 n_blocks: 残差模块数
        :参数 scaling_factor: 放大比例
        """
        super(SRResNet, self).__init__()

        # 放大比例必须为 2、 4 或 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "放大比例必须为 2、 4 或 8!"

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # 一系列残差模块, 每个残差模块包含一个跳连接
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # 放大通过子像素卷积模块实现, 每个模块放大两倍
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # 最后一个卷积模块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        """
        前向传播.

        :参数 lr_imgs: 低分辨率输入图像集, 张量表示，大小为 (N, 3, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (16, 3, 24, 24)
        residual = output  # (16, 64, 24, 24)
        output = self.residual_blocks(output)  # (16, 64, 24, 24)
        output = self.conv_block2(output)  # (16, 64, 24, 24)
        output = output + residual  # (16, 64, 24, 24)
        output = self.subpixel_convolutional_blocks(output)  # (16, 64, 24 * 4, 24 * 4)
        sr_imgs = self.conv_block3(output)  # (16, 3, 24 * 4, 24 * 4)

        return sr_imgs


class Generator(nn.Module):
    """
    生成器模型，其结构与SRResNet完全一致.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        参数 large_kernel_size：第一层和最后一层卷积核大小
        参数 small_kernel_size：中间层卷积核大小
        参数 n_channels：中间层卷积通道数
        参数 n_blocks: 残差模块数量
        参数 scaling_factor: 放大比例
        """
        super(Generator, self).__init__()
        self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                            n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)

    def forward(self, lr_imgs):
        """
        前向传播.

        参数 lr_imgs: 低精度图像 (N, 3, w, h)
        返回: 超分重建图像 (N, 3, w * scaling factor, h * scaling factor)
        """
        sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return sr_imgs


# 测试图像
imgPath = "img_lowR/img8.png"

# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 4      # 放大比例


# 预训练模型
srgan_checkpoint = "checkpoint_srgan.pth"
srresnet_checkpoint = "checkpoint_srresnet.pth"


class MyWindow(object):
    def setupUi(self, MainWindow):
        self.desktop = QApplication.desktop()
        # 获取显示器分辨率大小
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.height=self.height*2/3
        self.width=self.width*2/3
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.width, self.height)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 150, 550, 450))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 30, 300, 50))
        self.label_2.setObjectName("label_2")
        font = QtGui.QFont()
        font.setFamily("华文行楷")
        font.setPointSize(20)  # 设置字体
        self.label_2.setFont(font)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(800, 30, 300, 50))
        self.label_3.setObjectName("label_3")
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(15)  # 设置字体
        self.label_3.setFont(font)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(300, 30, 500, 50))
        self.label_5.setObjectName("label_5")
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(18)  # 设置字体
        self.label_5.setFont(font)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(650, 90, 500, 50))
        self.label_6.setObjectName("label_6")
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(15)  # 设置字体
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("color:white")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(650, 150, 550, 450))
        self.label_4.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(50, 100, 250, 35))
        self.lineEdit.setObjectName("lineEdit")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(330, 100, 100, 35))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(450, 100, 100, 35))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_min = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_min.setGeometry(QtCore.QRect(self.width - 90, 10, 20, 20))
        self.pushButton_min.setObjectName("pushButton_3")
        self.pushButton_close = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_close.setGeometry(QtCore.QRect(self.width - 55, 10, 20, 20))
        self.pushButton_close.setObjectName("pushButton_3")

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(1000, 39, 160, 35))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(["SRGAN（增强纹理）", "SRResnet（较平滑）"])  # 设置数据源



        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, self.width, self.height))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu.setTitle("选项")
        self.action1=QtWidgets.QAction(MainWindow)
        self.action1.setEnabled(True)#添加新建菜单并设置菜单可用
        self.action1.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.action1.setObjectName("action1")
        self.action1.setText("帮助")  # 设置菜单文本
        self.action1.setToolTip("帮助")  # 设置提示文本
        self.action1.triggered.connect(self.actionAbout)  # 只针对特定项绑定
        self.menu.addAction(self.action1)  # 加入菜单1
        self.menubar.addAction(self.menu.menuAction())  # 将菜单添加
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.QSS()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton_min.clicked.connect(MainWindow.showMinimized)
        self.pushButton_close.clicked.connect(self.queryExit)
        self.pushButton.clicked.connect(self.getinfo)
        self.pushButton_2.clicked.connect(self.multiimg)
        self.timer = QTimer()
        self.timer.timeout.connect(self.showtime)  # 这个通过调用槽函数来刷新时间
        self.timer.start(1000)  # 1s刷新一次

        self.Showimg = Thread(target=self.showimg)
        self.Multi= Thread(target=self.multi)

    def getinfo(self):
        file=QFileDialog()#创建文件对话框
        file.setDirectory("C:/")#设置初始路径为C盘
        if file.exec_():#如果选择了文件
            imgPath=file.selectedFiles()[0]#获取文件
            if imgPath.endswith("png") or imgPath.endswith("jpg"):
                self.statusbar.showMessage("重建中...", 0)
                self.label.setPixmap(QtGui.QPixmap(imgPath))
                self.label_4.setPixmap(QtGui.QPixmap("logo.png"))

                self.lineEdit.setText(imgPath)
                self.label_6.setText("正在重建中...")
                self.Showimg.start()
            else:
                QMessageBox.critical(MainWindow, "错误", "必须选择图片文件\n以 png 或 jpg 为后缀", QMessageBox.Ok)
    def showimg(self):
        start = time.time()
        # 加载图像
        imgPath=self.lineEdit.text()

        imgname=os.path.basename(imgPath)

        img = Image.open(imgPath, mode='r')
        img = img.convert('RGB')
        if max(img.size[0], img.size[1]) > 800:
            re_s = max(img.size[0], img.size[1]) / 800
            img.resize((int(img.size[0] / re_s), int(img.size[1] / re_s)), 1)

        # 图像预处理
        lr_img = convert_image(img, source='pil', target='imagenet-norm')
        lr_img.unsqueeze_(0)


        if self.comboBox.currentIndex() == 0:


            checkpoint = load(srgan_checkpoint)
            generator = Generator(large_kernel_size=large_kernel_size,
                                  small_kernel_size=small_kernel_size,
                                  n_channels=n_channels,
                                  n_blocks=n_blocks,
                                  scaling_factor=scaling_factor)
            # generator = generator.to(device)
            generator.load_state_dict(checkpoint['generator'])

            generator.eval()
            model = generator

            # 记录时间

            # 转移数据至设备
            # lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed

            #模型推理
            with no_grad():
                sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
                sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')

                sr_img.save('result/GAN_'+imgname)
            self.label_4.setPixmap(QtGui.QPixmap('result/GAN_'+imgname))

        else:
            checkpoint = load(srresnet_checkpoint)
            srresnet = SRResNet(large_kernel_size=large_kernel_size,
                                small_kernel_size=small_kernel_size,
                                n_channels=n_channels,
                                n_blocks=n_blocks,
                                scaling_factor=scaling_factor)
            # srresnet = srresnet.to(device)
            srresnet.load_state_dict(checkpoint['model'])
            srresnet.eval()
            model = srresnet

            # 转移数据至设备
            # lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
            # 模型推理
            with no_grad():
                sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
                sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
                sr_img.save('result/Resnet_' + imgname)
                self.label_4.setPixmap(QtGui.QPixmap('result/Resnet_' + imgname))



        self.statusbar.showMessage("重建完毕,  用时:" + str(time.time() - start)+"s,   输出图像在result文件夹下", 0)
        self.label_6.setText("重建完毕,用时:" +str(int(time.time() - start))+"s,结果在result文件夹")

    def multi(self):
        img_path = "img_lowR"
        filename = os.listdir(img_path)

        save_path = "img_superR/"
        start = time.time()

        if self.comboBox.currentIndex() == 0:
            checkpoint = load(srgan_checkpoint)
            generator = Generator(large_kernel_size=large_kernel_size,
                                  small_kernel_size=small_kernel_size,
                                  n_channels=n_channels,
                                  n_blocks=n_blocks,
                                  scaling_factor=scaling_factor)
            # generator = generator.to(device)
            generator.load_state_dict(checkpoint['generator'])

            generator.eval()
            model = generator
            for imgPath in filename:

                if imgPath.endswith("png") or imgPath.endswith("jpg"):
                    # 加载图像
                    img = Image.open(img_path + "/" + imgPath, mode='r')

                    img = img.convert('RGB')
                    if max(img.size[0], img.size[1]) > 800:
                        re_s = max(img.size[0], img.size[1]) / 800
                        img.resize((int(img.size[0] / re_s), int(img.size[1] / re_s)), 1)

                    # 图像预处理
                    lr_img = convert_image(img, source='pil', target='imagenet-norm')
                    lr_img.unsqueeze_(0)

                    with no_grad():
                        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
                        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
                        sr_img.save(save_path + 'GAN_' + imgPath)

        else:
            checkpoint = load(srresnet_checkpoint)
            srresnet = SRResNet(large_kernel_size=large_kernel_size,
                                small_kernel_size=small_kernel_size,
                                n_channels=n_channels,
                                n_blocks=n_blocks,
                                scaling_factor=scaling_factor)
            # srresnet = srresnet.to(device)
            srresnet.load_state_dict(checkpoint['model'])
            srresnet.eval()
            model = srresnet

            for imgPath in filename:
                if imgPath.endswith("png") or imgPath.endswith("jpg"):
                    # 加载图像
                    img = Image.open(img_path + "/" + imgPath, mode='r')
                    img = img.convert('RGB')
                    if max(img.size[0], img.size[1]) > 800:
                        re_s = max(img.size[0], img.size[1]) / 800
                        img.resize((int(img.size[0] / re_s), int(img.size[1] / re_s)), 1)

                    # 图像预处理
                    lr_img = convert_image(img, source='pil', target='imagenet-norm')
                    lr_img.unsqueeze_(0)
                    with no_grad():
                        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
                        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
                        sr_img.save('img_superR/Resnet_' + imgPath)

        self.statusbar.showMessage("重建完毕,  用时:" + str(time.time() - start)+"s,   输出图像在img_superR文件夹下", 0)
        self.label_6.setText("重建完毕,用时:" +str(int(time.time() - start))+"s,结果在img_superR文件夹")
    def multiimg(self):
        self.statusbar.showMessage("重建中，请勿退出...请留意输出文件夹！", 0)
        self.label.setPixmap(QtGui.QPixmap("logo.png"))
        self.label_4.setPixmap(QtGui.QPixmap("logo.png"))
        self.label_6.setText("正在重建中...")
        self.Multi.start()



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "超分辨率重建系统"))

        MainWindow.setWindowFlag(QtCore.Qt.FramelessWindowHint)#无边框
        self.label.setPixmap(QtGui.QPixmap("logo.png"))
        self.label.setScaledContents(True)#图片适应label大小
        self.label_4.setPixmap(QtGui.QPixmap("logo.png"))
        self.label_4.setScaledContents(True)#图片适应label大小
        self.pushButton.setText(_translate("MainWindow", "选择图片"))
        self.pushButton_2.setText(_translate("MainWindow", "批量输出"))
        self.label_2.setText(_translate("MainWindow", "超分辨率重建"))
        self.label_3.setText(_translate("MainWindow", "深度神经网络："))
        time = QDateTime.currentDateTime()  # 获取当前时间
        timedisplay = time.toString("yyyy-MM-dd hh:mm:ss dddd")  # 格式化一下时间
        self.label_5.setText(_translate("MainWindow", timedisplay))
        self.statusbar.showMessage("未开始，选择图片或者批量操作以开始", 0)  # 第二个参数为要显示的时间（毫秒为单位），为0则一直显示
        self.label_6.setText("等待操作...")

    def QSS(self):
        MainWindow.setStyleSheet('''QMainWindow{background-color:#576690;}''')
        self.menubar.setStyleSheet('''QMenuBar{background-color:#495579;color:white;}''')
        self.statusbar.setStyleSheet('''QStatusBar{background-color:#60709f;color:white;}''')
        self.menu.setStyleSheet('''QMenu{background-color:#07021d;color:white;}''')
        self.pushButton_close.setStyleSheet('''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.pushButton_min.setStyleSheet('''QPushButton{background:#FFFF00;border-radius:5px;}QPushButton:hover{background:#FFD700;}''')
        self.pushButton.setStyleSheet('''QPushButton{background:#7FFFD4;border-radius:10px;}QPushButton:hover{background:#00BFFF;}''')
        self.pushButton_2.setStyleSheet('''QPushButton{background:#ADD8E6;border-radius:10px;}QPushButton:hover{background:#8470FF;}''')

        self.comboBox.setStyleSheet('''QComboBox{background:#F5FFFA;border-radius:8px;}''')


    def queryExit(self):
        res = QtWidgets.QMessageBox.question(MainWindow, "关闭", "是否退出系统?",
                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        if res == QtWidgets.QMessageBox.Yes:
            QtCore.QCoreApplication.instance().exit()



    def showtime(self):
        time = QDateTime.currentDateTime()  # 获取当前时间
        timedisplay = time.toString("yyyy-MM-dd hh:mm:ss dddd")  # 格式化一下时间
        self.label_5.setText(timedisplay)
    def actionAbout(self):
        QMessageBox.about(MainWindow, "帮助", "Super-Resolution——超分辨率重建\n可选择两种深度神经网络\n注意图像的分辨率不能太大"
                                      "（右键文件-属性-详细信息）查看\n太大电脑承受不住，经测试在800以下比较好")


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)#必须在实例化对象前建立
    MainWindow = QtWidgets.QMainWindow()
    mywindow = MyWindow()
    mywindow.setupUi(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())
