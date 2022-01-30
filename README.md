# SR_Generator
**超分辨率重建：**
使用预训练好的SRGAN深度神经网络和SRResnet深度神经网络，能对低分辨率的图像重建为超分辨率图像，对纹理的构建尤为明显。

![8e1bb5b81a5065cc98f125879ab1f51](https://user-images.githubusercontent.com/78554147/151700461-07d3f495-df84-4fbb-b406-ed569ab42a20.jpg)
![4b70ae22329c248d2e5ab01e854c95f](https://user-images.githubusercontent.com/78554147/151700481-a8b19e4f-5f9d-469c-adac-eb1316a17012.png)

# 如何使用
* 先确保下载的压缩包内部的文件夹存在，否则输出会找不到路径
* 首先在两种网络中选取一种，其中SRGAN网络产生纹理更明显，SRResnet产生的图像较为平滑
* 接着选择图片或者批量操作：若选择图片，则需选择后缀为png或者jpg的图片（由于大多数是这两种所以没有考虑其他的后缀，如果需要可以联系我，更改很简单），对应的结果输出在result文件夹下；若批量操作，则先将需要操作的图片放入img_lowR文件夹下，接着进行批量操作，最后输出结果会在img_superR文件夹下。

# 注意事项
由于这两个神经网络比较大型，参数较多，考虑到机器的配置，请不要将太大的图片输入进去。
## 图片分辨率
要查看图片分辨率，只需右键图片，打开属性选项，点击详细信息即可。
图片的分辨率在400*400左右时，神经网络推理的时间比较快，比较推荐。
最大不要超过800，否则电脑会比较卡顿。因此当图片分辨率任何一个维度（宽或者高）超过800时，该软件会将其自动缩小至800。

# 开发者留言
由于时间比较紧张，整个开发过程只用了不到一天时间，因此网络的效果可能不会太好，着重推荐对低分辨率的风景使用，对人像添加的纹理不太自然，不过可以用来娱乐一下。另外，如果遇到bug请随时联系我。
