# Background
银行卡号识别系统-基于Tensorflow&openCV

输入用例：待识别银行卡图片

![image](test_images/img_3.png)

输出用例：识别结果

![img.png](img.png)

项目结构：
```
images: 训练集
test_images: 测试集
model: 训练模型
train.py: 入口文件
PreProcess.py & ImgHandle.py: 图像处理代码
forward.py: 深度学习模型前向传播代码
backward.py: 深度学习模型反向传播代码
model_called.py: 模型调用代码
```
# Enviroment
语言：`Python3.7`

深度学习框架：`TensorFlow`

图像处理：`OpenCV`

### 训练模型
删除`model`文件夹内容

进入`train.py`

将布尔变量`train`的值改为`true`

运行`train.py`
### 模型调用
进入`train.py`

修改变量`file_path`的值为想要识别的图片路径

运行`train.py`
