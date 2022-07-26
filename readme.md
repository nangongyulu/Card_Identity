# 项目背景
信用卡数字识别-基于Tensorflow&openCV

输入用例：待识别银行卡图片

![image](./for_test_image/img_3.png)

输出用例：识别结果

![img.png](picture_identity/img.png)

# 项目结构
## 图片识别 picture_identity
```
images: 训练集（分隔好的图片）
model: 训练模型
identity_card.py: 模型训练入口文件
PreProcess.py & ImgHandle.py: 图像处理代码
forward.py: 深度学习模型前向传播代码
backward.py: 深度学习模型反向传播代码
model_called.py: 模型调用代码
```
## 前端页面 flask_files
```
static:存放html页面.css样式表、页面所用图片
template:存放.html文件
app.py:主程序入口代码
form.py:控制上传照片格式（jpg、png）
```

## 其他文件夹
```
for_test_image:用于测试的图片
image:保存上传图片
result:识别结果输出
```

# 开发环境
语言：`Python3.8`

深度学习框架：`TensorFlow`

图像处理：`OpenCV`

前端框架：`Flask` `html/css`

# 使用方法
### 1.训练模型
删除`model`文件夹内容

进入`identity_card.py`

将布尔变量`train`的值改为 True

运行`identity_card.py`

将布尔变量`train`的值改为 False
### 2.正式使用
运行flask_files下app.py文件，进入前端界面

![img.png](picture_identity/login.png)

先注册后登录，进入主页面（HomePage）

![img.png](picture_identity/homepage.png)

主页面中可选择功能简介、图片上传、识别结果。

点击上传图片，完成上传后回到主页面，点击识别结果即可查看识别出的银行卡号
