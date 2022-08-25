# 项目背景
银行卡号识别系统-基于Tensorflow&openCV

输入用例：待识别银行卡图片

![image](./for_test_image/img_3.png)

输出用例：识别结果

![img.png](./Code_body/img.png)

# 项目结构
## 后端核心代码 Code_body
```
images: 训练集
model: 训练模型
identity_card.py: 入口文件
PreProcess.py & ImgHandle.py: 图像处理代码
forward.py: 深度学习模型前向传播代码
backward.py: 深度学习模型反向传播代码
model_called.py: 模型调用代码
```
## 前端页面 flask_files
```
static:存放html页面.css样式表、页面所用图片
template:存放.html文件
app:Flask框架代码
form:控制上传照片格式（jpg、png）
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

将布尔变量`train`的值改为`true`

运行`identity_card.py`
### 2.模型调用
进入`identity_card.py`

修改变量`file_path`的值为想要识别的图片路径

运行`identity_card.py`

### 3.正式使用
运行flask_files下app.py文件，进入前端界面

![img.png](./Code_body/login.png)

先注册后登录，进入主页面（HomePage）

![img.png](./Code_body/homepage.png)

主页面中可选择功能简介、图片上传、识别结果。

点击上传图片，完成上传后回到主页面，点击识别结果即可查看识别出的银行卡号
