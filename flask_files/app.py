import os
import time

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# 创建项目
app = Flask(__name__)
app.config['DEBUG'] = True
app.config['JSON_AS_ASCII'] = False
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 全局变量 共享的文件夹路径 可以根据需求更改
UPLOAD_FOLDER = 'image'
# 上传的文件夹路径
UPLOAD_PATH = os.path.join(app.root_path, '../image')
'''
     测试过程调用
     file_path表示用于测试的文件路径
     若想测试其他图片，请将file_path修改成相应的路径
     '''
FILE_PATH = '../image/img_3.png'


# 获取文件信息的函数
def get_files_data():
    files = []
    for i in os.listdir(UPLOAD_PATH):
        if len(i.split(".")) == 1:  # 判断此文件是否为一个文件夹
            continue

        # 拼接路径
        file_path = UPLOAD_PATH + "/" + i
        name = i
        size = os.path.getsize(file_path)  # 获取文件大小
        ctime = time.localtime(os.path.getctime(file_path))  # 格式化创建当时的时间戳

        # 列表信息
        files.append({
            "name": name,
            "size": size,
            "ctime": "{}年{}月{}日".format(ctime.tm_year, ctime.tm_mon, ctime.tm_mday),  # 拼接年月日信息
        })
    return files


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/intro")
def intro():
    return render_template("intro.html")


@app.route("/index")
def index():
    """共享文件主页"""
    return render_template("index.html", files=get_files_data())


@app.route("/HomePage")
def HomePage():
    return render_template("HomePage.html")


@app.route("/upload_file", methods=['POST'])
def upload():
    if request.method == 'POST':
        # 获取文件 拼接存储路径并保存
        upload_file = request.files.get("upload_file")
        upload_file.save(os.path.join(UPLOAD_PATH, upload_file.filename))
        #  返回上传成功的模板
        return render_template("upload_ok.html")

    return render_template("upload.html")


# 上传的网页
#
# @app.route('/upload_file', methods=['POST'])
# def upload():
#     upload_file = request.files.get("upload_file")
#     if upload_file and allowed_file(upload_file.filename):
#         filename = secure_filename(upload_file.filename)
#         # 将文件保存到 static/uploads 目录，文件名同上传时使用的文件名
#         upload_file.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename))
#         return render_template("upload_ok.html")
#     else:
#         return 'failed'


def ocr_detection():
    import Code_body.model_called
    Code_body.model_called.application(FILE_PATH)


@app.route('/ret_show', methods=['GET'])
def ret_show():
    ocr_detection()
    f = open('../result/result_show.txt', 'r')
    ret_list = f.readlines()
    print(ret_list)
    f.close()

    return render_template('ret_show.html', result_list=ret_list)


if __name__ == '__main__':
    app.run()
