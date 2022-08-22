import os
import time

from flask import Flask, render_template, request

import config

# 创建项目
app = Flask(__name__)
app.config.from_object(config)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 全局变量 共享的文件夹路径 可以根据需求更改
UPLOAD_FOLDER = 'image'
# 上传的文件夹路径
UPLOAD_PATH = os.path.join(app.root_path, '../image')


def allowed_file(filename):
    return '.' in filename and filename.rspliit('.', 1)[1] in ALLOWED_EXTENSIONS


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
def HomePage():
    return render_template("login.html")


@app.route("/index")
def index():
    """共享文件主页"""
    return render_template("index.html", files=get_files_data())


@app.route("/HomePage")
def login():
    return render_template("HomePage.html")


@app.route("/upload_file", methods=['GET', 'POST'])
def upload():
    """上传文件的URL 支持GET/POST请求"""
    if request.method == "POST":
        # 获取文件 拼接存储路径并保存
        upload_file = request.files.get("upload_file")
        upload_file.save(os.path.join(UPLOAD_PATH, upload_file.filename))
        #  返回上传成功的模板
        return render_template("upload_ok.html", filename=upload_file.filename)

    # 上传的网页

    return render_template("upload.html")


@app.route('/ret_show', methods=['GET'])
def ret_show():
    f = open('../result/result_show.txt', 'r')
    ret_list = f.readlines()
    print(ret_list)
    f.close()

    return render_template('ret_show.html', result_list=ret_list)


if __name__ == '__main__':
    app.run()
