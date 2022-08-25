import os
import time
from flask import Flask, render_template, request
from os import path
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict
from form import UploadForm
from Code_body.identity_card import identityCard

# 创建项目
app = Flask(__name__)
app.config['DEBUG'] = True
app.config['JSON_AS_ASCII'] = False
ALLOWED_EXTENSIONS = {'png', 'jpg'}
# 上传的文件夹路径
pwd = os.getcwd()
ROOT_PATH = os.path.dirname(pwd)
UPLOAD_PATH = ROOT_PATH + '/image/'
DIR_PY = ROOT_PATH + '/Code_body/' + 'identity_card'


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


@app.route("/upload_file", methods=['GET', 'POST'])
def upload():
    """上传文件的URL 支持GET/POST请求"""
    if request.method == 'GET':
        return render_template('upload.html')
    else:
        filelist = os.listdir(ROOT_PATH+'/image/')
        for f in filelist:
            os.remove(ROOT_PATH+'/image/' + f)
        if not path.exists(UPLOAD_PATH):
            os.makedirs(UPLOAD_PATH)  # 路径不存在时创建路径
        form = UploadForm(CombinedMultiDict([request.form, request.files]))
        if form.validate():
            # 获取文件 拼接存储路径并保存
            upload_file = request.files['file']
            filename = secure_filename(upload_file.filename)
            ext = filename.rsplit('.', 1)[1]  # 获取文件后缀
            # 重新用于命名，不会重复
            new_filename = 'img_' + '1' + '.' + ext
            upload_file.save(path.join(UPLOAD_PATH, new_filename))
            #  返回上传成功的模板
            return render_template("upload_ok.html")
        else:
            return "仅支持jpg和png格式的图片!"


def de_file():
    filelist = os.listdir(ROOT_PATH+'/result/')
    for f in filelist:
        os.remove(ROOT_PATH+'/result/' + f)


@app.route('/ret_show', methods=['GET'])
def ret_show():
    de_file()
    identityCard()
    file = open(ROOT_PATH+'/result/result_show.txt', 'r')
    ret_list = file.readlines()
    print('111111')
    print(ret_list)
    file.close()

    return render_template('ret_show.html', result_list=ret_list)


if __name__ == '__main__':
    app.run()
