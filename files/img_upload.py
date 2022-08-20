from flask import Flask
from flask import request, render_template, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename

# 设置文件上传参数
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
# 创建flask类实例app
app = Flask(__name__)
# 设置上传服务器文件夹
app.config['UPLOAD_FOLDER'] = os.getcwd() + '/upload'
# 设置上传文件大小限制
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.config['DEBUG'] = True

# 创建验证文件类型的自定义函数
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
# 创建获取上传文件的服务器地址的函数
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    error = ''
    if request.method == 'POST':
        file = request.files['uploadfile']
        # 判断客户端上传文件是否存在扩展名符合限制要求
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            return '图片上传成功！<br><br><img src=' + file_url + '>'
        else:
            error = '错误：上传文件类型错误！（允许后缀名:jpg、jpeg、png）'
            return render_template("uploadimg.html", error=error)
    return render_template("uploadimg.html")

if __name__ == '__main__':
    app.run()
