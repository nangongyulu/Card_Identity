# # 用来测试代码
#
# # fp = open("./result/result_show.txt", "w+")  # w+ 如果文件不存在就创建
# # print("我打印成功了！", file=fp)
# # fp.close()
# #
# # import sys
# # import os
# # from picture_identity.identity_card import identityCard
# #
# # pwd = os.getcwd()
# # ROOT_PATH = os.path.dirname(pwd)
# # sys.path.append(ROOT_PATH + '/picture_identity')
# #
# # if __name__=='__main__':
# #    identityCard()
#

# def upload_file(file_id):
#     if 'file' not in request.files:
#         return False, "No file part"
#     file = request.files['file']
#     if file.filename == '':
#         return False, "No selected file"
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         save_name = f"{file_id}-{filename}"
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], save_name))
#         return True, save_name
#
#
# @blueprint.route('/<string:file_name>', methods=['GET'])
# def download_file(file_name):
#     if '/' in file_name:
#         return 'error', 400
#     if file_name:
#         return send_file(f"{FILE_FOLDER}/{file_name}", attachment_filename='file.jpg')
#     return 'error', 400
