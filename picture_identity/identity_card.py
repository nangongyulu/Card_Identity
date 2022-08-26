from picture_identity import backward as bw
from picture_identity import model_called as model_called
import os


def identityCard():
    """
    训练过程调用
    train表示是否需要继续训练
    若需要继续训练，将该参数改为True，
    若需要重新训练，请将model文件夹清空并将该参数改为True
    """
    train = False
    if train:
        bw.main()

    # 获取文件路径
    pwd = os.getcwd()
    filelist = os.listdir(os.path.dirname(pwd) + '/image/')
    for f in filelist:
        model_called.application(os.path.dirname(pwd) + '/image/' + f)


if __name__ == '__main__':
    identityCard()
