import Code_body.backward as bw
import Code_body.model_called as model_called


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

    '''
    测试过程调用
    file_path表示用于测试的文件路径
    若想测试其他图片，请将file_path修改成相应的路径
    '''

    file_path = '../image/img_1.png'
    model_called.application(file_path)


identityCard()
