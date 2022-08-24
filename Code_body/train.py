import backward as bw
if __name__ == '__main__':
    '''
    训练过程调用
    train表示是否需要继续训练
    若需要继续训练，将该参数改为True，
    若需要重新训练，请将model文件夹清空并将该参数改为True
    '''
    train = False
    if train:
        bw.main()