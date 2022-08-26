import tensorflow as tf
from picture_identity import backward
from picture_identity import forward
from picture_identity import PreProcess as PP
import os

pwd = os.getcwd()
ROOT_PATH = os.path.dirname(pwd)


def restore_model(testArr):
    # 获取当前默认的计算图。
    with tf.Graph().as_default():
        # 输入x 输出y 识别结果preValue
        x = tf.compat.v1.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y = forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        # 在加载模型的时候将影子变量直接映射到变量的本身，所以我们在获取变量的滑动平均值的时候只需要获取到变量的本身值而不需要去获取影子变量。
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.compat.v1.train.Saver(variables_to_restore)

        # 加载，在会话中运行
        with tf.compat.v1.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue = sess.run(preValue, feed_dict={x: testArr})
                # 返回识别结果
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def application(file_path):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = PP.image_process(file_path)
    lable = ''
    if len(data) == 0:
        print("识别失败，请传入更清晰的图片")
    else:
        print("正在识别......")
        for i in range(len(data)):
            preValue = restore_model(data[i:i + 1])[0]
            lable += str(preValue)
        fp = open(ROOT_PATH+'/result/result_show.txt', "w+")  # w+ 如果文件不存在就创建
        print("识别结果：" + lable, file=fp)
        fp.close()

