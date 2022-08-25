import tensorflow as tf
import Code_body.backward as backward
import Code_body.forward as forward
import Code_body.PreProcess as PP


def restore_model(testArr):
    with tf.Graph().as_default():
        x = tf.compat.v1.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y = forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.compat.v1.train.Saver(variables_to_restore)

        with tf.compat.v1.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue = sess.run(preValue, feed_dict={x: testArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def application(file_path):
    data = PP.image_process(file_path)
    lable = ''
    if len(data) == 0:
        print("识别失败，请传入更清晰的图片")
    else:
        print("正在识别......")
        for i in range(len(data)):
            preValue = restore_model(data[i:i + 1])[0]
            lable += str(preValue)
        desktop_path = '../result'  # 新创建的txt文件的存放路径
        full_path = desktop_path + 'result_show' + '.txt'  # 也可以创建一个.doc的word文档
        fp = open(full_path, 'w+')  # r 的含义为可进行读
        print("识别结果：" + lable, file=fp)
        fp.close()
