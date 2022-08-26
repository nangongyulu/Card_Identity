import tensorflow._api.v2.compat.v1 as tf
from picture_identity import forward
import os
from picture_identity import ImgHandle as IMG
import random

tf.disable_v2_behavior()

BATCH_SIZE = 20  # 每轮训练位入20个数据
REGULARIZER = 0.001  # 正则化系数0.001
STEPS = 10000  # 训练10000轮
MOVING_AVERAGE_DECAY = 0.01  # 滑动平均系数0.01
MODEL_SAVE_PATH = "./model/"  # 模型存储路径
MODEL_NAME = "train_model"  # 模型名称
FILE_NAME = "Classification.xlsx"  # 文件名称


def backward(data, label):
    tf.compat.v1.disable_eager_execution()
    # 禁用默认的即时执行模式。

    # x 样本数据集 y_标签——正确结果
    # 前两行定义样本和标签的形状  样本长256 标签长10
    # 为将输入的张量插入占位符placeholder
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, forward.INPUT_NODE))
    y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, forward.OUTPUT_NODE))
    # 调用前向传播forwar方法
    y = forward.forward(x, REGULARIZER)
    # 创建变量  全局步数
    global_step = tf.Variable(0, trainable=False)

    # 计算logits和labels之间的稀疏softmax交叉熵。
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # tf.reduce_mean计算某一维的平均值，用于降维
    cem = tf.reduce_mean(ce)
    # 正则化  将所有正则化的变量取出来放入一个列表中，最后将列表值加起来，加到loss上，完成整个正则化过程
    # 定义损失函数
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 训练过程
    # 使损失函数变到最小
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss, global_step=global_step)  # 计算梯度 应用到到变量更新中
    # 梯度降维 原始梯度下降方法 参数学习率0.001

    # tf.train.ExponentialMovingAverage这个函数用于更新参数，就是采用滑动平均的方法更新参数。
    # 这个函数初始化需要提供一个衰减速率（decay），用于控制模型的更新速度。
    # MOVING_AVERAGE_DECAY滑动平均系数 0.01
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 增加模型的泛化性
    # ema.apply([]) 定义了对括号里列出的参数求滑动平均。在实际应用中，会写ema.apply(tf.trainable_variables())。
    # 用tf.trainable_variable()函数可以自动把所有待训练的参数汇总成列表。
    ema_op = ema.apply(tf.trainable_variables())
    # 控制计算流图 指定计算顺序
    # 返回一个控制依赖的上下文管理器，使用with关键字可以让在这个上下文环境中的操作都在[train_step, ema_op]中执行。
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    # 保存训练模型
    saver = tf.train.Saver()

    # 建立会话
    with tf.Session() as sess:
        # 初始化模型的参数
        init_op = tf.global_variables_initializer()
        sess.run(init_op)  # 运行模型

        # 从“检查点”文件返回CheckpointState原型
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):  # 循环10000轮
            start = (i * BATCH_SIZE) % len(data)
            end = start + BATCH_SIZE
            # x和y_作为输入喂入神经网络
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: data[start:end], y_: label[start:end]})
            # 每100轮打印当时的loss值
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


# 训练过程入口
def main():
    data, label = IMG.img_handle()
    for i in range(len(data)):
        x, y = random.randint(0, len(data) - 1), random.randint(0, len(data) - 1)
        temp_data = data[x]
        data[x] = data[y]
        data[y] = temp_data
        temp_label = label[x]
        label[x] = label[y]
        label[y] = temp_label
    print(len(data), len(label))
    backward(data, label)
