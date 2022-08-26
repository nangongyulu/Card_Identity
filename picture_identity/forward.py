# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

INPUT_NODE = 256  # 输入层
OUTPUT_NODE = 10  # 输出层
LAYER1_NODE = 100  # 隐藏层
LAYER2_NODE = 100  # 隐藏层2


# 设置权重
def get_weight(shape, regularizer):
    # 生成截断正态分布的初始化程序。
    w = tf.Variable(tf.compat.v1.random.truncated_normal(shape, stddev=0.1))
    # 正则化器
    if regularizer is not None:
        tf.add_to_collection('losses', tf.keras.regularizers.l2(regularizer)(w))
    return w

# 设置偏置
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    # 先获取权重和偏置 x输入值 y1输出值
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    # 卷积 激活函数tf.nn.relu 线性整流函数
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)      # 计算本层的输出 即下一层的输入

    # 第二层重复操作 这一层的输入是上一层的输出y1
    w2 = get_weight([LAYER1_NODE, LAYER2_NODE], regularizer)
    b2 = get_bias([LAYER2_NODE])
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)

    # x是前向传播输入 y是前向传播输出
    w3 = get_weight([LAYER2_NODE, OUTPUT_NODE], regularizer)
    b3 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y2, w3) + b3
    return y
