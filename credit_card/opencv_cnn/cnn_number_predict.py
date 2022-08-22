'''
使用简单卷积神经网络分类，网络结构为：
Conv -> ReLU -> Max Pooling -> Conv -> ReLU -> Max Pooling ->
FC1(1024) -> ReLU -> Dropout -> Affine -> Softmax
'''

import os
import numpy as np
import cv2 as cv
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


MODEL_PATH = "model/cnn_num/num.ckpt"
TRAIN_DIR = "data/num_train"
TEST_DIR = "data/num_test"
# 英文图片重置的宽、高
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
# 给出类别数、类别-数值 字典
CLASSIFICATION_COUNT = 10
LABEL_DICT = {
	'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,	'8': 8,	'9': 9
}
ENGLISH_LABELS = []
NUMBER_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# 设置GPU内存为陆续分配，防止一次性的全部分配GPU内存，导致系统过载
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


# 本质：完成数据的正则化
def normalize_data(data):
    return (data - data.mean()) / data.max()

# 构建 独热编码
def onehot_labels(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots

# 设置权重，并根据shape，使用截断正态分布获取随机数进行初始化
def weight_variable(shape):
    # 会从 [mean（默认为0） ± 2stddev] 的范围内产生随机数
    # 如果产生的随机数不在这个范围内，就会再次随机产生，直到随机数落在这个范围内。
    # 经验：使用这种方式产生的 weight 不容易出现梯度消失或者爆炸的问题。
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 设置偏置，并初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 设置卷积层
def conv2d(x, W):
    # strides 设置步长，设置水平和垂直步长均为 1
    # tf规定是一个一维具有四个元素的张量，其规定前后必须为1，可以改的是中间两个数
    # 中间两个数分别代表了水平滑动和垂直滑动步长值。
    # padding='SAME',使卷积输出的尺寸=ceil(输入尺寸/stride)，必要时自动padding
    # 此时因为步长为1，所以卷积后保持图像原尺寸不变，当前数据集图像尺寸为：20*20
    # padding='VALID',不会自动padding，对于输入图像右边和下边多余的元素，直接丢弃
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



# 设置池化层
def max_pool_2x2(x):
    # 设置池化核为2x2：ksize=[1, 2, 2, 1]
    # 设置池化步长，水平和垂直均为2：strides=[1, 2, 2, 1]
    # 设置池化必要时自动padding：padding='SAME'
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def load_number_model():
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # cnn第一层
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # cnn第二层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 10x10
    h_pool2 = max_pool_2x2(h_conv2)  # 10x10 => 5x5

    # 全连接神经网络的第一个隐藏层
    W_fc1 = weight_variable([5 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 在全链接层之后，往往会跟着 Dropout 操作。
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全连接神经网络输出层
    W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT])
    b_fc2 = bias_variable([CLASSIFICATION_COUNT])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # 至此， CNN 的网络构建完成，但只是的前向部分。

    learning_rate = 1e-4  # 学习率
    max_epochs = 40  # 代数
    batch_size = 50  # 批大小
    check_step = 10  # 检查点步长

    # 完成交叉损失熵的计算
    logits = y_conv
    y = tf.nn.softmax(logits=logits)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

    # 完成反向传播
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # 以上完成反向BP算法，构建出一个 交叉熵 损失函数，对损失函数做梯度下降

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 至此，完成整个评估模型的构建

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_PATH)

    return (sess, x, keep_prob, y_conv)

def predict_number(char_image,model):
    # 将原始图变成规定大小
    origin_hight, origin_width = char_image.shape
    resize_hight = IMAGE_WIDTH - 2 if origin_hight > IMAGE_HEIGHT else origin_hight
    resize_width = IMAGE_WIDTH - 2 if origin_width > IMAGE_WIDTH else origin_width
    resize_image = cv.resize(char_image, (resize_width, resize_hight))

    working_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    x_idx = (IMAGE_WIDTH - resize_width) // 2
    y_idx = (IMAGE_HEIGHT - resize_hight) // 2
    working_image[y_idx:y_idx+resize_hight, x_idx:x_idx+resize_width] = resize_image

    working_image = normalize_data(working_image)
    data = []
    data.append(working_image.ravel())

    sess, x, keep_prob, y_conv = model
    results = sess.run(y_conv, feed_dict={x:data, keep_prob:1.0})
    predict_number = np.argmax(results[0])
    return NUMBER_LABELS[predict_number]

if __name__ == '__main__':
    char_image_file = 'data/num_test/1/210.jpg'
    char_image = cv.imread(char_image_file, cv.COLOR_BGR2GRAY)

    char = predict_number(char_image, load_number_model())
    print(char)




