import cv2 as cv
import numpy as np
import os
# 导入 多次感知机的分类模型 MLPClassifier
from sklearn.neural_network import MLPClassifier
# 导入 完成模型持久化的 joblib
import joblib


# 初始配置
# 训练集、测试集位置
TRAIN_DIR = 'data/enu_train'
TEST_DIR = 'data/enu_test'
# 图片宽、高
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
# 给出类别数、类别-数值 字典
CLASSIFICATION_COUNT = 26
LABEL_DICT = {
	'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,	'I': 8,
	'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17,
	'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25
}
# 模型持久化的位置
MLP_ENU_MODEL_PATH = 'model/mlp_enu.m'


# 1. 加载数据集
# 参数：数据集所在的目录位置， dir_path
# 返回值：样本的特征矩阵、标签向量，features, labels
def load_data(dir_path):
	data = []
	labels = []
	# 获取数据集目录下的所有的子目录，并逐一遍历
	for item in os.listdir(dir_path):
		# 获取每一个具体样本类型的 os 的路径形式
		item_path = os.path.join(dir_path, item)
		# 判断只有目录，才进入进行下一级目录的遍历
		if os.path.isdir(item_path):
			# 到了每一个样本目录，遍历其下的每一个训练集样本文件-图片
			for subitem in os.listdir(item_path):
				subitem_path = os.path.join(item_path, subitem)
				gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)
				# imread()函数读取图片  cv.IMREAD_GRAYSCALE以灰度图的方式读取
				resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
				data.append(resized_image.ravel())
				# append()函数在列表末尾增加新的值 ravel()函数将多维数组拉成一维数组
				labels.append(LABEL_DICT[item])
	# 分别赋值 样本数据特征、样本数据标签
	features = np.array(data)
	labels = np.array(labels)
	# 返回特征矩阵、标签向量
	return features, labels


# 2. 预处理 : 标准化
# 参数：特征矩阵，data
# 返回值：执行标准化后的 data
def normalize_data(data):
	return (data - data.mean()) / data.max()


# 3. 训练 + 保存
def train():
	# 加载训练数据
	train_data, train_labels = load_data(TRAIN_DIR)
	# 数据的预处理
	normalized_data = normalize_data(train_data)

	# 模型创建
	model = MLPClassifier(hidden_layer_sizes=(48, 25), solver='lbfgs', alpha=1e-5, random_state=42)

	# 模型训练
	model.fit(normalized_data, train_labels)

	# 模型保存
	joblib.dump(model, MLP_ENU_MODEL_PATH)


# 4. 测试 + 评估
def test():
	test_data, test_labels = load_data(TEST_DIR)
	normalized_data = normalize_data(test_data)

	model = joblib.load(MLP_ENU_MODEL_PATH)

	predicts = model.predict(normalized_data)

	errors = np.count_nonzero(predicts-test_labels)
	# np.count_nonzero()是用于统计矩阵中非零元素的个数。
	print(errors)
	print((len(predicts)-errors)/len(predicts))

if __name__ == '__main__':
	train()
	test()
