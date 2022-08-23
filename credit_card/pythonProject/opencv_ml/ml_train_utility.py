# 加载数据集并进行预处理

# 导入所需的第三方扩展包
import os
import numpy as np
import cv2 as cv


# 1. 加载数据集
# 参数：数据集所在的目录位置， dir_path
# 	额外参数： IMAGE_WIDTH, IMAGE_HEIGHT, LABEL_DICT
# 返回值：样本的特征矩阵、标签向量，features, labels
def load_data(dir_path, IMAGE_WIDTH, IMAGE_HEIGHT, LABEL_DICT):
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
				resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
				data.append(resized_image.ravel())
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

