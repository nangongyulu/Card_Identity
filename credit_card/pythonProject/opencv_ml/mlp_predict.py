import joblib
import ml_predict_utility

# 英文标签
ENGLISH_LABELS = [
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
	'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
	'W', 'X', 'Y', 'Z']

# 数字标签
NUMBER_LABELS = (
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# 模型路径
ENGLISH_MODEL_PATH = "model/mlp/mlp_enu.m"
NUMBER_MODEL_PATH = "model/mlp/mlp_num.m"

# 设置图片大小
ENGLISH_IMAGE_WIDTH = 20
ENGLISH_IMAGE_HEIGHT = 20
NUMBER_IMAGE_WIDTH = 20
NUMBER_IMAGE_HEIGHT = 20

# 读入数据  图片路径
digit_image_path = "images/digit.jpg"
english_image_path = "images/english.jpg"

# 数据预处理
# 装载图片、转换成适当的尺寸并作归一化处理
digit_image = ml_predict_utility.load_image(digit_image_path, ENGLISH_IMAGE_WIDTH, ENGLISH_IMAGE_HEIGHT)
english_image = ml_predict_utility.load_image(english_image_path, ENGLISH_IMAGE_WIDTH, ENGLISH_IMAGE_HEIGHT)


# 获取模型（装载模型）
enu_model = joblib.load(ENGLISH_MODEL_PATH)
num_model = joblib.load(NUMBER_MODEL_PATH)

# 执行预测并输出预测的文本结果
predicts = enu_model.predict(digit_image)
print(ENGLISH_LABELS[predicts[0]])
predicts = enu_model.predict(english_image)
print(ENGLISH_LABELS[predicts[0]])

