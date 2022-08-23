import cv2 as cv
from tensorflow._api.v2.compat.v1 import disable_v2_behavior

disable_v2_behavior()

from opencv_ml.cnn_number_predict import load_number_model
from opencv_char_seperator.card_char_seperator import get_candidate_card_number_chars
from opencv_locate.card_information_locator import get_card_number_by_sobel
from opencv_ml.cnn_number_predict import predict_number

# 传入信用卡图片
card_image = cv.imread('card/1 (1).jpg')

# 获取信用卡号候选区域
candidate_card_number = get_card_number_by_sobel(card_image)

# 打印候选区域列表
for i in range(len(candidate_card_number)):
    print(candidate_card_number[i])

# 获取信用卡号候选区域分割后的字符
candidate_char_card_number_images = []
for i in range(len(candidate_card_number)):
    candidate_char_card_number_images.extend(get_candidate_card_number_chars(candidate_card_number[i]))

# 将分割后的字符传入模型进行训练，识别信用卡号
card_number_char = []
for i in range(len(candidate_char_card_number_images)):
    card_number_char.extend(predict_number(candidate_char_card_number_images[i], load_number_model()))

# 打印银行卡号
for i in range(len(card_number_char)):
    print(card_number_char[i])

if __name__ == '__main__':
    print('PyCharm')
