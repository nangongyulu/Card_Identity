# 导包
import cv2 as cv
import numpy as np


# 信用卡信息区域拆分成为候选字符列表   卡号 card_number  有效日期  date   姓名 name
# 参数： 信用卡信息候选区域（某一个），candidate_card_image
# 返回值： 拆分后的信用卡信息字符 按顺序生成信用卡信息字符候选列表
# 信用卡号拆分字符
def get_candidate_card_number_chars(candidate_card_number_image):
    # 1.图片预处理： 灰度图 二值化
    gray_image = cv.cvtColor(candidate_card_number_image, cv.COLOR_BGR2GRAY)
    is_success, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)

    # 2.向内缩进 去除外边框
    # 经验值
    offset_x = 1
    offset_y = 1
    # 切片内嵌区域
    offset_region = binary_image[offset_y:-offset_y, offset_x:-offset_x]
    # 生成工作区域
    working_region = offset_region

    # 3.对信用卡号区域进行等值线找区域
    char_contours, _ = cv.findContours(working_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    # 4. 过滤不合适的轮廓（等值线框）
    # 经验值
    CHAR_MIN_WIDTH = working_region.shape[1] // 40
    CHAR_MIN_HEIGHT = working_region.shape[0] * 7 // 10

    # 5. 逐个遍历所有候选的字符区域轮廓==等值线框，按照大小进行过滤
    valid_char_card_number_regions = []
    for i in np.arange(len(char_contours)):
        x, y, w, h = cv.boundingRect(char_contours[i])
        if w >= CHAR_MIN_WIDTH and h >= CHAR_MIN_HEIGHT:
            # 将字符区域的中心点x的坐标 和 字符区域 作为一个元组，放入 valid_char_regions 列表
            valid_char_card_number_regions.append((x, offset_region[y:y + h, x:x + w]))

    # 6. 按找区域的x坐标进行排序，并返回字符列表
    sorted_regions = sorted(valid_char_card_number_regions, key=lambda region: region[0])
    # valid_char_card_number_regions
    # sorted_regions
    candidate_char_card_number_images = []
    for i in np.arange(len(sorted_regions)):
        candidate_char_card_number_images.append(sorted_regions[i][1])

    return candidate_char_card_number_images


# 信用卡有效日期拆分字符
def get_candidate_card_date_chars(candidate_card_date_image):
    # 1.图片预处理： 灰度图 二值化
    gray_image = cv.cvtColor(candidate_card_date_image, cv.COLOR_BGR2GRAY)
    is_success, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)

    # 2.向内缩进 去除外边框
    # 经验值
    offset_x = 1
    offset_y = 1
    # 切片内嵌区域
    offset_region = binary_image[offset_y:-offset_y, offset_x:-offset_x]
    # 生成工作区域
    working_region = offset_region

    # 3.对信用卡有效日期区域进行等值线找区域
    char_contours, _ = cv.findContours(working_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    # 4. 过滤不合适的轮廓（等值线框）
    # 经验值
    CHAR_MIN_WIDTH = working_region.shape[1] // 40
    CHAR_MIN_HEIGHT = working_region.shape[0] * 7 // 10

    # 5. 逐个遍历所有候选的字符区域轮廓==等值线框，按照大小进行过滤
    valid_char_card_date_regions = []
    for i in np.arange(len(char_contours)):
        x, y, w, h = cv.boundingRect(char_contours[i])
        if w >= CHAR_MIN_WIDTH and h >= CHAR_MIN_HEIGHT:
            # 将字符区域的中心点x的坐标 和 字符区域 作为一个元组，放入 valid_char_regions 列表
            valid_char_card_date_regions.append((x, offset_region[y:y + h, x:x + w]))

    # 6. 按找区域的x坐标进行排序，并返回字符列表
    sorted_regions = sorted(valid_char_card_date_regions, key=lambda region: region[0])
    # valid_char_card_number_regions
    # sorted_regions
    candidate_char_card_date_images = []
    for i in np.arange(len(sorted_regions)):
        candidate_char_card_date_images.append(sorted_regions[i][1])

    return candidate_char_card_date_images
