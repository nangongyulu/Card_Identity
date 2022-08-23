import cv2 as cv
import numpy as np
import util_python


# 借助sobel算子完成信用卡号区域的提取
# 参数：信用卡（含有信用卡号的车辆）图片
# 返回值：所有可能含有信用卡号的候选区域 = list
def get_card_number_by_sobel(card_image):
    # 1. 对含有信用卡信息的图片进行预处理(sobel)
    preprocess_image = util_python.preprocess_card_image_by_sobel(card_image)
    # 2. 提取所有的等值线|信用卡信息(可能)的区域
    contours, _ = cv.findContours(preprocess_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # cv.RETR_EXTERNAL只检测最外围轮廓，内部忽视    cv.CHAIN_APPROX_NONE保存边界连续轮廓点到contours中
    # 3. 判断并获取所有的信用卡信息区域的候选区域列表
    candidate_card_number = []
    # 遍历所有的可能的信用卡信息轮廓|等值线框
    for i in np.arange(len(contours)):
        # 逐一获取某一个可能的信用卡号轮廓区域
        contour = contours[i]
        # 根据面积、长宽比判断是否是候选的信用卡号区域
        if util_python.verify_card_number_sizes(contour):
            # 完成旋转
            output_image = util_python.rotate_card_image(contour, card_image)
            # 统一尺寸
            uniformed_image = util_python.unify_card_image(output_image)
            # 追加到信用卡号区域的候选区域列表中
            candidate_card_number.append(uniformed_image)
    # 返回含有所有的可能信用卡号区域的候选区域列表
    return candidate_card_number


# 借助sobel算子完成信用卡有效日期区域的提取
# 参数：信用卡（含有信用卡有效日期的车辆）图片
# 返回值：所有可能含有信用卡有效日期的候选区域 = list
def get_card_date_by_sobel(card_image):
    # 1. 对含有信用卡有效日期的图片进行预处理(sobel)
    preprocess_image = util_python.preprocess_card_image_by_sobel(card_image)
    # 2. 提取所有的等值线|信用卡有效日期(可能)的区域
    contours, _ = cv.findContours(preprocess_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # cv.RETR_EXTERNAL只检测最外围轮廓，内部忽视    cv.CHAIN_APPROX_NONE保存边界连续轮廓点到contours中
    # 3. 判断并获取所有的信用卡有效日期区域的候选区域列表
    candidate_card_date = []
    # 遍历所有的可能的信用卡有效日期轮廓|等值线框
    for i in np.arange(len(contours)):
        # 逐一获取某一个可能的信用卡有效日期轮廓区域
        contour = contours[i]
        # 根据面积、长宽比判断是否是候选的信用卡有效日期区域
        if util_python.verify_card_number_sizes(contour):
            # 完成旋转
            output_image = util_python.rotate_card_image(contour, card_image)
            # 统一尺寸
            uniformed_image = util_python.unify_card_image(output_image)
            # 追加到信用卡有效日期区域的候选区域列表中
            candidate_card_date.append(uniformed_image)
    # 返回含有所有的可能信用卡有效日期区域的候选区域列表
    return candidate_card_date
