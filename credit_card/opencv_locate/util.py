# 工具类
# 导包
import cv2 as cv
import numpy as np


# 0. 预处理信用卡图片，借助 sobel 算子完成
# 参数：信用卡图片（含有卡号card_name、有效日期date、姓名name），card_image
# 返回值：预处理后的信用卡图片，  preprocess_image
def preprocess_card_image_by_sobel(card_image):
    # 图片预处理
    # 高斯模糊
    blured_image = cv.GaussianBlur(card_image, (5, 5), 0)
    # 转成灰度图
    gray_image = cv.cvtColor(blured_image, cv.COLOR_BGR2GRAY)
    # 使用Sobel算子，求水平方向一阶导数
    # 使用 cv.CV_16S
    grad_x = cv.Sobel(gray_image, cv.CV_16S, 1, 0, ksize=3)
    # 转成 CV-_8U - 借助 cv.convertScaleAbs()方法
    abs_grad_x = cv.convertScaleAbs(grad_x)
    # 叠加水平和垂直（此处不用）方向，获取 sobel 的输出
    gray_image = cv.addWeighted(abs_grad_x, 1, 0, 0, 0)
    # cv.destroyAllWindows()
    # 二值化操作
    is_success, threshold_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
    # 执行闭操作=>信息连成矩形区域
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 3))
    morphology_image = cv.morphologyEx(threshold_image, cv.MORPH_CLOSE, kernel)
    preprocess_image = morphology_image
    return preprocess_image

# 1. 判断是否是有效信息区域（依据：面接、长宽比）
# 参数：某个轮廓-候选信息区域 ： card_number_contour、date_contour、name_contour
# 返回值： bool（True/False)
# 判断信用卡号区域
def verify_card_number_sizes(contour):
    # 声明常量：长宽比(最小、最大)，面积(最小、最大) == 可以微调
    MIN_ASPECT_RATIO_1 = 2.0
    MAX_ASPECT_RATIO_1 = 3.0
    MIN_AREA_1 = 50.0 * 17
    MAX_AREA_1 = 250.0 * 100

    # 获取矩形特征描述的等值线区域，返回：中心点坐标、长和宽、旋转角度--float
    (center_x, center_y), (w, h), angle = cv.minAreaRect(contour)
    # 获取宽、高=>int
    w = int(w)
    h = int(h)

    # 进行面积判断
    area = w * h
    if area > MAX_AREA_1 or area < MIN_AREA_1:
        return False

    # 进行长宽比判断
    # 获取长宽比
    aspect_ratio = w / h
    # 判定信用卡号是否竖排
    if aspect_ratio < 1:
        aspect_ratio = 1.0 / aspect_ratio
    # 判定
    if aspect_ratio > MAX_ASPECT_RATIO_1 or aspect_ratio < MIN_ASPECT_RATIO_1:
        return False

    return True

# 判断信用卡有效期区域
def verify_card_date_sizes(contour):
    # 声明常量：长宽比(最小、最大)，面积(最小、最大) == 可以微调
    MIN_ASPECT_RATIO_2 = 3.0
    MAX_ASPECT_RATIO_2 = 4.0
    MIN_AREA_2 = 40.0 * 10
    MAX_AREA_2 = 210.0 * 70

    # 获取矩形特征描述的等值线区域，返回：中心点坐标、长和宽、旋转角度--float
    (center_x, center_y), (w, h), angle = cv.minAreaRect(contour)
    # 获取宽、高=>int
    w = int(w)
    h = int(h)

    # 进行面积判断
    area = w * h
    if area > MAX_AREA_2 or area < MIN_AREA_2:
        return False

    # 进行长宽比判断
    # 获取长宽比
    aspect_ratio = w / h
    # 判定有效日期是否竖排
    if aspect_ratio < 1:
        aspect_ratio = 1.0 / aspect_ratio
    # 判定
    if aspect_ratio > MAX_ASPECT_RATIO_2 or aspect_ratio < MIN_ASPECT_RATIO_2:
        return False

    return True





# 2.信用卡信息区域旋转矫正（依据：根据长宽判断旋转角度是否需要修正、借助转换|旋转矩阵和原始的图片|扩充图片完成旋转|仿射）
# 参数：信用卡信息区域：contour， 原始图片|信用卡信息区域旋转态图片：plate_image
# 返回值： output_image  == 完成旋转矫正后的信用卡信息区域图片
def rotate_card_image(contour, card_image):
    # 获取信用卡信息区域的正交外接矩形，同时也会返回 长、宽
    # boundingRect 用于获取与 等值线框（轮廓框）contour 的四个角点正交的矩形
    # 返回 左上的坐标（x, y），宽（w）、高（h）
    x, y, w, h = cv.boundingRect(contour)
    # 生成该外接正交矩形的图片矩阵:对原始信用卡图片的切片提取
    bounding_image = card_image[y: y+h, x: x+w]

    # *1. 判断并修订旋转角度
    # 获取矩形特征描述的等值线区域，返回：中心点坐标、长和宽、旋转角度
    rect = cv.minAreaRect(contour)
    # 获取整数形式的 长、宽
    rect_width, rect_height = np.int0(rect[1])
    # 获取旋转角度|畸变角度
    angle = np.abs(rect[2])
    # 自行调整：1. 大小关系、2.角度修订
    if rect_width > rect_height:
        temp = rect_width
        rect_width = rect_height
        rect_height = temp
        angle = 90 + angle    # 需要理解&修改
    # 对于较小的旋转角度，不予理会，具体值：需要微调的经验值
    if angle <= 5.0 or angle >= 175:
        # 直接返回包含信用卡信息区域的信用卡图片，不旋转
        return bounding_image

    # 完成旋转
    # 创建一个放大图片区域，保存旋转之后的结果
    enlarged_width = w * 3 // 2
    enlarged_height = h * 3 // 2
    enlarged_image = np.zeros((enlarged_height, enlarged_width, card_image.shape[2]), dtype=card_image.dtype)

    x_in_enlarged = (enlarged_width - w) // 2
    y_in_enlarged = (enlarged_height - h) // 2
    # 获取放大图片的居中图片（全0）
    roi_image = enlarged_image[y_in_enlarged:y_in_enlarged + h, x_in_enlarged:x_in_enlarged + w, :]
    # 将旋转前的图片(bounding_image)放置到放大图片的居中位置 == copy
    cv.addWeighted(roi_image, 0, bounding_image, 1, 0, roi_image)
    # 计算旋转中心：就是放大图片的中心
    new_center = (enlarged_width // 2, enlarged_height // 2)
    # *2. 开始旋转
    # 计算获取旋转的转换矩阵，旋转角度需要自行微调
    transform_matrix = cv.getRotationMatrix2D(new_center, angle+270, 1.0)
    # 进行|完成旋转：原始图片和旋转转换矩阵的仿射计算
    transform_image = cv.warpAffine(enlarged_image, transform_matrix, (enlarged_width, enlarged_height))

    # 获取输出图：截取与最初的等值线框|信用卡信息轮廓的长宽相同的部分
    output_image = cv.getRectSubPix(transform_image, (rect_height, rect_width), new_center)

    return output_image

# 3. 统一尺寸：将信用卡信息图片调整到统一一致的大小
# 参数：需要统一尺寸的车牌， card_information_image
# 返回值：resize 之后的统一尺寸的图片，uniformed_image
def unify_card_image(card_information_image):
    # 声明统一的尺寸
    PLATE_STD_HEIGHT = 36
    PLATE_STD_WIDTH = 136
    # 完成 resize
    uniformed_image = cv.resize(card_information_image, (PLATE_STD_WIDTH, PLATE_STD_HEIGHT))
    return uniformed_image

