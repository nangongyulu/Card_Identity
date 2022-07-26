import cv2
import numpy as np



def image_process(file_path):
    img = cv2.imread(file_path, 0)
    blur = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊     减少噪声及不必要的细节
    ret, binary = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)  # 二值化
    # ret是否读取到图像 ture/false  binary 二值化后的图像

    kernel = np.ones((1, 50), np.uint8)
    # np.ones() 函数返回给定形状和数据类型的新数组，元素设为1
    # (1,50)形状  uint8专门用于存储图像

    # 定位银行卡
    erosion = cv2.erode(binary, kernel)  # 膨胀
    dilation = cv2.dilate(erosion, kernel)  # 腐蚀

    # 寻找轮廓    cv2.RETR_TREE建立一个等级树结构轮廓   cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
    # 返回值：contours轮廓  hierarchy轮廓之间嵌套和临近关系
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sp = dilation.shape
    x, y, w, h = 0, 0, 0, 0
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        # boundingRect寻找最小正矩形  返回值是轮廓i的左上角坐标和宽高
        # sp[0]高 sp[1]宽   这一步用于筛选代表银行卡号的子块
        if h > sp[0] * 0.05 and w > sp[1] * 0.5 and sp[0] * 0.2 < y < sp[0] * 0.8 and w / h > 5:
            # 卡号字符定位
            img = binary[y:y + h, x:x + w]
            break

    return num_split(img)

# 字符分割
def num_split(img):
    height, width = img.shape
    v = [0] * width
    z = [0] * height
    a = 0

    # 垂直投影：统计并存储每一列的黑点数
    for x in range(0, width):
        for y in range(0, height):
            if img[y, x] == 255:
                continue
            else:
                a = a + 1
        v[x] = a
        a = 0

    # 创建空白图片，绘制垂直投影图
    len(v)
    emptyImage = np.full((height, width), 255, dtype=np.uint8)
    for x in range(0, width):
        for y in range(0, v[x]):
            emptyImage[y, x] = 0

    # 分割字符
    Position = []
    Wstart = 0
    Wend = 0
    W_Start = 0
    W_End = 0
    v[0], v[len(v) - 1] = 0, 0
    for j in range(len(v)):
        if v[j] > 0 and Wstart == 0:
            W_Start = j
            Wstart = 1
            Wend = 0
        if v[j] <= 0 and Wstart == 1:
            W_End = j
            Wstart = 0
            Wend = 1
        if Wend == 1:
            Position.append([W_Start, 0, W_End, height])
            Wend = 0

    data = []
    for m in range(len(Position)):
        temp_img = img[Position[m][1]:Position[m][3], Position[m][0]:Position[m][2]]

        h1, w1 = temp_img.shape
        if w1 > h1:
            return []
        temp_img = cv2.resize(temp_img, (16, 16))

        h0, w0 = temp_img.shape
        temp_data = []
        for hx in range(h0):
            for wx in range(w0):
                temp_data.append(float(temp_img[hx, wx]))
        data.append(temp_data)

    return data
