# 导入工具包

# imutils是在OPenCV基础上的一个封装,达到更为简结的调用OPenCV接口的目的
from imutils import contours
# 主要用于对多维数组执行计算，极大地简化了向量和矩阵的操作处理
import numpy as np
# argparse 是 python 用于解析命令行参数和选项的标准模块
import argparse
# OpenCV2（OpenCV是一个基于BSD许可(开源)发行的跨平台计算机视觉库）
import cv2
# 导入myutils.py文件里面的方法
import myutils

# 设置参数（输入图像，模板的位置）

# argparse 模块是 Python 内置的一个用于命令项选项与参数解析的模块，
# argparse 模块可以让人轻松编写用户友好的命令行接口。
# 通过在程序中定义好我们需要的参数，然后 argparse 将会从 sys.argv 解析出这些参数。
# argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",default='./images/credit_card_03.png', required=False,
	help="path to input image")
ap.add_argument("-t", "--template",default='./images/ocr_a_reference.png', required=False,
	help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}
# 绘图展示
# def cv_show(name,img):
# 	cv2.imshow(name, img)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()
# 读取一个模板图像（因为BGR，所以每个像素都是[255 255 255]）
img = cv2.imread(args["template"])
# cv_show('img',img)
# 转换为灰度图（BGR 到 GRAY）
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv_show('ref',ref)
# 二值图像（选项为cv2.THRESH_BINARY_INV）
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
# cv_show('ref',ref)

# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓

# ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# -1表示绘制所有轮廓，后面是画笔颜色，画笔大小
# refCnts返回绘制了轮廓的图像
# cv2.drawContours(img,refCnts,-1,(0,0,255),3)
# cv_show('img',img)
print (np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0] #排序，从左到右，从上到下
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
	# 计算外接矩形并且resize成合适大小
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# 每一个数字对应每一个模板
	digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#读取输入图像，预处理
image = cv2.imread(args["image"])
# cv_show('image',image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show('gray',gray)

#顶帽操作，突出更明亮的区域
#根据自己指定的核大小进行顶帽操作
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel) 
# cv_show('tophat',tophat)

# ksize:是Sobel算子的大小,即卷积核的大小,必须为奇数,默认值为3（ksize=-1相当于用3*3的）
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

# 绝对值
gradX = np.absolute(gradX)
# 归一化
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print (np.array(gradX).shape)
# cv_show('gradX',gradX)

#通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel) 
# cv_show('gradX',gradX)
#THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
# cv_show('thresh',thresh)

#重复一次闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
# cv_show('thresh',thresh)

# 计算轮廓

# thresh_, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3) 
# cv_show('img',cur_img)
locs = []

# 遍历轮廓
for (i, c) in enumerate(cnts):
	# 计算矩形
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)

	# 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
	if ar > 2.5 and ar < 4.0:

		if (w > 40 and w < 55) and (h > 10 and h < 20):
			#符合的留下来
			locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
	# initialize the list of group digits
	groupOutput = []

	# 根据坐标提取每一个组
	group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	# cv_show('group',group)
	# 预处理
	group = cv2.threshold(group, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# cv_show('group',group)
	# 计算每一组的轮廓
	# group_,digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	digitCnts = contours.sort_contours(digitCnts,
		method="left-to-right")[0]

	# 计算每一组中的每一个数值
	for c in digitCnts:
		# 找到当前数值的轮廓，resize成合适的的大小
		(x, y, w, h) = cv2.boundingRect(c)
		roi = group[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		# cv_show('roi',roi)

		# 计算匹配得分
		scores = []

		# 在模板中计算每一个得分
		for (digit, digitROI) in digits.items():
			# 模板匹配
			result = cv2.matchTemplate(roi, digitROI,
				cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		# 得到最合适的数字
		groupOutput.append(str(np.argmax(scores)))

	# 画出来
	cv2.rectangle(image, (gX - 5, gY - 5),
		(gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
	cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	# 得到结果
	output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)