# 识别信用卡的数字
# 导入工具包
from imutils import contours
import  numpy as np
import argparse
import imutils
import cv2
import myutils

# 设置参数
# ap=argparse.ArgumentParser()  #  argparse 模块可以让人轻松编写用户友好的命令行接口
# ap.add_argument("-i","--image",required=True,
#                 help="path to input image")
# ap.add_argument("-t","--template",required=True,
#                 help="path to template OCR-A image")
args= {"template":"ocr_a_reference.png","image":"credit_card_01.png"}

# 指定信用卡类型
FIRST_NUMBER={
    "3":"American Express",
    "4":"Visa",
    "5":"MasterCard",
    "6":"Discover Card"
}

# 绘图显示函数
def cv_show(name,img):
    """
    用于显示图片
    :param name: 图片名字
    :param img: image对象
    :return: NULL
    """
    cv2.imshow(name,img)
    cv2.waitKey(0)  # 任意键退出
    cv2.destroyAllWindows()

# 读取一个模板图像
# 其中args["template"]是从命令行中获取模板路径参数
img =cv2.imread(args["template"])
#cv_show("template_image",img)

# 灰度图--将图片转换成灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv_show("GRAY_Image",ref)

# 二值图像--将灰度图转换成二值图像
ref=cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
#cv_show("ref",ref)

# 计算轮廓
# cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,
# cv2.RETR_EXTERNAL只检测外轮廓，
# cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
# 返回的list中每个元素都是图像中的一个轮廓
# 新版本opencv只返回2个参数
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# 在信用卡上画出轮廓
cv2.drawContours(img,refCnts,-1,(0,0,255),3)
#cv_show("refCnts",img)
print(np.array(refCnts).shape)
# 排序，从左到右，从上到下 --对轮廓进行排序 得到 0123456789的轮廓元组
refCnts= myutils.sort_contours(refCnts,method="left-to-right")[0]
digits={}

# 遍历每一个轮廓
# 此时元组中是0123456789模板的轮廓
# i是index
for (i,c) in enumerate(refCnts):
    # 计算外接矩形并resize成合适的大小
    (x,y,w,h)=cv2.boundingRect(c)
    # 每个数字模版外接矩形区域
    roi=ref[y:y+h,x:x+w]
    # 调整大小方便输出调试
    roi=cv2.resize(roi,(57,88))
    # 将数字模板和对应关系放在一个字典中
    digits[i]=roi
    #cv_show("num{}".format(i),roi)

# 初始化卷积核,及形态学操作的单位大小
rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# 读取待识别图像，预处理
image = cv2.imread(args["image"])
#cv_show("orignal Iamge" , image)
image = myutils.resize(image , width=300)
# 转换成对应灰度图
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv_show("gray",gray)

# 礼帽操作,突出更加明亮的区域（得到噪声区域)--即信用卡上突出的文字信息
tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
#cv_show("tophat",tophat)

# 使用sobel算子进行边缘检测
#ksize=-1相当于使用3*3
gradX=cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
gradX=np.absolute(gradX)
(minVal,maxVal)=(np.min(gradX),np.max(gradX))
gradX=(255*((gradX-minVal)/(maxVal-minVal)))
gradX=gradX.astype("uint8")

print(np.array(gradX).shape)
#cv_show("gradx",gradX)

# 通过闭操作（先膨胀，在腐蚀），将相邻的四位数字连在一个区域
gradX=cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
#cv_show("after MORPH_Close",gradX)

# ThRESH_OTSU会自动寻找合适的阈值，适合双峰，需要办阈值参数设置为0
thresh = cv2.threshold(gradX,0,255,
                       cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
#cv_show("after thresh",thresh)

# 通过闭操作（先膨胀，在腐蚀），将相邻的四位数字连在一个区域
thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,rectKernel)
#cv_show("after MORPH_CLOSE",thresh)
# 计算轮廓
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

# 在待识别原图中画出轮廓
cnts= threshCnts
cur_img=image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
#cv_show("image",cur_img)
locs=[]

for (i,c) in enumerate(cnts):
    # 获取轮廓的外界矩形
    (x,y,w,h)=cv2.boundingRect(c)
    ar=w/float(h)
    # 选择合适的区域(根据实际任务筛选)，本项目为四个连续数字的区域
    if ar > 2.5 and ar < 4.0:

        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 符合的留下来
            locs.append((x, y, w, h))

#将符合规律的轮廓从左到右排序
locs=sorted(locs,key=lambda x:x[0])
output=[]

# 遍历轮廓中的每一个数字
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    groupOutput=[]

    #提取数字区域(稍微扩大，保证不遗漏）
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    #cv_show('group', group)

    #对每一个区域再次进行轮廓检测，切分出每一个数字的区域
    # 二值化
    group = cv2.threshold(group, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv_show('group', group)
    # 计算每一组的轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]

    # 遍历该区域的轮廓数组，并于模板中012345789进行匹配
    for c in digitCnts:
        # 获取外接矩形
        (x,y,w,h)=cv2.boundingRect(c)
        #用于调试时显示每一个数字的区域
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        #cv_show('roi', roi)

        #计算匹配得分
        scores=[]

        #在模板中计算每一个得分（匹配度)
        for(digit,digitRoi) in digits.items():
            # 模板匹配
            result=cv2.matchTemplate(roi,digitRoi,cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 得到最合适的数字(匹配度最高的对应的数字--字典中的key)
        groupOutput.append(str(np.argmax(scores)))
    # 在原图上画出来
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