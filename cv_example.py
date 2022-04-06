# 示例源码地址
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
# 多目标匹配

import cv2 as cv
import numpy as np

img_rgb = cv.imread('resource\\all.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('resource\\dot.png',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img_rgb)