# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:53:42 2019

@author: user
"""

import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 使用 XVID 編碼    
out = cv2.VideoWriter("output.avi", fourcc, 30.0, (640, 480))
# 建立 VideoWriter 物件，輸出影片至 output.avi
# FPS 值為 30.0，解析度為 640x360 
   

# 使用Numpy创建新視窗
img = np.zeros((480,640,3), np.uint8)
# 使用黑色填充图片区域
img.fill(255)



while(1):  
    ret,frame = cap.read()
    
     
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) 
    
    lower_HSV = np.array([0,15,0])
    upper_HSV = np.array([70,200,210])
    
    mask = cv2.inRange(hsv,lower_HSV,upper_HSV)
    #res = cv2.bitwise_and(frame,frame,mask = mask)
    # 分割後的膚色區域
    res = cv2.bitwise_and(img,img,mask = mask)
    # 遮罩部分
    
    
    
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    
    out.write(res)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
out.release()   
cv2.destroyAllWindows()


