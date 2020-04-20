#載入必要模組

#from sklearn.neighbors import KNeighborsClassifier

#from skimage import exposure

from skimage import feature

#from imutils import paths

#import argparse

#import imutils

import cv2

import numpy as np

#使用參數方式傳入Training和Test的dataset

#ap = argparse.ArgumentParser()

#ap.add_argument("-t", "-test", required=True, help="Path to the test dataset")

#args = vars(ap.parse_args())

import pickle

with open('first_model.pickle', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

# 使用Numpy创建新視窗
img = np.zeros((400,400,3), np.uint8)
# 使用白色填充图片区域
img.fill(255)

while (1):

    try:            

        _, frame = cap.read()


        roi=frame[50:450, 50:450]
        # 感興趣區域
        cv2.rectangle(frame,(50,50),(450,450),(255,0,0),4)
        # 於主畫面顯示特定區域

        hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV) 
        
        lower_HSV = np.array([0,15,0])
        upper_HSV = np.array([70,200,210])
        
        mask = cv2.inRange(hsv,lower_HSV,upper_HSV)
        res = cv2.bitwise_and(roi,roi,mask = mask)
        # 分割後的膚色區域

        in_frame = cv2.bitwise_and(img,img,mask = mask)
        # 遮罩部分

        #cv2.imshow("1",frame)
        #cv2.imshow("2",res)
        #cv2.imshow("3",in_frame)
        #cv2.waitKey(0)


        blurred = cv2.GaussianBlur(in_frame, (7, 7), 0)

                #作threshold(固定閾值)處理

        binaryIMG = cv2.Canny(blurred, 20, 160)  

        #cv2.imshow("3",binaryIMG)     

               #尋找輪廓，只取最大的那個

        (cnts,_) = cv2.findContours(binaryIMG.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        c = max(cnts, key=cv2.contourArea)

            #取出輪廓的長寬高，用來裁切原圖檔。

        (x, y, w, h) = cv2.boundingRect(c)

        Cutted = in_frame[y:y + h, x:x + w]

        #將裁切後的圖檔尺寸更改為500×500。

        Cutted = cv2.resize(Cutted, (400, 400))

            #cv2.imshow("Cutted", Cutted)

   
            #取得其HOG資訊及視覺化圖檔

        (H, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, visualise=True)

        #print(H.shape)

            #使用訓練的模型預測此圖檔

        pred = model.predict(H.reshape(1, -1))[0]

        cv2.putText(frame, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 255, 0), 5,cv2.LINE_AA)

        print("----------------------")
        print(pred.title())

        cv2.imshow("Test Image", frame)
        cv2.imshow("123",res)


    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release() 
   
cv2.destroyAllWindows()
