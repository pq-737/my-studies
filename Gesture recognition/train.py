#載入必要模組

from sklearn.neighbors import KNeighborsClassifier

from skimage import exposure

from skimage import feature

from imutils import paths

import argparse

import imutils

import cv2


#使用參數方式傳入Training和Test的dataset

ap = argparse.ArgumentParser()

ap.add_argument("-d", "-train-data", required=True, help="Path to the logos training dataset")

ap.add_argument("-n", "-model_name", required=True, help="Path to the test dataset")

args = vars(ap.parse_args())

#data用來存放HOG資訊，labels則是存放對應的標籤

data = []

labels = []

#依序讀取train-data中的圖檔

for imagePath in paths.list_images(args["d"]):

        print(imagePath)

        #將資料夾的名稱取出作為該圖檔的標籤

        make = imagePath.split("\\")[-2]

        #—-以下為訓練圖檔的預處理—-#

        #載入圖檔,轉為灰階,作模糊化處理

        image = cv2.imread(imagePath)

        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(image, (7, 7), 0)

            #作threshold(固定閾值)處理

        binaryIMG = cv2.Canny(blurred,20,160)

        #(T, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

            #使用Canny方法偵側邊緣

        #dged = imutils.auto_canny(thresh)

        #尋找輪廓，只取最大的那個

        (cnts, _) = cv2.findContours(binaryIMG.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        c = max(cnts, key=cv2.contourArea)

        #取出輪廓的長寬高，用來裁切原圖檔。

        (x, y, w, h) = cv2.boundingRect(c)

        Cutted = image[y:y + h, x:x + w]

    #將裁切後的圖檔尺寸更改為500×500。

        Cutted = cv2.resize(Cutted, (400, 400))

        #cv2.imshow("Cutted", Cutted)
    
#—-訓練圖檔預處理結束—-#

        #取得其HOG資訊及視覺化圖檔

        (H, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10),

                cells_per_block=(2, 2), transform_sqrt=True, visualise=True)

        #將HOG資訊及標籤分別放入data及labels陣列

        data.append(H)

        labels.append(make)

#使用KNN模型來訓練

model = KNeighborsClassifier(n_neighbors=3)

#傳入data及labels陣列開始訓練

model.fit(data, labels)

name = args["n"]

print(name)

# 匯入模型打包套件
import pickle
# 模型打包
with open(name+".pickle","wb") as f:
    pickle.dump(model,f)

print("模型訓練完成")

