(使用方式)
使用make_data.py進行訓練樣本錄製
使用data_cut.py對樣本進行分割
將分割好的訓練樣本依照分類丟入train-data中的子資料夾
(可新增或刪除類別)


執行 python train.py -d train-data -n (model_name)
//使用train-data資料夾中的樣本進行訓練
//輸出模型命名為"model_name"


執行 python demo.py -n (model_name)




# https://morvanzhou.github.io/tutorials/machine-learning/sklearn/3-5-save/
# 保存模型參考
#https://chtseng.wordpress.com/2017/01/09/%e4%bd%bf%e7%94%a8%e6%96%b9%e5%90%91%e6%a2%af%e5%ba%a6%e8%88%87knn%e6%a8%a1%e5%9e%8b%e9%80%b2%e8%a1%8c%e5%bd%b1%e5%83%8f%e8%ad%98%e5%88%a5-1/
# 訓練程式參考
