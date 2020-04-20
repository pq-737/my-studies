import cv2
import os

def get_images_from_video(video_name, time_F, save_path):
    #video_images = []
    vc = cv2.VideoCapture(video_name)
    c = 1
    i = 1
    
    if vc.isOpened(): #判斷是否開啟影片
        rval, video_frame = vc.read()
    else:
        rval = False

    while rval:   #擷取視頻至結束
        rval, video_frame = vc.read()
        
        if(c % time_F == 0): #每隔幾幀進行擷取
            cv2.imwrite(os.path.join(save_path,"tr"+ str(i) +".png"), video_frame)
            #video_images.append(video_frame)
            i = i + 1 
        c = c + 1
    vc.release()
    
    #return video_images
    return 0

time_F = 6 #time_F越小，取樣張數越多



video_path = "output.avi"


save_path = 'D:/108-2/IMG/cut_png' #切片後儲存路徑

get_images_from_video(video_path, time_F, save_path)

cv2.destroyAllWindows