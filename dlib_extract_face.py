import dlib
import image_to_numpy
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import os 

datadir = 'dataset/face_train_2/Stranger/'

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

images = os.listdir(datadir)
count = 0

for image in images :
    print('manage ' + str(image))
    img = image_to_numpy.load_image_file(datadir+image)

    b,g,r = cv2.split(img)           # get b, g, r
    img = cv2.merge([r,g,b]) 

    # 偵測人臉
    face = detector(img, 1)

    # 取出所有偵測的結果
    for i, d in enumerate(face):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

    height = y2 - y1
    width = x2 - x1

    #print(height , width)
    #print(x1,x2,y1,y2)
    #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

    #建立新空矩陣圖片
    extract = np.zeros((height,width,3) , np.uint8)

    for i in range(0 , height) :
        for j in range (0 , width) : 
            extract[i][j] = img[y1+i][x1+j]

    cv2.imwrite('dataset/face_train_2/Stranger/'+str(count)+'.png' , extract)
    count += 1