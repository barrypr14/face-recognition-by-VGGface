import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
import numpy as np
import dlib
#設定顯示數值的取位
np.set_printoptions(precision=4 , suppress= True)

#匯入模型
nb_class = 5 #預測組別
hidden_dim = 512

vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))

last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
net = Model(vgg_model.input, out)

"""vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3),weights='vggface',model='resnet50')

last_layer = vgg_model.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
out = Dense(nb_class, activation='softmax', name='classifier')(x)
net = Model(vgg_model.input, out)"""

#匯入模型參數
net.load_weights('dataset/face_train/vgg_model_0102_ver2.h5')
#預測類別名稱
class_list = ['Barry','Corn','Liangyu','TUNG','Wenson']

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

#確認攝像頭
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Cannot open camera")
    exit()

index = 10

while (True) :
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # 顯示圖片
    cv2.putText(frame, 'press s to check in ', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    if  index != 10:
        cv2.putText(frame, class_list[index]+' finishes check in ', (10, 250), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    # 按下 q 鍵離開迴圈
    if cv2.waitKey(1) == ord('s') :
        print('start predict')
        #偵測人臉
        face = detector(frame, 1)

        #取出人臉大小
        for i, d in enumerate(face):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()

        height = y2 - y1
        width = x2 - x1
        #切割人臉圖片
        extract = np.zeros((height,width,3) , np.uint8)
        for i in range(0 , height) :
            for j in range (0 , width) : 
                extract[i][j] = frame[y1+i][x1+j]
        #修改大小
        face_image = cv2.resize(extract , (224,224) , interpolation=cv2.INTER_CUBIC)

        img_tensor = image.img_to_array(face_image)
        img_tensor_exp = np.expand_dims(img_tensor , axis = 0)
        #預測是誰
        prediction = net.predict(img_tensor_exp)[0]
        print(prediction)
        #顯示在圖上
        index = np.argmax(prediction)

    elif cv2.waitKey(1) == 27:
        break

# 釋放該攝影機裝置
cap.release()
cv2.destroyAllWindows()