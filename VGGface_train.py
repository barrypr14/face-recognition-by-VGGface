from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam
import os
import tensorflow as tf 
import datetime
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

#custom parameters
nb_class = 5
hidden_dim = 512

datasetdir = "dataset/face_train"
os.chdir(datasetdir)  ##開啟檔案位置

imgdatagen = ImageDataGenerator(validation_split = 0.2 ,
                                width_shift_range = 0.3,
                                height_shift_range = 0.3,
                                brightness_range = [0.2,1.0],
                                rotation_range = 45)


batch_size = 4

height, width = (224,224)

train_dataset = imgdatagen.flow_from_directory(os.getcwd(),target_size = (height, width), 
                    classes = ('Barry','Corn','Liangyu','TUNG','Wenson'),batch_size = batch_size,
                    shuffle = True,subset = 'training',class_mode = 'sparse')

val_dataset = imgdatagen.flow_from_directory(os.getcwd(),target_size = (height, width), 
                    classes = ('Barry','Corn','Liangyu','TUNG','Wenson'),batch_size = batch_size,
                    shuffle = True , subset = 'validation',class_mode = 'sparse')


vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3),weights='vggface')

last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, out)

for i in range (len(custom_vgg_model.layers)-3) :
    custom_vgg_model.layers[i].trainable = False

print(custom_vgg_model.summary())

custom_vgg_model.compile(
    loss='sparse_categorical_crossentropy',#'categorical_crossentropy',
    optimizer=Adam(lr=0.0001),
    metrics=['accuracy']
)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0 ,write_graph = False,
                                                        update_freq = batch_size)
history=custom_vgg_model.fit(train_dataset,
        steps_per_epoch = train_dataset.samples // batch_size ,
        validation_data = val_dataset , 
        validation_steps = val_dataset.samples // batch_size,
        epochs = 10 ,  
        callbacks = [tensorboard_callback])

custom_vgg_model.save('vgg_model_test.h5')