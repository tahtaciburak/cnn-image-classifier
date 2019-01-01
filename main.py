
# coding: utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import os
import numpy as np
from PIL import Image
import random

def one_hot_labeling(class_name):
    if class_name == "headphone":
        return(np.array([1,0,0,0,0,0,0,0,0,0]))
    elif class_name == "butterfly":
        return(np.array([0,1,0,0,0,0,0,0,0,0]))
    elif class_name == "cup":
        return(np.array([0,0,1,0,0,0,0,0,0,0]))
    elif class_name == "airplanes":
        return(np.array([0,0,0,1,0,0,0,0,0,0]))
    elif class_name == "dolphin":
        return(np.array([0,0,0,0,1,0,0,0,0,0]))
    elif class_name == "cellphone":
        return(np.array([0,0,0,0,0,1,0,0,0,0]))
    elif class_name == "car_side":
        return(np.array([0,0,0,0,0,0,1,0,0,0]))
    elif class_name == "laptop":
        return(np.array([0,0,0,0,0,0,0,1,0,0]))
    elif class_name == "pizza":
        return(np.array([0,0,0,0,0,0,0,0,1,0]))
    elif class_name == "Motorbikes":
        return(np.array([0,0,0,0,0,0,0,0,0,1]))

def read_dataset(path):
    IMG_SIZE = 200
    classes = os.listdir(path)
    classes.remove('.DS_Store')
    dataset = []
    
    for cls in classes:
        images = os.listdir(path+"/"+cls)
        for img in images:
            image = Image.open(path+"/"+"/"+cls+"/"+img)
            image = image.convert('L')
            image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            
            dataset.append([np.array(image),one_hot_labeling(cls)])

    random.shuffle(dataset)
    return dataset
dataset = read_dataset('Dataset')

train_dataset = dataset[int(len(dataset)/5):]
test_dataset = dataset[:int(len(dataset)/5)]

print(len(train_dataset))
print(len(test_dataset))

IMG_SIZE = 200
trainImages = np.array([i[0] for i in train_dataset]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array([i[1] for i in train_dataset])


IMG_SIZE = 200
testImages = np.array([i[0] for i in test_dataset]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
testLabels = np.array([i[1] for i in test_dataset])


model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.fit(trainImages,trainLabels, batch_size = 50, epochs = 2, verbose = 1)

test_loss, test_acc = model.evaluate(testImages, testLabels)

print('Test accuracy:', test_acc)

predictions = model.predict(testImages)

for p in predictions:
    print(np.argmax(p))

def id_to_label(c_id):
    if c_id == 0:
        return "headphone"
    elif c_id == 1:
        return "butterfly"
    elif c_id == 2:
        return "cup"
    elif c_id == 3:
        return "airplane"
    elif c_id == 4:
        return "dolphin"
    elif c_id == 5:
        return "cell_phone"
    elif c_id == 6:
        return "car_side"
    elif c_id == 7:
        return "laptop"
    elif c_id == 8:
        return "pizza"
    elif c_id == 9:
        return "Motorbikes"

IMG_SIZE = 200
def test_one_image(path):
    image = Image.open(path)
    image = image.convert('L')
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    predictions = model.predict(np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 1))
    return (id_to_label(np.argmax(predictions)))

for f in os.listdir("validasyon"):
    if f != ".DS_Store":
        print("File -> ",f," Prediction ->",test_one_image("validasyon/"+f))
