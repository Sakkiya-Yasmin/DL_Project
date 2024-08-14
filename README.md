# DL_Project


# **Importing libraries**


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
import os
from skimage.io import imread
from skimage .transform import resize
import warnings
warnings.filterwarnings("ignore")

os.listdir('/content/drive/MyDrive/Helmet_detection')

datapath='/content/drive/MyDrive/Helmet_detection'

"""# **Loading and preprocessing data**"""

X = []
y = []

Categories=['without helmet', 'with helmet']

for dir in Categories:
  dirpath=os.path.join(datapath,dir)
  images=os.listdir(dirpath)
  for img in images:
    imgpath=os.path.join(dirpath,img)
    img_arr=imread(imgpath)
    img_resize=resize(img_arr,(128,128,1))
    X.append(img_resize)
    y.append(Categories.index(dir))
  print("Loaded...",dir)

"""# **Splitting data into training and testing sets**"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

len(X_train),len(y_train)

len(X_test),len(y_test)

# converting the lists to numpy arrays
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

"""# **Building the neural network model**




*   4 Conv2D layers (2 with 128 filters and 2 with 64 filters)
*   2 BatchNormalization layers
*   2 MaxPooling2D layers
*   3 Dropout layers (2 with dropout rate 0.25 and 1 with dropout rate 0.5)
*   1 Flatten layer
*   2 Dense layers (1 with 32 neurons and 1 with 2 neurons)





"""

model = Sequential()

model.add(Conv2D(128, kernel_size=(2, 2), input_shape=(128, 128, 1), padding = 'Same'))
model.add(Conv2D(128, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs = 30, batch_size = 64, verbose = 1,validation_data = (X_test, y_test))

"""# **Prediction**



"""

new_img_path='/content/drive/MyDrive/img.jpg'

new_img=imread(new_img_path)
resized=resize(new_img,(128,128,1))
reshaped=resized.reshape(1,128,128,1)

new_img_pred=model.predict(reshaped)

ind=new_img_pred.argmax(axis=1)
ind

Categories[ind.item()]

model.save("my_model.h5")
