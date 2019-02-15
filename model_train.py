from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os, cv2, csv, numpy as np

def shape_of_array(arr):
    array = np.array(arr)
    return array.shape

def get_label(num):
    with open('./human_ratings.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['#Image'] == str(num):
                return float(row['Attractiveness label'])

def load_image_data(filedir):
    label = []
    image_data_list = []
    train_image_list = os.listdir(filedir)

    for img in train_image_list:
        url = os.path.join(filedir + img)
        image = cv2.imread(url)
        image_data_list.append(image)

        img_num = int(img[:img.find('.')])
        att_label = get_label(img_num)
        label.append(att_label)

    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255.0
    return img_data, label


def make_network():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

# Training:
train_x, train_y = load_image_data('./face128_train/')
model = make_network()
model.summary()
model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['mae'])
hist = model.fit(train_x, train_y, batch_size=100, epochs=100, verbose=1)
scores=model.evaluate(train_x, train_y)
print('mse=%f, mae=%f' % (scores[0],scores[1]))

model.save('model_450faces.h5')
