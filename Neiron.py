from google.colab import drive
drive.mount('/content/drive')

# ИМпорт библиотек для работы с файловой системой
# для операций с файлами и каталогами (копирование, перемещение, создание, удаление)
import shutil
import os
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator #функция для загрузки картинок и генератор
from tensorflow.keras.models import Sequential #модели из библиотеке Керас
from tensorflow.keras.layers import Conv2D, MaxPooling2D #слои нейросети
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow import keras

from os import listdir, sep
from os.path import abspath, basename, isdir
def tree(dir, padding= '  ', print_files=False):
    """
    Эта функция строит дерево поддиректорий и файлов для заданной директории

    Параметры
    ----------
    dir : str
        Path to needed directory
    padding : str
        String that will be placed in print for separating files levels
    print_files : bool
        "Print or not to print" flag
    """
    cmd = "find '%s'" % dir
    files = os.popen(cmd).read().strip().split('\n')
    padding = '|  '
    for file in files:
        level = file.count(os.sep)
        pieces = file.split(os.sep)
        symbol = {0:'', 1:'/'}[isdir(file)]
        if not print_files and symbol != '/':
            continue
        print (padding*level + pieces[-1] + symbol)


def plot_samples(train_dir, N=4):
  import random
  fig, ax = plt.subplots(2,N,figsize=(5*N,5*2))

  for i,name in enumerate(['buildings','forest']):
    filenames = os.listdir(os.path.join(train_dir,name))

    for j in range(N):
      sample = random.choice(filenames)
      image = load_img(os.path.join(train_dir,name,sample))
      ax[i][j].imshow(image)
      ax[i][j].set_xticks([])
      ax[i][j].set_yticks([])
      ax[i][j].set_title(name)
  plt.grid(False)
  plt.show()

# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
#было первоначаль
epochs = 30
#epochs = 3
# Размер мини-выборки
batch_size = 16
# Количество изображений для обучения
nb_train_samples = 22500
# Количество изображений для проверки
nb_validation_samples = 2500
# Количество изображений для тестирования
nb_test_samples = 2500



base_dir = '/content/drive/MyDrive/Colab Files/buildings-vs-forests'


train_dir = os.path.join(base_dir, 'train')

test_dir = os.path.join(base_dir, 'test')

# Посмотрим, как именно расположены директории с датасетом относительно друг друга
tree(base_dir,print_files=False)

# Посмотрим на содержание датасета при помощи функции plot_samples
plot_samples(train_dir, N=4)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

keras.utils.plot_model(model, 'info.png', show_shapes=True)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    #val_dir,
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
#было первоначаль
epochs = 3
#epochs = 3
# Размер мини-выборки
batch_size = 32
# Количество изображений для обучения
nb_train_samples = 938
# Количество изображений для проверки
nb_validation_samples = 100
# Количество изображений для тестирования
nb_test_samples = 560

history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_validation_samples // batch_size)



# Построим изменение точности на трейновой (оранжевая линия) и тестовой (синяя линия)
# выборках
plt.plot(history.history['val_accuracy'], '-o', label='validation accuracy')
plt.plot(history.history['accuracy'], '--s', label='training accuracy')
plt.legend();


# Построим изменение потерь на трейновой (оранжевая линия) и тестовой (синяя линия)
# выборках
plt.plot(history.history['val_loss'], '-o', label='validation loss')
plt.plot(history.history['loss'], '--s', label='training loss')
plt.legend();

# Взглянем на итоговые результаты классификации на тестовой выборке
# (функция ошибки, точность)
model.evaluate(test_generator)

print("Сохраняем сеть")
# Сохраняем сеть для последующего использования
# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("CAT_and_DOG.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("CAT_and_DOG.h5")
print("Сохранение сети завершено")

#вызов метода для работы с файлами (загрузки)
from google.colab import files


#собственно сохранение
files.download("CAT_and_DOG.json")

#сохраняем веса сети
files.download("CAT_and_DOG.h5")

from keras.models import model_from_json


# посмотрим содержимое директории в колабе с помощью команды:
!ls

#!head Указать имя файла для просмотра просмотр сохраненной нейросети, нужно вписать название файла по факту
!head CAT_and_DOG.json



print("Загружаю сеть из файлов")
# Загружаем данные об архитектуре сети
json_file = open("CAT_and_DOG.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель
loaded_model = model_from_json(loaded_model_json)
# Загружаем сохраненные веса в модель
loaded_model.load_weights("CAT_and_DOG.h5")
print("Загрузка сети завершена")


loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Взглянем на итоговые результаты классификации этой загруженной модели на тестовой выборке
# (функция ошибки, точность). Она по идее совпадает со значениями ранее оубченной и сохраненной сети
loaded_model.evaluate(test_generator)

from IPython.display import Image
from tensorflow.keras.preprocessing import image


#если выше не было этого вызова вызов метода для работы с файлами (загрузки)
from google.colab import files

f = files.upload()

#смотрим, что есть в облачном хранилище, есть ли там наша картинка
!ls

# если хочется посмотреть текущйи каталог]
!pwd

#Покажем файл в облачном хранилище
Image ('/content/People_test/1.jpg')

img_path = '/content/People_test/3.jpg'

def catOrDog(num):
  if num == 1:
    return 'Dog'
  else:
    return 'Cat'


import os
import numpy as np
from pathlib import Path

directory = '/content/People_test'
for entry in Path(directory).iterdir():
  print(entry)
  # Преобразуем картинку в вектор , массив numpy
  Image(entry, width=150, height=150)
  #устанавливаем целевой размер, как и ранее при обучении - 150 на 150
  img = image.load_img(entry, target_size=(150, 150), grayscale=False)
  # Преобразуем изображением в массив numpy и нормализуем
  x = image.img_to_array(img)
  x = 255 - x
  x /= 255
  x = np.expand_dims(x, axis=0)
  prediction = model.predict(x)
  print(catOrDog(round(prediction[0][0])))


# prediction =
print (prediction)

# Попробуем получить доступ к внутренним слоям модели
from keras.models import Model

# Имя слоя берется из Summury нейросети
model2 = Model(model.input, model.get_layer('conv2d_2').output)
# model2= Model(model.input,model.layers[1].output)


preds=model2.predict(x)

# разбираемся в форме результата
np.shape(preds)


# смотрим отдельные элементы , например
preds_a=preds[0,:,:,:]

np.shape(preds_a)

# можно создать еще одну модель, которая будет иметь доступ к другому внутреннему слою
model3= Model(model.input,model.get_layer('max_pooling2d_2').output)

preds3=model3.predict(x)
np.shape(preds3)
preds3_a=preds3[0,:,:]

np.shape(preds3_a)
prob_img=preds3_a[:,:]
np.shape(prob_img[:,:,:1])

model4= Model(model.input,model.get_layer('activation').output)
preds4=model4.predict(x)
preds4_a=preds4[0,:,:]
np.shape(preds4_a)

