# загружаем json и модель для оценки точности
from __future__ import division
from keras.models import model_from_json
import numpy as np


if __name__ == '__main__':
    json_file = open('fer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # Загрузим веса в новую модель
    loaded_model.load_weights("fer.h5")
    print("Loaded model from disk")

    truey = []
    predy = []
    x = np.load('./modXtest.npy')
    y = np.load('./modytest.npy')

    yhat = loaded_model.predict(x)
    yh = yhat.tolist()
    yt = y.tolist()
    count = 0

    for i in range(len(y)):
        yy = max(yh[i])
        yyt = max(yt[i])
        predy.append(yh[i].index(yy))
        truey.append(yt[i].index(yyt))
        if yh[i].index(yy) == yt[i].index(yyt):
            count += 1

    acc = (count / len(y)) * 100

    # сохраняем значения для дальнейшей визуализации
    np.save('truey', truey)
    np.save('predy', predy)
    print("Предугаданные и реальзыне значения сохранены")
    print("Точность:" + str(acc) + "%")