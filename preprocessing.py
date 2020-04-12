import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    data = pd.read_csv('./fer2013.csv')
    width, height = 48, 48
    datapoints = data['pixels'].tolist()

    # получаем все свойства для дальнейшего обучения
    X = []
    for xseq in datapoints:
        xx = [int(xp) for xp in xseq.split(' ')]
        xx = np.asarray(xx).reshape(width, height)
        X.append(xx.astype('float32'))

    X = np.asarray(X)
    X = np.expand_dims(X, -1)

    # получем лэйблы для обучения
    y = pd.get_dummies(data['emotion']).values

    # сохраняем как numpy файлы
    np.save('fdataX', X)
    np.save('flabels', y)

    print("Предобработка выполнена")
    print("Кол-во черт: " + str(len(X[0])))
    print("Кол-во лэйблов (категорий): " + str(len(y[0])))
    print("Кол-во примеров в датасете:" + str(len(X)))
    print("X и y сохранены в fdataX.npy и flabels.npy соответственно")
