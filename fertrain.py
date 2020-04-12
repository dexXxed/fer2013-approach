import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2


if __name__ == '__main__':
    num_features = 64
    num_labels = 7
    batch_size = 64
    epochs = 100
    width, height = 48, 48

    x = np.load('./fdataX.npy')
    y = np.load('./flabels.npy')

    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    # для просмотра картинок
    # for xx in range(10):
    #    plt.figure(xx)
    #    plt.imshow(x[xx].reshape((48, 48)), interpolation='none', cmap='gray')
    # plt.show()

    # разделяем на обучающую, тестовую и валидирующую выборки
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.7, random_state=41)

    # сохраняем тестовые сэмплы, чтобы их использовать в дальнейшем
    np.save('modXtest', X_test)
    np.save('modytest', y_test)

    # пропишем саму CNN
    model = Sequential()

    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(2*2*2*num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2*2*num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2*num_features, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels, activation='softmax'))

    # model.summary()  # просмотрим структуру модели

    # компилируем модель с adam оптимизатором и categorical crossentropy loss-ом
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    # обучаем модель
    model.fit(np.array(X_train), np.array(y_train),
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(np.array(X_valid), np.array(y_valid)),
              shuffle=True)

    # сохраняем для дальнейшего использования
    fer_json = model.to_json()

    with open("fer.json", "w") as json_file:
        json_file.write(fer_json)

    model.save_weights("fer.h5")
    print("Saved model to disk!")
