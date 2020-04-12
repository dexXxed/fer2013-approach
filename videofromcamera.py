from __future__ import division
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import cv2


if __name__ == '__main__':
    # загружаем модель
    json_file = open('fer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # загружаем веса в модель
    loaded_model.load_weights("fer.h5")
    print("Loaded model from disk")

    # устанавлимаем необходимые константы
    WIDTH = 48
    HEIGHT = 48
    x = None
    y = None
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    cap = cv2.VideoCapture(0)

    while True:
        # обрабатываем кадр за кадром
        _, full_size_image = cap.read()
        print("Image Loaded")
        gray = cv2.cvtColor(full_size_image, cv2.COLOR_BGR2GRAY)
        face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face.detectMultiScale(gray, 1.3, 10)

        # определяем лица
        for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
                cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # предсказываем эмоцию
                yhat = loaded_model.predict(cropped_img)
                plt.axis('off')
                plt.imshow(cropped_img.reshape(48, 48), interpolation='none', cmap='gray')
                plt.savefig('1.png', bbox_inches='tight', pad_inches=0)
                cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 1, cv2.LINE_AA)
                print("Emotion: " + labels[int(np.argmax(yhat))])

        # показываем кадр с нашим предиктом
        cv2.imshow('Emotions detector (press \'q\' to exit)', full_size_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
