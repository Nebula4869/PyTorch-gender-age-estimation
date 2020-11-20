import numpy as np
import onnxruntime
import dlib
import cv2


GENDER_DICT = {0: 'male', 1: 'female'}

onnx_session = onnxruntime.InferenceSession('models-2020-11-20-14-37/best-epoch47-0.9314.onnx')
detector = dlib.get_frontal_face_detector()

img = cv2.imread('test.jpg')
face_rects = detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0)

for face_rect in face_rects:
    cv2.rectangle(img,
                  (face_rect.left(), face_rect.top()),
                  (face_rect.right(), face_rect.bottom()),
                  (255, 255, 255))
    face = img[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right(), :]
    inputs = np.transpose(cv2.resize(face, (64, 64)), (2, 0, 1))
    inputs = np.expand_dims(inputs, 0).astype(np.float32) / 255.
    predictions = onnx_session.run(['output'], input_feed={'input': inputs})[0][0]
    gender = GENDER_DICT[int(np.argmax(predictions[:2]))]
    age = int(predictions[2])
    cv2.putText(img, 'Gender: {}, Age: {}'.format(gender, age), (face_rect.left(), face_rect.top()), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    print('Gender: {}, Age: {}'.format(gender, age))

cv2.imshow('', img)
cv2.waitKey()
