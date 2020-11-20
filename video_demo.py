import numpy as np
import onnxruntime
import dlib
import cv2


GENDER_DICT = {0: 'male', 1: 'female'}

onnx_session = onnxruntime.InferenceSession('models-2020-11-20-14-37/best-epoch47-0.9314.onnx')
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face_rects = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
        for face_rect in face_rects:
            cv2.rectangle(frame,
                          (face_rect.left(), face_rect.top()),
                          (face_rect.right(), face_rect.bottom()),
                          (255, 255, 255))
            face = frame[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right(), :]
            inputs = np.transpose(cv2.resize(face, (64, 64)), (2, 0, 1))
            inputs = np.expand_dims(inputs, 0).astype(np.float32) / 255.
            predictions = onnx_session.run(['output'], input_feed={'input': inputs})[0]
            gender = int(np.argmax(predictions[0, :2]))
            age = int(predictions[0, 2])
            cv2.putText(frame, 'Gender: {}, Age: {}'.format(['Male', 'Female'][gender], age), (face_rect.left(), face_rect.top()), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        cv2.imshow('', frame)
        cv2.waitKey(1)
