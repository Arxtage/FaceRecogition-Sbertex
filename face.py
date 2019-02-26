#!/usr/bin/env python
import face_recognition
import os
import pickle
import numpy as np


class Face():
    def __init__(self):
        self.IMAGES = []
        self.MAX_COUNT = 60
        self.encodings = []

    def webcam(self):
        import cv2
        self.IMAGES = []
        face_detector = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.namedWindow('preview')
        vc = cv2.VideoCapture(0)
        count = 0
        while vc.isOpened():
            if count < self.MAX_COUNT // 3:
                id = 'Plain face'
            elif count < 2 * self.MAX_COUNT // 3:
                id = 'Show me your smile'
            else:
                id = 'Turn up your head'
            img = vc.read()[1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(20, 20)
            )
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, id, (x + 30, y + w + 30), font, 1, (255, 255,
                                                                     255), 2)
            self.IMAGES.append(img)
            cv2.imshow('preview', img)
            key = cv2.waitKey(100)
            if count == self.MAX_COUNT or key == 27: # exit on ESC
                break
            count += 1
        cv2.destroyWindow('preview')

    def get_faces(self):
        from tqdm import tqdm
        self.encodings = []
        for _ in tqdm(self.IMAGES, 'encoding'):
            self.encodings += face_recognition.face_encodings(_)[:1]

    def test_faces(self):
        import random
        random.shuffle(self.encodings)
        part = round(len(self.encodings) * 0.75)
        train = self.encodings[:part]
        test = self.encodings[part:]
        for _ in test:
            if min(face_recognition.face_distance(train, _)) > 0.6:
                raise ValueError("We can't recognize you always")
        print('Faces are valid')

    def dump(self, id):
        dirname = 'data'
        os.makedirs(dirname, exist_ok=True)
        length = len(self.encodings)
        ans = np.empty((length, length))
        for i, _ in enumerate(self.encodings):
            ans[i] = face_recognition.face_distance(self.encodings, _)
        with open(self.filename(id), 'wb') as f:
            pickle.dump((ans.mean(), ans.std(), self.encodings), f)

    def filename(self, id):
        dirname = 'data'
        os.makedirs(dirname, exist_ok=True)
        return os.path.join('data', f'{id}.pkl')

    def compare_id_face(self, id, pic):
        img = face_recognition.load_image_file(pic)
        try:
            enc = face_recognition.face_encodings(img)[0]
        except IndexError:
            raise IndexError('Face in your photo is unrecognizable')
        with open(self.filename(id), 'rb') as f:
            mean, std, e = pickle.load(f)
            L = face_recognition.face_distance(e, enc)
        num = (L < 0.6).sum()
        length = len(L)
        if num < length // 6:
            return False
        if num > 5 * length // 6:
            return True
        return (abs(L.mean() - mean) / (std + L.std())) < 1


face = Face()
#face.webcam()
#face.get_faces()
#face.test_faces()
#face.dump(50)

print(face.compare_id_face(50, '1.jpg'))
print(face.compare_id_face(50, '2.jpg'))
print(face.compare_id_face(50, '3.jpg'))
print(face.compare_id_face(50, '4.jpg'))
print(face.compare_id_face(50, '5.jpg'))
print(face.compare_id_face(50, '6.jpg'))
print(face.compare_id_face(50, '7.jpg'))
