import numpy as np
import face_recognition
import os

faces = []
files = os.listdir('faces')
for file in files:
    faces.append(np.load("faces/"+file))

picture_of_me = face_recognition.load_image_file("mayur.jpeg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

results = face_recognition.compare_faces(faces, my_face_encoding, 0.4)

print(results)

if results[0]:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
