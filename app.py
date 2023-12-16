import streamlit as st
import face_recogntion
import cv2
import numpy as np
import pickle

pickle1 = open('known_face_encodings.pkl','rb')
known_face_encodings = pickle.load(pickle1)
pickle2 = open('known_face_names.pkl','rb')
known_face_name = pickle.load(pickle2)
pickle1.close()
pickle2.close()



face_locations = []
face_encodings = []
face_names = []


st.header('RCEE :: FACE RECOGNTION')
st.title('AI&DS')

image = st.file_uploader('Pick any Image')
if image:
    st.image(image)
    image = np.array(image)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image,face_locations)

    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            face_names.append(name)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    

    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
