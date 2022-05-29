from email.mime import image
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

rad = st.sidebar.radio("Navigation", ["Mark Attendance", "New User Registration", "Attendance Report"])

if rad == "Mark Attendance":
    st.title("FACE RECOGNITION ATTENDANCE SYSTEM :bust_in_silhouette: :memo:")
    st.write()
    run = st.checkbox('Start')
    window = st.image([])
    path = 'images'
    images = []
    userName = []
    userList = os.listdir(path)
    # print(userList)

    for cu_img in userList:
        current_img = cv2.imread(f'{path}/{cu_img}')
        images.append(current_img)
        userName.append(os.path.splitext(cu_img)[0])
    # print(userName)

    def FaceEncode(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList
    
    def markAttend(name):
        with open('Attendance.csv','r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
    
    encodeListKnown = FaceEncode(images)
    print("All Encodings Completed!!!")

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

        facesCurrentFrame = face_recognition.face_locations(faces)
        encodeCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

        for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = userName[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 2)
                cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                markAttend(name)

        window.image(frame)

    else:
        st.write('Click on start to record your attendance')
if rad == "New User Registration":
    image_file = st.file_uploader("Upload an Image (File name should be name of the user)")
    if image_file is not None:
        with open(os.path.join("images", image_file.name), "wb") as f:
            f.write(image_file.getbuffer())
        st.write("New Employee Registered! :smile:")

if rad == "Attendance Report":
    data = pd.read_csv("Attendance.csv")
    st.table(data)
