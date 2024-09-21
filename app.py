import streamlit as st
import cv2
import face_recognition
from search_face import search_face
from recognize import recognize_faces_realtime, load_know_encodings, isFaceExists, isFaceSaved
import os
import numpy as np
from PIL import Image
from count import count_passengers


st.sidebar.title("Paramètres")

menu = ["Realtime", "Recherche_visage"]
choice = st.sidebar.selectbox("", menu)

# Tolerance = st.sidebar.slider("Tolerance", 0.0, 1.0, 0.5, 0.01)

if choice == "Recherche_visage":
    st.title("Recherche image dans la base de données")

    st.text("Fonctionnalité bientôt disponible ... (dispo juste sur console python) ")

    # uploaded_image = st.file_uploader(
    #     "Upload", type=['jpg', 'png', 'jpeg'])

    # if uploaded_image is not None:
    #     img_array = convert_to_array(uploaded_image)
    #     c = search_face(img_array, 'known_faces')
    #     st.text(c)

    # else:
    #     st.info("upload an image")


elif choice == 'Realtime':
    st.title("RealTime Face Recognition")
    known_encodings = list(load_know_encodings().values())
    known_names = list(load_know_encodings().keys())

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    FRAME_WINDOW = st.image([])
    i = 0
    known_faces = []
    count = 0
    previous = None
    st.write("Nombre passagers")
    while True:
        ret, frame = cam.read()
        c = count_passengers(frame, known_faces)
        if isFaceExists(frame):
            if not isFaceSaved(frame, known_encodings):
                cv2.imwrite(f"photosproof/image {i}.jpg", frame)
                cv2.imwrite(f"known_faces/image {i}.jpg", frame)
                i += 1

        count += c
        if count != previous:
            st.text(count)

        previous = count

        if not ret:
            st.error("Failed to capture frame from camera")
            st.info(
                "Please turn off the other app that is using the camera and restart app")
            st.stop()
        image, name = recognize_faces_realtime(
            frame, known_encodings, known_names)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(image)
