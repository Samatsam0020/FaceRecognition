import face_recognition
import os
import streamlit as st


def isFaceExists(img):
    face_location = face_recognition.face_locations(img)
    if len(face_location) == 0:
        return False
    return True


def count_passengers(image, known_faces, tolerance=0.4, ):
    count = 0
    if isFaceExists(image):
        face_locations = face_recognition.face_locations(image)

        new_faces = []
        face_encodings = face_recognition.face_encodings(
            image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                known_faces,  face_encoding, tolerance=tolerance)
            if not any(matches):
                known_faces.append(face_encoding)
                new_faces.append(face_encoding)
                count += 1

    return count
