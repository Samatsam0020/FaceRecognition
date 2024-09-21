import face_recognition
import os
import cv2


def extract_face(img_path, faces_dir):
    try:
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)

        for face_location in face_locations:
            top, right, bottom, left = face_location

            face_image = image[top:bottom, left:right]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                f"{faces_dir}/visage_{face_locations.index(face_location)}.jpg", face_image)
    except:
        pass
