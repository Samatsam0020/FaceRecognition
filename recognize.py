import cv2
import face_recognition
import os


def load_know_encodings(known_faces_dir="known_faces"):
    dossier_visages_connus = known_faces_dir

    # Chargement et encodage des visages connus
    visages_connus = []
    noms_visages_connus = []

    for file in os.listdir(dossier_visages_connus):
        file_path = os.path.join(dossier_visages_connus, file)
        image = face_recognition.load_image_file(file_path)

        encodage = face_recognition.face_encodings(image)[0]
        visages_connus.append(encodage)

        noms_visages_connus.append(os.path.splitext(file)[0])

    dico = dict(zip(noms_visages_connus, visages_connus))

    return dico


def recognize_faces_realtime(image, known_encodings, known_names, tolerance=0.4):

    name = 'Unknown'

    face_locations = face_recognition.face_locations(image)

    face_encodings = face_recognition.face_encodings(
        image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=tolerance)
        name = "Inconnu"

        distance = face_recognition.face_distance(
            known_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)

            name = known_names[first_match_index]
            distance = round(distance[first_match_index], 2)
            cv2.putText(image, str(distance), (left, top-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return image, name


def isFaceExists(img):
    face_location = face_recognition.face_locations(img)
    if len(face_location) == 0:
        return False
    return True


def isFaceSaved(img, known_encodings, tolerance=0.5):

    face_locations = face_recognition.face_locations(img)

    face_encodings = face_recognition.face_encodings(
        img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=tolerance)

        if True not in matches:
            return False

        else:
            return True
