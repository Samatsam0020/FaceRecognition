import face_recognition
import os


def search_face(face_path, known_faces_dir, tolerance=0.5):

    dossier_visages_connus = known_faces_dir

    visages_connus = []
    noms_visages_connus = []

    for file in os.listdir(dossier_visages_connus):
        file_path = os.path.join(dossier_visages_connus, file)
        image = face_recognition.load_image_file(file_path)

        encodage = face_recognition.face_encodings(image)[0]
        visages_connus.append(encodage)

        noms_visages_connus.append([file, os.path.splitext(file)[0]])

    # Charger et encoder la face à rechercher
    face = face_recognition.load_image_file(
        face_path)
    face_enconding = face_recognition.face_encodings(
        face)[0]

    # Comparaison des vecteurs d'encodage
    correspondances = face_recognition.compare_faces(
        visages_connus, face_enconding, tolerance=tolerance)

    correspondance = [nom for correspondance, nom in zip(
        correspondances, noms_visages_connus) if correspondance]

    return correspondance


print(search_face('image 4.jpg', 'known_faces'))
