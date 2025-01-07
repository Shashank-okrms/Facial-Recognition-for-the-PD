import cv2
import numpy as np
import pickle
import face_recognition
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate(r"key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://spaceman-87147-default-rtdb.asia-southeast1.firebasedatabase.app/"
})
ref = db.reference('Persons')

print("Loading Encode File ...")
with open('EncodeFile.p', 'rb') as file:
    encode_list_known_with_ids = pickle.load(file)
encode_list_known, person_ids = encode_list_known_with_ids
print("Encode File Loaded.")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(img_small_rgb)
    face_encodings = face_recognition.face_encodings(img_small_rgb, face_locations)

    for encode_face, face_loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_distance = face_recognition.face_distance(encode_list_known, encode_face)

        if len(face_distance) > 0:
            match_index = np.argmin(face_distance)
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            if matches[match_index]:
                person_id = person_ids[match_index]
                person_data = ref.child(person_id).get()
                tag = person_data['tag']
                color = (0, 0, 255) if tag == "CRIMINAL" else (0, 255, 0)
                label = f"{tag}: {person_id}"
            else:
                color = (0, 255, 0)
                label = "Unknown"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()