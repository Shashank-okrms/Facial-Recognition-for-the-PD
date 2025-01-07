import os
import pickle
import cv2
import numpy as np
import face_recognition
import firebase_admin
from firebase_admin import credentials, db, storage

cred = credentials.Certificate(r"key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://spaceman-87147-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "frp_bucket_1"
})

bucket = storage.bucket()
ref = db.reference('Persons')

def get_next_person_id():
    """
    Get the next available unique person ID from Firebase, or the existing ID if the person has been previously added.
    """
    ids = ref.get().keys() if ref.get() else []
    existing_ids = [key.replace('ID_', '') for key in ids]  # Remove 'ID_' prefix to compare numerically
    
    if existing_ids:
        max_id = max([int(id.split('_')[0]) for id in existing_ids])  # Split ID and convert to int
    else:
        max_id = 0
    
    new_id = f"ID_{max_id + 1}"
    
    # Check if this ID already exists
    if new_id in existing_ids:
        new_id = f"ID_{max_id + 2}"  # Increment to avoid existing ID
    
    return new_id

def save_face_data(images, ids, tags):
    """
    Saves encoded faces and metadata (ID and tag) to Firebase.
    """
    def encode_faces(images_list):
        encode_list = []
        for img in images_list:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img_rgb)[0]
            encode_list.append(encode)
        return encode_list

    encode_list = encode_faces(images)
    encoded_data = [encode_list, ids]

    with open("EncodeFile.p", 'wb') as file:
        pickle.dump(encoded_data, file)

    print("Encodings saved successfully!")

    for idx, person_id in enumerate(ids):
        filename = f'Images/{person_id}.jpg'
        filepath = os.path.join("Images", f"{person_id}.jpg")
        cv2.imwrite(filepath, images[idx])
        blob = bucket.blob(filename)
        
        existing_blob = bucket.blob(filename)
        if existing_blob.exists():
            existing_blob.delete()
        
        is_upload_successful = blob.upload_from_filename(filepath)

        ref.child(person_id).set({"tag": tags[idx]})

    print("Data uploaded to Firebase.")

def process_video_stream():
    """
    Opens camera feed and processes faces.
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Camera initialized. Press 'p' to capture a picture, 'q' to quit.")

    person_images = []
    person_ids = []
    person_tags = []
    capture_next_frame = True

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to access camera.")
            break

        cv2.imshow("Video Stream", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            cap.release()
            cv2.destroyAllWindows()

            face_locations = face_recognition.face_locations(img)
            if not face_locations:
                print("No faces found in the current frame.")
            else:
                for face_loc in face_locations:
                    y1, x2, y2, x1 = face_loc
                    face_img = img[y1:y2, x1:x2]

                    # Display the cropped face to the user
                    cv2.imshow("Cropped Face", face_img)

                    tag = get_user_tag()
                    if tag is not None:
                        tag_label = "CRIMINAL" if tag == 1 else "NON-CRIMINAL"
                        person_id = get_next_person_id()
                        person_id_with_tag = f"{person_id}_{tag_label}"
                        
                        # Check if the person_id already exists in the database
                        existing_person = ref.child(person_id_with_tag).get()
                        if existing_person:
                            # Replace existing image
                            filename = f'Images/{person_id}.jpg'
                            filepath = os.path.join("Images", f"{person_id}.jpg")
                            cv2.imwrite(filepath, face_img)
                            blob = bucket.blob(filename)
                            if bucket.blob(filename).exists():
                                bucket.blob(filename).delete()
                            is_upload_successful = blob.upload_from_filename(filepath)
                            ref.child(person_id_with_tag).set({"tag": tag_label})
                            print(f"Updated existing person ID: {person_id_with_tag}")
                        else:
                            # New person
                            person_images.append(face_img)
                            person_ids.append(person_id_with_tag)
                            person_tags.append(tag_label)
                            print(f"ID: {person_id_with_tag}, Tag: {tag_label} - Saved.")

            break

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    return person_images, person_ids, person_tags

def get_user_tag():
    """
    Prompts the user to tag the face as 'CRIMINAL' or 'NON-CRIMINAL'.
    """
    print("Is this person a CRIMINAL or NON-CRIMINAL? (1 for CRIMINAL, 2 for NON-CRIMINAL): ")
    tag = int(input())

    if tag in [1, 2]:
        return tag
    return None

if __name__ == "__main__":
    print("Starting face capture and data collection...")
    images, ids, tags = process_video_stream()
    save_face_data(images, ids, tags)
    print("Process complete.")