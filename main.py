import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle
import time


class SmartAttendanceSystem:
    def __init__(self, data_path="face_data", attendance_log="attendance_log.csv"):
        self.data_path = data_path
        self.attendance_log = attendance_log
        self.known_face_encodings = []
        self.known_face_names = []
        self.already_marked = set()


        if not os.path.exists(data_path):
            os.makedirs(data_path)


        if not os.path.exists(attendance_log):
            with open(attendance_log, 'w') as f:
                f.write("Name,Date,Time\n")


        self.load_faces()

    def load_faces(self):

        print("Loading registered faces...")

        try:

            if os.path.exists(f"{self.data_path}/face_data.pkl"):
                with open(f"{self.data_path}/face_data.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_names)} faces from pickle file")
                return
        except Exception as e:
            print(f"Error loading pickle data: {e}")

        for filename in os.listdir(self.data_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                try:
                    name = os.path.splitext(filename)[0]
                    image_path = os.path.join(self.data_path, filename)
                    image = face_recognition.load_image_file(image_path)

                    face_encodings = face_recognition.face_encodings(image)
                    if len(face_encodings) > 0:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        print(f"Loaded {len(self.known_face_names)} faces")


        try:
            with open(f"{self.data_path}/face_data.pkl", 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
        except Exception as e:
            print(f"Error saving pickle data: {e}")

    def register_face(self, name):

        print(f"Registering new face for: {name}")


        if name in self.known_face_names:
            print(f"Warning: {name} is already registered. This will update their face data.")

        cap = cv2.VideoCapture(0)
        success = False

        while not success:
            print("Position your face in the frame and press 'c' to capture or 'q' to quit")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break


                cv2.imshow("Register Face - Press 'c' to capture", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
                elif key == ord('c'):

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)

                    if len(face_locations) == 0:
                        print("No face detected. Please try again.")
                        time.sleep(1)
                        break
                    elif len(face_locations) > 1:
                        print("Multiple faces detected. Please ensure only one face is in the frame.")
                        time.sleep(1)
                        break
                    else:

                        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]


                        image_name = f"{name}.jpg"
                        image_path = os.path.join(self.data_path, image_name)
                        cv2.imwrite(image_path, frame)


                        if name in self.known_face_names:
                            idx = self.known_face_names.index(name)
                            self.known_face_encodings.pop(idx)
                            self.known_face_names.pop(idx)


                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)


                        try:
                            with open(f"{self.data_path}/face_data.pkl", 'wb') as f:
                                pickle.dump({
                                    'encodings': self.known_face_encodings,
                                    'names': self.known_face_names
                                }, f)
                        except Exception as e:
                            print(f"Error saving pickle data: {e}")

                        print(f"Successfully registered {name}")
                        success = True
                        break

        cap.release()
        cv2.destroyAllWindows()
        return True

    def mark_attendance(self, name):

        if name in self.already_marked:
            return

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with open(self.attendance_log, 'a') as f:
            f.write(f"{name},{date_str},{time_str}\n")

        self.already_marked.add(name)
        print(f"Marked attendance for {name} at {time_str}")

    def start_recognition(self):

        print("Starting attendance system...")


        if len(self.known_face_encodings) == 0:
            print("No faces registered! Please register at least one face first.")
            return

        cap = cv2.VideoCapture(0)
        process_this_frame = True

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break


            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)


            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)


            if process_this_frame:

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:

                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    name = "Unknown"


                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - face_distances[best_match_index]

                        if matches[best_match_index] and confidence > 0.5:
                            name = self.known_face_names[best_match_index]

                            self.mark_attendance(name)

                    face_names.append(f"{name} ({confidence:.2f})" if name != "Unknown" else name)

            process_this_frame = not process_this_frame


            for (top, right, bottom, left), name in zip(face_locations, face_names):

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4


                color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)


                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)


            cv2.imshow('Smart Attendance System - Press "q" to quit, "r" to register', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):

                cv2.destroyAllWindows()
                name = input("Enter name to register: ")
                if name.strip():
                    self.register_face(name)

                cap.release()
                cap = cv2.VideoCapture(0)

        cap.release()
        cv2.destroyAllWindows()

    def view_attendance_log(self):

        print("\n--- Attendance Log ---")
        try:
            with open(self.attendance_log, 'r') as f:
                print(f.read())
        except Exception as e:
            print(f"Error reading attendance log: {e}")



if __name__ == "__main__":
    attendance_system = SmartAttendanceSystem()

    while True:
        print("\n=== Smart Attendance System ===")
        print("1. Register a new face")
        print("2. Start attendance recognition")
        print("3. View attendance log")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            name = input("Enter name to register: ")
            if name.strip():
                attendance_system.register_face(name)
        elif choice == '2':
            attendance_system.start_recognition()
        elif choice == '3':
            attendance_system.view_attendance_log()
        elif choice == '4':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")