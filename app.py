import cv2
import face_recognition
import numpy as np
import os
import pickle
from flask import Flask, request, jsonify, render_template, send_from_directory
from datetime import datetime
import base64
import json
import socket
import psutil


app = Flask(__name__)

DATA_PATH = "face_data"
ENCODINGS_FILE = os.path.join(DATA_PATH, "face_encodings.pkl")
LOG_FILE = "attendance_log.csv"
KNOWN_ENCODINGS = []
KNOWN_NAMES = []
TOLERANCE = 0.6

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as f:
        f.write("Name,Date,Time\n")

def load_encodings():
    global KNOWN_ENCODINGS, KNOWN_NAMES
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                KNOWN_ENCODINGS = data['encodings']
                KNOWN_NAMES = data['names']
            print(f"Loaded {len(KNOWN_NAMES)} face encodings")
        except Exception as e:
            print(f"Error loading encodings: {e}")
    else:
        print("No encodings file found. Starting with empty database.")

def save_encodings():
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({
            'encodings': KNOWN_ENCODINGS,
            'names': KNOWN_NAMES
        }, f)

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    today_marked = False
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            for line in f:
                if name in line and date_str in line:
                    today_marked = True
                    break

    if not today_marked:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{name},{date_str},{time_str}\n")
        print(f"Marked attendance for {name}")
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
def register():
    if 'face_image' not in request.files or 'name' not in request.form:
        return jsonify({'success': False, 'message': 'Missing image or name'})

    try:
        name = request.form['name']
        image_file = request.files['face_image']

        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)

        if len(face_locations) == 0:
            return jsonify({'success': False, 'message': 'No face detected in the image'})

        if len(face_locations) > 1:
            return jsonify(
                {'success': False, 'message': 'Multiple faces detected. Please use an image with only one face'})

        face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]

        if len(KNOWN_ENCODINGS) > 0:
            matches = face_recognition.compare_faces(KNOWN_ENCODINGS, face_encoding, tolerance=TOLERANCE)
            if True in matches:
                match_index = matches.index(True)
                return jsonify(
                    {'success': False, 'message': f'This face is already registered as {KNOWN_NAMES[match_index]}'})

        KNOWN_ENCODINGS.append(face_encoding)
        KNOWN_NAMES.append(name)
        save_encodings()

        os.makedirs(os.path.join(DATA_PATH, 'images'), exist_ok=True)
        image_path = os.path.join(DATA_PATH, 'images', f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        cv2.imwrite(image_path, img)

        return jsonify({'success': True, 'message': f'Successfully registered {name}'})

    except Exception as e:
        print(f"Error during registration: {e}")
        return jsonify({'success': False, 'message': f'Error during registration: {str(e)}'})

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    if not request.json or 'image' not in request.json:
        return jsonify({'success': False, 'message': 'No image provided'})

    try:
        img_data = request.json['image']
        if img_data.startswith('data:image'):
            img_data = img_data.split(',')[1]

        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)

        if not face_locations:
            return jsonify({'success': True, 'faces': []})

        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        faces = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            attendance_marked = False

            if KNOWN_ENCODINGS:
                face_distances = face_recognition.face_distance(KNOWN_ENCODINGS, face_encoding)
                best_match_index = np.argmin(face_distances)

                if face_distances[best_match_index] <= TOLERANCE and get_wifi_ip()=="192.168.161.12":
                    name = KNOWN_NAMES[best_match_index]
                    attendance_marked = mark_attendance(name)

            top, right, bottom, left = face_location
            faces.append({
                'name': name,
                'location': [top, right, bottom, left],
                'attendance_marked': attendance_marked
            })

        return jsonify({'success': True, 'faces': faces})

    except Exception as e:
        print(f"Error during recognition: {e}")
        return jsonify({'success': False, 'message': f'Error during recognition: {str(e)}'})

@app.route('/api/attendance/today')
def today_attendance():
    today = datetime.now().strftime("%Y-%m-%d")
    attendees = set()

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[1] == today:
                    attendees.add(parts[0])

    return jsonify({
        'success': True,
        'count': len(attendees),
        'attendees': list(attendees)
    })

@app.route('/api/attendance/report', methods=['POST'])
def attendance_report():
    if not request.json or 'date' not in request.json:
        return jsonify({'success': False, 'message': 'No date provided'})

    selected_date = request.json['date']
    report_data = []

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[1] == selected_date:
                    report_data.append({
                        'name': parts[0],
                        'date': parts[1],
                        'time': parts[2]
                    })

    return jsonify({
        'success': True,
        'date': selected_date,
        'entries': report_data
    })

@app.route('/api/registered')
def registered_count():
    return jsonify({
        'success': True,
        'count': len(KNOWN_NAMES),
        'names': KNOWN_NAMES
    })

@app.route('/api/recent_activity')
def recent_activity():
    activities = []

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            next(f)
            lines = list(f)[-10:]
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    activities.append({
                        'name': parts[0],
                        'date': parts[1],
                        'time': parts[2]
                    })

    activities.reverse()
    return jsonify({
        'success': True,
        'activities': activities
    })

def get_wifi_ip():
    wifi_keywords = ['wi-fi', 'wlan', 'wireless']

    interfaces = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    for interface_name in interfaces:
        name_lower = interface_name.lower()
        if any(keyword in name_lower for keyword in wifi_keywords):
            if stats[interface_name].isup:
                for snic in interfaces[interface_name]:
                    if snic.family == socket.AF_INET:
                        return f"{snic.address}"
    return "false"

if __name__ == '__main__':
    load_encodings()
    app.run(debug=True)