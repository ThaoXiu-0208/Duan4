from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import mediapipe as mp
import json
import os
import numpy as np
import requests

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Các chỉ số landmark cho các đặc điểm khuôn mặt
EYEBROW_LANDMARKS = [33, 133, 153, 154]
MOUTH_LANDMARKS = list(range(61, 81))
NOSE_LANDMARKS = [1, 2, 5, 6, 7, 8, 9, 10, 11]
EAR_LANDMARKS = [234, 454]
HAIR_LANDMARKS = [1, 2, 3]

def get_landmark_positions(face_landmarks, landmark_indices):
    positions = {}
    for index in landmark_indices:
        x = face_landmarks.landmark[index].x
        y = face_landmarks.landmark[index].y
        positions[index] = {'x': x, 'y': y}
    return positions

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def analyze_face_landmarks(face_landmarks):
    landmark_data = {
        "ears": "attached",
        "eyebrows": "up",
        "facialHair": "scruff",
        "hair": "full",
        "nose": "curve",
        "mouth": "smile"
    }

    eyebrow_positions = get_landmark_positions(face_landmarks, EYEBROW_LANDMARKS)
    eyebrow_y_values = [pos['y'] for pos in eyebrow_positions.values()]
    if np.mean(eyebrow_y_values) < 0.5:
        landmark_data["eyebrows"] = "up"
    else:
        if any(pos['y'] < 0.5 for pos in eyebrow_positions.values()):
            landmark_data["eyebrows"] = "eyelashesDown"
        else:
            landmark_data["eyebrows"] = "down"

    mouth_positions = get_landmark_positions(face_landmarks, MOUTH_LANDMARKS)
    mouth_y_values = [pos['y'] for pos in mouth_positions.values()]

    if len(mouth_y_values) < 16:
        landmark_data["mouth"] = "unknown"
    else:
        top_lip_y = mouth_y_values[0:8]
        bottom_lip_y = mouth_y_values[8:16]
        
        top_lip_height = np.max(top_lip_y) - np.min(top_lip_y)
        bottom_lip_height = np.max(bottom_lip_y) - np.min(bottom_lip_y)

        if top_lip_height > 0.02 and bottom_lip_height > 0.02:
            landmark_data["mouth"] = "smile"
        elif top_lip_height < 0.015 and bottom_lip_height < 0.015:
            landmark_data["mouth"] = "frown"
        elif top_lip_height < 0.02 and bottom_lip_height > 0.02:
            landmark_data["mouth"] = "pucker"
        elif top_lip_height > 0.02 and bottom_lip_height < 0.02:
            landmark_data["mouth"] = "smirk"
        elif top_lip_height < 0.015 and bottom_lip_height > 0.03:
            landmark_data["mouth"] = "sad"
        elif top_lip_height > 0.03 and bottom_lip_height < 0.015:
            landmark_data["mouth"] = "laughing"
        elif top_lip_height > 0.015 and bottom_lip_height < 0.015:
            landmark_data["mouth"] = "nervous"
        elif top_lip_height > 0.025 and bottom_lip_height > 0.025:
            landmark_data["mouth"] = "surprised"

    nose_positions = get_landmark_positions(face_landmarks, NOSE_LANDMARKS)
    nose_width = calculate_distance(
        (nose_positions[NOSE_LANDMARKS[0]]['x'], nose_positions[NOSE_LANDMARKS[0]]['y']),
        (nose_positions[NOSE_LANDMARKS[1]]['x'], nose_positions[NOSE_LANDMARKS[1]]['y'])
    )
    if nose_width > 0.05:
        landmark_data["nose"] = "pointed"
    elif nose_width < 0.03:
        landmark_data["nose"] = "round"
    else:
        landmark_data["nose"] = "curve"

    hair_positions = get_landmark_positions(face_landmarks, HAIR_LANDMARKS)
    hair_width = calculate_distance(
        (hair_positions[HAIR_LANDMARKS[0]]['x'], hair_positions[HAIR_LANDMARKS[0]]['y']),
        (hair_positions[HAIR_LANDMARKS[1]]['x'], hair_positions[HAIR_LANDMARKS[1]]['y'])
    )
    if hair_width < 0.05:
        landmark_data["hair"] = "dannyPhantom"
    elif hair_width < 0.1:
        landmark_data["hair"] = "dougFunny"
    elif hair_width < 0.15:
        landmark_data["hair"] = "fonze"
    else:
        landmark_data["hair"] = "full"

    ear_positions = get_landmark_positions(face_landmarks, EAR_LANDMARKS)
    if len(ear_positions) == 2:
        ear_distance = calculate_distance(
            (ear_positions[EAR_LANDMARKS[0]]['x'], ear_positions[EAR_LANDMARKS[0]]['y']),
            (ear_positions[EAR_LANDMARKS[1]]['x'], ear_positions[EAR_LANDMARKS[1]]['y'])
        )
        if ear_distance > 0.2:
            landmark_data["ears"] = "detached"
        else:
            landmark_data["ears"] = "attached"

    return landmark_data

def generate_avatar(landmark_data):
    base_url = "https://api.dicebear.com/9.x/micah/svg"
    params = {
        "mouth": landmark_data.get("mouth", "smile"),
        "ears": landmark_data.get("ears", "attached"),
        "eyebrows": landmark_data.get("eyebrows", "up"),
        "facialHair": landmark_data.get("facialHair", "scruff"),
        "hair": landmark_data.get("hair", "full"),
        "size": 96
    }
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        with open("static/avatar.svg", "wb") as f:
            f.write(response.content)
        return "/static/avatar.svg"
    else:
        return None

def process_image(image_path):
    image = cv2.imread(image_path)
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmark_data = analyze_face_landmarks(face_landmarks)
            avatar_url = generate_avatar(landmark_data)
            return {
                'feature_data': landmark_data,
                'avatar_url': avatar_url
            }
        else:
            return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    result = process_image(file_path)
    if result:
        return jsonify({
            'feature_data': result['feature_data'],
            'avatar_url': result['avatar_url']
        })
    else:
        return jsonify({'error': 'Failed to generate avatar'}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
