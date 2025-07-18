from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition
import base64
from datetime import datetime
import db_operations
import os

app = Flask(__name__, static_url_path='')
CORS(app)

# Directory for storing student images
STUDENT_IMAGES_DIR = 'student_images'

# Create the directory if it doesn't exist
os.makedirs(STUDENT_IMAGES_DIR, exist_ok=True)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/registration.html')
def registration():
    return send_from_directory('.', 'registration.html')

@app.route('/student_images/<path:filename>')
def serve_student_image(filename):
    return send_from_directory(STUDENT_IMAGES_DIR, filename)

@app.route('/mark_attendance')
def start_attendance():
    # Load images and encodings from MongoDB
    images, roll_names, encodings_known = db_operations.load_student_data()
    
    if not images:
        return jsonify({"error": "No student profiles found in database"})
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Ensuring DirectShow backend for better compatibility
    if not cap.isOpened():
        return jsonify({"error": "Camera could not be opened"})
    
    # Track recognized students to avoid duplicate entries in the same session
    recognized_students = set()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_small = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        faces_cur_frame = face_recognition.face_locations(frame_rgb)
        encodings_cur_frame = face_recognition.face_encodings(frame_rgb, faces_cur_frame)
        
        for (top, right, bottom, left), encoding in zip(faces_cur_frame, encodings_cur_frame):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            matches = face_recognition.compare_faces(encodings_known, encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(encodings_known, encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
            
            if best_match_index is not None and matches[best_match_index]:
                roll_name = roll_names[best_match_index]
                roll_no, name = roll_name.split("_")
                
                # Create a unique key for this student
                student_key = f"{roll_no}_{name}"
                
                # Mark attendance only once per session
                if student_key not in recognized_students:
                    recognized_students.add(student_key)
                    # Mark attendance in MongoDB and CSV
                    db_operations.mark_student_attendance(roll_no, name)
                
                # Display bounding box and name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({roll_no})", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        cv2.imshow('Attendance System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Attendance marking complete", "students_marked": list(recognized_students)})

@app.route('/saveProfile', methods=['POST'])
def save_profile():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        roll_no = data['id']
        fullname = data['fullname']
        
        # Decode base64 image
        img_binary = base64.b64decode(image_data)
        
        # Save to database and file system
        result = db_operations.save_student_profile(roll_no, fullname, img_binary)
        
        if result["success"]:
            return jsonify({"message": result["message"], "file_path": result.get("file_path", "")})
        else:
            return jsonify({"error": result["message"]}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    result = db_operations.get_all_attendance_records()
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify({"error": result["error"]}), 500

@app.route('/get_students', methods=['GET'])
def get_students():
    result = db_operations.get_all_students()
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify({"error": result["error"]}), 500

@app.route('/get_student_image/<roll_no>', methods=['GET'])
def get_student_image(roll_no):
    result = db_operations.get_student_image(roll_no)
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify({"error": result["error"]}), 404

if __name__ == '__main__':
    app.run(debug=True)