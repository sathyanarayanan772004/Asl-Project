from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, session
import cv2
import numpy as np
import logging
from datetime import datetime
import json
import os
import pyttsx3
import threading
from asl_recognition import ASLRecognizer
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///asl_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the ASL Recognition app.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Admin access required.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asl_logs.txt'),
        logging.StreamHandler()
    ]
)

# Initialize ASL recognizer
asl_recognizer = ASLRecognizer()

# TTS lock to prevent concurrent access
tts_lock = threading.Lock()

# Store recent predictions and accumulated sentence
recent_letters = []
accumulated_letters = []
current_sentence = ""

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_letters')
def get_letters():
    return jsonify(recent_letters[-10:])  # Return last 10 letters

@app.route('/get_sentence')
def get_sentence():
    return jsonify({
        'sentence': current_sentence,
        'accumulated_letters': accumulated_letters
    })

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    global current_sentence
    if current_sentence:
        # Run TTS in a separate thread to avoid blocking
        def speak():
            try:
                with tts_lock:
                    # Create new TTS engine instance for each request
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 0.9)
                    engine.say(current_sentence)
                    engine.runAndWait()
                    engine.stop()
                    del engine
            except Exception as e:
                logging.error(f"TTS Error: {e}")
        
        thread = threading.Thread(target=speak)
        thread.daemon = True
        thread.start()
        
        logging.info(f"Speaking: {current_sentence}")
        return jsonify({'status': 'success', 'message': f'Speaking: {current_sentence}'})
    else:
        return jsonify({'status': 'error', 'message': 'No sentence to speak'})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global accumulated_letters, current_sentence
    accumulated_letters = []
    current_sentence = ""
    logging.info("Sentence cleared")
    return jsonify({'status': 'success', 'message': 'Sentence cleared'})

@app.route('/add_space', methods=['POST'])
def add_space():
    global accumulated_letters, current_sentence
    accumulated_letters.append(' ')
    current_sentence = ''.join(accumulated_letters)
    logging.info("Space added to sentence")
    return jsonify({'status': 'success', 'sentence': current_sentence})

@app.route('/add_manual_text', methods=['POST'])
def add_manual_text():
    global accumulated_letters, current_sentence
    data = request.get_json()
    text = data.get('text', '')
    
    if text:
        # Add space before text if sentence is not empty
        if accumulated_letters and accumulated_letters[-1] != ' ':
            accumulated_letters.append(' ')
        
        # Add the text
        for char in text:
            accumulated_letters.append(char)
        
        current_sentence = ''.join(accumulated_letters)
        logging.info(f"Manual text added: {text} | Current sentence: {current_sentence}")
        return jsonify({'status': 'success', 'sentence': current_sentence})
    else:
        return jsonify({'status': 'error', 'message': 'No text provided'})

def generate_frames():
    global accumulated_letters, current_sentence
    import time
    
    cap = None
    # Try multiple methods to open camera
    camera_backends = [
        (0, cv2.CAP_DSHOW),  # DirectShow (Windows)
        (1, cv2.CAP_DSHOW),  # Try second camera with DirectShow
        (0, cv2.CAP_MSMF),   # Media Foundation (Windows)
        (1, cv2.CAP_MSMF),   # Second camera with Media Foundation
        (0, None),           # Default backend camera 0
        (1, None),           # Default backend camera 1
    ]
    
    for cam_index, backend in camera_backends:
        try:
            if backend is not None:
                cap = cv2.VideoCapture(cam_index, backend)
                logging.info(f"Trying camera {cam_index} with backend {backend}")
            else:
                cap = cv2.VideoCapture(cam_index)
                logging.info(f"Trying camera {cam_index} with default backend")
            
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    logging.info(f"Successfully opened camera {cam_index}")
                    break
                else:
                    cap.release()
                    cap = None
        except Exception as e:
            logging.error(f"Error with camera {cam_index}: {e}")
            if cap:
                cap.release()
            cap = None
    
    # Check if camera is opened successfully
    if cap is None or not cap.isOpened():
        logging.error("Failed to open any camera")
        # Generate a blank frame with error message
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Camera Error: Cannot access camera", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', blank_frame)
        frame = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Discard first few frames (camera warm-up)
    for _ in range(5):
        cap.read()
    
    time.sleep(0.1)
    logging.info("Camera opened successfully and ready")
    
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            logging.warning("Failed to read frame from camera")
            break
        
        try:
            # Process frame and get prediction
            processed_frame, letter = asl_recognizer.process_frame(frame)
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            processed_frame = frame
            letter = None
        
        if letter and letter not in ['nothing', 'del']:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            letter_data = {'letter': letter, 'timestamp': timestamp}
            recent_letters.append(letter_data)
            
            # Handle special cases
            if letter == 'space':
                accumulated_letters.append(' ')
            elif letter == 'del':
                # Remove last character if exists
                if accumulated_letters:
                    accumulated_letters.pop()
            else:
                # Add regular letter
                accumulated_letters.append(letter)
            
            # Update current sentence
            current_sentence = ''.join(accumulated_letters)
            
            # Log the detected letter
            logging.info(f"Detected letter: {letter} | Current sentence: {current_sentence}")
            
            # Keep only last 50 letters in memory
            if len(recent_letters) > 50:
                recent_letters.pop(0)
        
        # Display current sentence on frame
        if current_sentence:
            cv2.putText(processed_frame, f"Sentence: {current_sentence}", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        try:
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                logging.error("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logging.error(f"Error encoding frame: {e}")
            break
    
    cap.release()
    logging.info("Camera released")

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=True)
            user.update_last_login()
            flash('Login successful!')
            
            # Redirect to admin dashboard if admin user
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long')
            return render_template('signup.html')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('signup.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('signup.html')
        
        # Create new user
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    users = User.query.all()
    user_stats = {
        'total_users': len(users),
        'admin_users': len([u for u in users if u.is_admin]),
        'regular_users': len([u for u in users if not u.is_admin])
    }
    return render_template('admin_dashboard.html', users=users, stats=user_stats)

def create_admin_user():
    """Create default admin user if it doesn't exist"""
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin_user = User(username='admin', email='admin@asl.com', password='admin', is_admin=True)
        db.session.add(admin_user)
        db.session.commit()
        print("Default admin user created: username='admin', password='admin'")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_admin_user()
    app.run(debug=True, threaded=True)