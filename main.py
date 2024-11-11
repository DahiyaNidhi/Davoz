from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField
from wtforms.validators import DataRequired, Length
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_user, LoginManager, login_required, current_user, logout_user
from transformers import pipeline
from faster_whisper import WhisperModel
from translate import Translator
import os
import time
import ctypes

# Load necessary libraries for Faster Whisper
ctypes.cdll.LoadLibrary(r'D:\AD_proj\venv\Lib\site-packages\faster_whisper\cudnn_ops_infer64_8.dll')
ctypes.cdll.LoadLibrary(r'D:\AD_proj\venv\Lib\site-packages\faster_whisper\cudnn_cnn_infer64_8.dll')
ctypes.cdll.LoadLibrary(r'D:\AD_proj\venv\Lib\site-packages\faster_whisper\cublas64_12.dll')

# Flask app and config
app = Flask(__name__)
app.config['SECRET_KEY'] = "my-secrets"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///video-meeting.db"
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# User Model for the app
class Register(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(50), unique=True, nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

    def is_active(self):
        return True

    def get_id(self):
        return str(self.id)

    def is_authenticated(self):
        return True

# Initialize database
with app.app_context():
    db.create_all()

# Forms
class RegistrationForm(FlaskForm):
    email = EmailField(label='Email', validators=[DataRequired()])
    first_name = StringField(label="First Name", validators=[DataRequired()])
    last_name = StringField(label="Last Name", validators=[DataRequired()])
    username = StringField(label="Username", validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField(label="Password", validators=[DataRequired(), Length(min=8, max=20)])

class LoginForm(FlaskForm):
    email = EmailField(label='Email', validators=[DataRequired()])
    password = PasswordField(label="Password", validators=[DataRequired()])

# Routes for user registration, login, logout, and dashboard
@login_manager.user_loader
def load_user(user_id):
    return Register.query.get(int(user_id))

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["POST", "GET"])
def login():
    form = LoginForm()
    if request.method == "POST" and form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        user = Register.query.filter_by(email=email, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for("dashboard"))
    return render_template("login.html", form=form)

@app.route("/logout", methods=["GET"])
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully!", "info")
    return redirect(url_for("login"))

@app.route("/register", methods=["POST", "GET"])
def register():
    form = RegistrationForm()
    if request.method == "POST" and form.validate_on_submit():
        new_user = Register(
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            username=form.username.data,
            password=form.password.data
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Account created Successfully! <br>You can now log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", form=form)

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", first_name=current_user.first_name, last_name=current_user.last_name)

@app.route("/meeting")
@login_required
def meeting():
    return render_template("meeting.html", username=current_user.username)

@app.route("/join", methods=["GET", "POST"])
@login_required
def join():
    if request.method == "POST":
        room_id = request.form.get("roomID")
        return redirect(f"/meeting?roomID={room_id}")
    return render_template("join.html")

@app.route('/set_language', methods=['POST'])
def set_language():
    try:
        data = request.get_json()
        preferred_language = data.get('language')
        if not preferred_language:
            return jsonify({'status': 'error', 'message': 'Language not provided'}), 400
        session['preferred_language'] = preferred_language
        print(f"Preferred language is: {preferred_language}")
        return jsonify({'status': 'success', 'redirect_url': url_for('meeting')}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error setting language: {str(e)}'}), 500

# Audio processing and transcription classes and functions
class WhisperTranscriber:
    def __init__(self, model_size='large-v3', sample_rate=44100):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, device='cuda', compute_type='float16')

    def audio_transcription(self, audio_file):
        segments, info = self.model.transcribe(audio_file, beam_size=5)
        whisper_transcription = ""
        for segment in segments:
            whisper_transcription += segment.text + " "
        return whisper_transcription

def process_audio_chunks(language_input):
    translations = {}
    audio_chunk_folder = 'audio_chunks'
    process_times = []
    for audio_file_name in os.listdir(audio_chunk_folder):
        audio_path = os.path.join(audio_chunk_folder, audio_file_name)
        try:
            start_time = time.time()
            translation = voice_to_voice(audio_path, language_input)
            end_time = time.time()
            process_times.append({audio_file_name: end_time - start_time})
            translations[audio_file_name] = translation
        except Exception as e:
            print(f"Error processing {audio_file_name}: {e}")
            translations[audio_file_name] = f"Error: {str(e)}"
    return translations, process_times

def voice_to_voice(audio_file, language_input):
    languages = {'afrikaans':'af', 'arabic':'ar', 'bengali':'bn', 'german':'de', 'english':'en', 'spanish':'es', 'hindi':'hi', 'italian':'it', 'japanese':'ja', 'kannada':'kn', 'korean':'ko', 'russian':'ru', 'chinese':'zh', 'french':'fr'}
    target_language = languages[language_input.lower()]
    transcriber = WhisperTranscriber()
    transcription_response = transcriber.audio_transcription(audio_file)
    pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    language_detected = pipe(transcription_response)
    return text_translation(transcription_response, language_detected[0]['label'], target_language)

def text_translation(text, language_detected, target_language):
    translator = Translator(from_lang=language_detected, to_lang=target_language)
    print(translator.translate(text))
    return translator.translate(text)

UPLOAD_FOLDER = 'audio_chunks'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_chunk = request.files['audio_data']
    chunk_name = request.form['chunk_name']
    chunk_path = os.path.join(UPLOAD_FOLDER, f"{chunk_name}.webm")
    audio_chunk.save(chunk_path)
    return jsonify({"message": "Audio chunk received successfully", "chunk": chunk_name})

@app.route('/process_audio_chunks', methods=['POST'])
def process_chunks():
    language_input = request.form['language_input']
    translations, process_times = process_audio_chunks(language_input)
    return jsonify({
        'translations': translations,
        'processing_times': process_times
    })

@app.route('/process_audio_chunk', methods=['POST'])
def process_audio_chunk():
    chunk_name = request.form['chunk_name']
    language_input = request.form['language_input']
    audio_path = os.path.join(UPLOAD_FOLDER, f"{chunk_name}.webm")
    try:
        translation = voice_to_voice(audio_path, language_input)
        return jsonify(translation)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
