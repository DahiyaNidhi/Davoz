from flask import Flask, request, jsonify
import os
import assemblyai as aai
from translate import Translator
from transformers import pipeline
import time
from pydub import AudioSegment

app = Flask(__name__)

def process_audio_chunks(language_input):
    translations = {}
    audio_chunk_folder = 'audio_chunks'
    process_times = []

    # Process each audio chunk file one by one
    for audio_file_name in os.listdir(audio_chunk_folder):
        audio_path = os.path.join(audio_chunk_folder, audio_file_name)

        # Call voice_to_voice for each chunk and collect the translation
        try:
            start_time = time.time()  # Start time
            translation = voice_to_voice(audio_path, language_input)
            end_time = time.time()  # End time

            processing_time = end_time - start_time
            process_times.append({audio_file_name: processing_time})
            
            translations[audio_file_name] = translation
        except Exception as e:
            print(f"Error processing {audio_file_name}: {e}")
            translations[audio_file_name] = f"Error: {str(e)}"

    return translations, process_times

def voice_to_voice(audio_file, language_input):
    languages = {'afrikaans':'af','arabic':'ar', 'bengali':'bn', 'german':'de', 'english':'en', 'spanish':'es', 'hindi':'hi', 'italian':'it', 'japanese':'ja', 'kannada':'kn', 'korean':'ko', 'russian':'ru', 'chinese':'zh'}
    target_language = languages[language_input.lower()]

    # Transcribe audio
    transcription_response, language_detected = audio_transcription(audio_file)

    if transcription_response.status == aai.TranscriptStatus.error:
        return jsonify({'error': transcription_response.error})
    else:
        text = transcription_response.text

    target_language_translation = text_translation(text, language_detected, target_language)
    return target_language_translation

def audio_transcription(audio_file):
    start_time = time.time()  # Start time for transcription
    aai.settings.api_key = "43f9318c3f4b4ee3bc20f236391bc3d4"
    config = aai.TranscriptionConfig(language_detection=True)
    transcriber = aai.Transcriber(config=config)
    transcription = transcriber.transcribe(audio_file)
    end_time = time.time()
    print(f"Time taken for audio_transcription: {end_time - start_time} seconds")
    start_time = time.time()
    pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    language_detected = pipe(transcription.text)
    end_time = time.time()  # End time for transcription

    print(f"Time taken for text_detection: {end_time - start_time} seconds")
    return transcription, language_detected[0]['label']

def text_translation(text, language_detected, target_language):
    start_time = time.time()  # Start time for translation
    translator = Translator(from_lang=language_detected, to_lang=target_language)
    translated_text = translator.translate(text)
    end_time = time.time()  # End time for translation

    print(f"Time taken for text_translation: {end_time - start_time} seconds")
    return translated_text

# Folder to store audio files
UPLOAD_FOLDER = 'audio_chunks'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return app.send_static_file('index(3)(1).html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_chunk = request.files['audio_data']
    chunk_name = request.form['chunk_name']
    
    # Save the audio file
    chunk_path = os.path.join(UPLOAD_FOLDER, f"{chunk_name}.webm")
    audio_chunk.save(chunk_path)
    return jsonify({"message": "Audio chunk received successfully", "chunk": chunk_name})

@app.route('/process_audio_chunks', methods=['POST'])
def process_chunks():
    # Get the input language from the request
    language_input = request.form['language_input']
    
    # Process all audio chunks and return the translations and processing times
    translations, process_times = process_audio_chunks(language_input)
    return jsonify({
        'translations': translations,
        'processing_times': process_times
    })

if __name__ == "__main__":
    app.run(debug=True)
