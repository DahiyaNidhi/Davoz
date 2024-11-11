# Use a pipeline as a high-level helper
from transformers import pipeline
from faster_whisper import WhisperModel
import ctypes
ctypes.cdll.LoadLibrary(r'D:\AD_proj\venv\Lib\site-packages\faster_whisper\cudnn_ops_infer64_8.dll')
ctypes.cdll.LoadLibrary(r'D:\AD_proj\venv\Lib\site-packages\faster_whisper\cudnn_cnn_infer64_8.dll')
ctypes.cdll.LoadLibrary(r'D:\AD_proj\venv\Lib\site-packages\faster_whisper\cublas64_12.dll')


from flask import Flask, request, jsonify
import os
import assemblyai as aai
from translate import Translator
from transformers import pipeline
import time
from pydub import AudioSegment

app = Flask(__name__)

class WhisperTranscriber:
    def __init__(self, model_size='large-v3', sample_rate=44100):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, device='cuda', compute_type='float16')
    def audio_transcription(self, audio_file):
        segments, info = self.model.transcribe(audio_file, beam_size=5)
        whisper_transcription=""
        for segment in segments:
            print(segment.text)
            whisper_transcription += segment.text + " "
        return whisper_transcription

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
    start_time = time.time()
    transcriber = WhisperTranscriber()
    transcription_response  = transcriber.audio_transcription(audio_file)
    end_time = time.time()
    print(transcription_response)
    print("Time taken for audio transcription: ", end_time-start_time)
    start_time = time.time()
    pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    language_detected = pipe(transcription_response)
    end_time = time.time()
    target_language_translation = text_translation(transcription_response, language_detected[0]['label'], target_language)
    return target_language_translation


def text_translation(text, language_detected, target_language):
    print("Entered in text translation function")
    language_detected = language_detected
    target_language = target_language
    start_time = time.time()  # Start time for translation
    translator = Translator(from_lang=language_detected, to_lang=target_language)
    translated_text = translator.translate(text)
    # Use a pipeline as a high-level helper
    #pipe = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")
    #translated_text = pipe(text, src_lang=language_detected, tgt_lang=target_language)
    end_time = time.time()  # End time for translation
    print(translated_text)
    print(f"Time taken for text_translation: {end_time - start_time} seconds")
    #print(translated_text[0]['translation_text'][5:])
    #return translated_text[0]['translation_text'][5:]
    return translated_text

# Folder to store audio files
UPLOAD_FOLDER = 'audio_chunks'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return app.send_static_file('index(3)(2).html')

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

@app.route('/process_audio_chunk', methods=['POST'])
def process_audio_chunk():
    # Get chunk name and language from request
    chunk_name = request.form['chunk_name']
    language_input = request.form['language_input']

    # Path to the audio chunk
    audio_path = os.path.join(UPLOAD_FOLDER, f"{chunk_name}.webm")

    try:
        # Process the audio chunk for transcription and translation
        translation = voice_to_voice(audio_path, language_input)
        return jsonify(translation)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
