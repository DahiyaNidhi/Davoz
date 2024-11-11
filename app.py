import gradio as gr
import assemblyai as aai
from translate import Translator
import uuid
from pathlib import Path
from transformers import pipeline

def voice_to_voice(audio_file, language_input):
    languages = {'afrikaans':'af','arabic':'ar', 'bengali':'bn', 'german':'de', 'english':'en', 'spanish':'es', 'hindi':'hi', 'italian':'it', 'japanese':'ja', 'kannada':'kn', 'korean':'ko', 'russian':'ru', 'chinese':'zh'}
    target_language = languages[language_input.lower()]
    #transcribe audio
    transcription_response, language_detected = audio_transcription(audio_file)
    #transcription_response = audio_transcription(audio_file)

    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        text = transcription_response.text

    target_language_translation = text_translation(text, language_detected, target_language)
    #target_language_translation = text_translation(text, target_language)
    return target_language_translation


def audio_transcription(audio_file):

    aai.settings.api_key = "43f9318c3f4b4ee3bc20f236391bc3d4"
    config = aai.TranscriptionConfig(language_detection=True)
    transcriber = aai.Transcriber(config=config)
    transcription = transcriber.transcribe(audio_file)
    pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    language_detected=pipe(transcription.text)
    return transcription, language_detected[0]['label']
    #return transcription

def text_translation(text, language_detected, target_language):
#def text_translation(text, target_language):    
    translator = Translator(from_lang=language_detected, to_lang=target_language)
    translated_text = translator.translate(text)

    return translated_text

audio_input = gr.Audio(
    sources=["microphone"],
    type="filepath",
)

language_input = gr.Textbox(label="Enter the preferred language")

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=[audio_input, language_input],
    outputs='text'
)

if __name__ == "__main__":
    demo.launch()