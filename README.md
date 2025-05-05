# DA VOZ

DA VOZ is a multilingual real-time audio transcription and translation pipeline designed to make spoken communication accessible across languages and formats. It leverages advanced speech recognition, language detection, and neural machine translation to convert voice to text and translate it into the desired language-fast, accurately, and at scale.


## Overview

DA VOZ addresses the challenge of converting and translating spoken audio in real-time. This system has been developed through three iterative models, each improving on performance, latency, and scalability:

- **Model 1:** Baseline model using Gradio + AssemblyAI API for transcription and translation.
- **Model 2:** Introduced audio chunking and parallel processing to support long audio streams with incremental transcription and translation.
- **Model 3:** Implemented Faster Whisper for faster, lower-latency transcription with improved scalability and deployment readiness.


## Key Features

- **Real-time Audio Transcription:** Converts speech to text in real-time using powerful ASR models.
- **Multilingual Translation:** Supports transcription and translation in multiple languages using the NLLB 200 model.
- **Incremental Processing:** Efficient chunk-wise handling of long audios using audio segmentation.
- **Faster Whisper Integration:** Boosts speed and reduces computational load for seamless real-time response.
- **Modular Architecture:** Easily extendable system with pluggable ASR, translation, and UI modules.
- **Scalable Deployment:** Designed with GPU optimization, multiprocessing, and caching mechanisms.


## How It Works

### Model 1: Baseline (Gradio + AssemblyAI)
- Accepts audio via Gradio interface.
- Transcribes audio using AssemblyAI API.
- Detects language and translates using Facebook’s NLLB 200 model.
- Suitable for short audio clips with high transcription accuracy.

### Model 2: Audio Chunking + Parallel Translation
- Splits long audio into overlapping chunks (e.g., 15s with 3s overlap).
- Processes chunks in parallel using `multiprocessing` for fast execution.
- Performs transcription, language detection, and translation on each chunk.
- Combines translated text into a cohesive output.
- Addresses latency and memory issues of Model 1.

### Model 3: Faster Whisper Integration
- Replaces AssemblyAI with Faster Whisper for low-latency, offline transcription.
- Optimized to run on GPU and reduced hardware requirements.
- Maintains chunk-wise processing while delivering near real-time feedback.
- Best suited for production-grade performance and deployment.


## Tools & Technologies

- **Python**, **PyTorch**, **TensorFlow**
- **AssemblyAI**, **Faster Whisper**
- **Facebook NLLB 200** (Translation)
- **Gradio** (UI for Model 1)
- **FFmpeg**, **Pydub** (Audio processing)
- **Multiprocessing**, **Concurrency** (Parallel processing)
- **GPU Optimization** with `fp16` inference


## Evaluation & Performance

- Evaluated on multiple short and long audio samples across English, Hindi, and Spanish.
- Model 3 showed ~40% latency reduction vs Model 2.
- Translation accuracy dependent on transcription quality and source language.
