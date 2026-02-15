#!/bin/bash
cd ~/treehacks-anchor
source venv/bin/activate

echo "=== Starting Dashboard ==="
python -m src.viz.judges_window &
sleep 3

echo "=== Starting Stress Detector (loading wav2vec2...) ==="
python -m src.core.stress_detector &
sleep 30

echo "=== Starting Tactic Inference (loading Qwen...) ==="
python -m src.core.tactic_inference --interval 10 &
sleep 45

echo "=== Starting Speech Recognition (loading Whisper...) ==="
python -m src.core.speech_recognition &
sleep 10

echo "=== Starting Audio Capture ==="
python -m src.core.audio_capture --device 0 --native-rate 44100 --seconds 3600
