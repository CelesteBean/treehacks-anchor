#!/bin/bash
cd ~/treehacks-anchor
source venv/bin/activate

echo "=== Starting Dashboard ==="
python -m src.viz.judges_window &
sleep 3

echo "=== Starting Content Analyzer (lightweight, ~200MB) ==="
python -m src.core.content_analyzer --interval 5 &
sleep 15

echo "=== Starting Speech Recognition (loading Whisper...) ==="
python -m src.core.speech_recognition &
sleep 10

echo "=== Starting Audio Capture ==="
python -m src.core.audio_capture --device 0 --native-rate 44100 --seconds 3600
