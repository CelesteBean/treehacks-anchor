#!/bin/bash
cd ~/treehacks-anchor && source venv/bin/activate
pkill -9 -f "src.core" 2>/dev/null || true
pkill -9 -f "src.viz" 2>/dev/null || true
fuser -k 5555/tcp 5556/tcp 5557/tcp 5558/tcp 8080/tcp 2>/dev/null || true
sleep 3
rm -f /tmp/*.log
echo "=== Starting Audio Capture ===" && free -h | head -2
python -m src.core.audio_capture --device 0 --native-rate 44100 --seconds 3600 >> /tmp/audio.log 2>&1 &
sleep 5
echo "=== Starting Speech Recognition ===" && free -h | head -2
python -m src.core.speech_recognition >> /tmp/speech.log 2>&1 &
sleep 15
echo "=== Starting Stress Detector ===" && free -h | head -2
python -m src.core.stress_detector >> /tmp/stress.log 2>&1 &
sleep 20
echo "=== Starting Tactic Inference ===" && free -h | head -2
python -m src.core.tactic_inference --interval 10 >> /tmp/tactics.log 2>&1 &
sleep 15
echo "=== All started ===" && free -h | head -2
ps aux | grep "src.core" | grep -v grep
