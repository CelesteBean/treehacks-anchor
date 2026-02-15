#!/bin/bash
# stop_pipeline.sh - Kill all pipeline processes

echo "Stopping Anchor pipeline..."

pkill -9 -f "src.core.audio_capture" 2>/dev/null
pkill -9 -f "src.core.speech_recognition" 2>/dev/null
pkill -9 -f "src.core.stress_detector" 2>/dev/null
pkill -9 -f "src.core.tactic_inference" 2>/dev/null
pkill -9 -f "src.core.content_analyzer" 2>/dev/null
pkill -9 -f "src.viz.judges_window" 2>/dev/null

for port in 5555 5556 5557 5558 8080; do
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
done

sleep 2
echo "Done. All components stopped."