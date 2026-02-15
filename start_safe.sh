#!/bin/bash
# start_safe.sh - Lightweight analyzer pipeline
# Replaces stress_detector + tactic_inference with unified content_analyzer

set -e

echo "========================================="
echo "Anchor Pipeline - Safe Startup"
echo "========================================="

# ===== AGGRESSIVE CLEANUP =====
echo "[1/6] Killing existing processes..."

# Kill by module name
pkill -9 -f "src.core.audio_capture" 2>/dev/null || true
pkill -9 -f "src.core.speech_recognition" 2>/dev/null || true
pkill -9 -f "src.core.stress_detector" 2>/dev/null || true
pkill -9 -f "src.core.tactic_inference" 2>/dev/null || true
pkill -9 -f "src.core.content_analyzer" 2>/dev/null || true
pkill -9 -f "src.core.audio_intervention" 2>/dev/null || true
pkill -9 -f "src.viz.judges_window" 2>/dev/null || true
pkill -9 -f "src.core.system_monitor" 2>/dev/null || true

# Kill by port (multiple methods for reliability)
for port in 5555 5556 5557 5558 5559 8080; do
    fuser -k ${port}/tcp 2>/dev/null || true
    lsof -ti:${port} | xargs kill -9 2>/dev/null || true
done

# Clean up any zombie Python processes holding ZMQ sockets
pkill -9 -f "zmq" 2>/dev/null || true

sleep 3

# Verify cleanup
echo "Verifying ports are free..."
PORTS_BUSY=0
for port in 5555 5556 5557 5558 5559 8080; do
    if lsof -i:$port > /dev/null 2>&1; then
        echo "  ERROR: Port $port still in use!"
        lsof -i:$port
        PORTS_BUSY=1
    fi
done

if [ $PORTS_BUSY -eq 1 ]; then
    echo "Some ports still busy. Waiting 5 more seconds..."
    sleep 5
fi

# Clean log files
rm -f /tmp/audio.log /tmp/speech.log /tmp/analyzer.log /tmp/intervention.log /tmp/dashboard.log /tmp/system.log

# ===== ENVIRONMENT =====
echo "[2/6] Activating environment..."
cd ~/treehacks-anchor
source venv/bin/activate

# ===== CHECK DEPENDENCIES =====
echo "Checking new dependencies..."
python -c "import vaderSentiment; print('  vaderSentiment: OK')" 2>/dev/null || {
    echo "  Installing vaderSentiment..."
    pip install vaderSentiment --break-system-packages -q
}
python -c "import sentence_transformers; print('  sentence_transformers: OK')" 2>/dev/null || {
    echo "  Installing sentence_transformers..."
    pip install sentence-transformers --break-system-packages -q
}

# ===== START COMPONENTS =====
echo "[3/8] Starting audio capture..."
python -m src.core.audio_capture --device 0 --native-rate 44100 --seconds 3600 >> /tmp/audio.log 2>&1 &
AUDIO_PID=$!
sleep 3

# Verify audio started
if ! kill -0 $AUDIO_PID 2>/dev/null; then
    echo "  ERROR: Audio capture failed to start!"
    cat /tmp/audio.log
    exit 1
fi
echo "  Audio capture started (PID: $AUDIO_PID)"

echo "[4/8] Starting speech recognition (Whisper)..."
python -m src.core.speech_recognition --debug >> /tmp/speech.log 2>&1 &
SPEECH_PID=$!
sleep 12

if ! kill -0 $SPEECH_PID 2>/dev/null; then
    echo "  ERROR: Speech recognition failed to start!"
    tail -20 /tmp/speech.log
    exit 1
fi
echo "  Speech recognition started (PID: $SPEECH_PID)"

echo "[5/8] Starting content analyzer..."
python -m src.core.content_analyzer --debug >> /tmp/analyzer.log 2>&1 &
ANALYZER_PID=$!
sleep 8

if ! kill -0 $ANALYZER_PID 2>/dev/null; then
    echo "  ERROR: Content analyzer failed to start!"
    tail -20 /tmp/analyzer.log
    exit 1
fi
echo "  Content analyzer started (PID: $ANALYZER_PID)"

echo "[6/8] Starting audio intervention (Piper TTS)..."
# LLM warnings: ensure models/qwen-0.5b/ is populated via scripts/download_qwen_model.sh
# Use --no-llm to disable LLM and use static templates only
python -m src.core.audio_intervention --debug >> /tmp/intervention.log 2>&1 &
INTERVENTION_PID=$!
sleep 3

if ! kill -0 $INTERVENTION_PID 2>/dev/null; then
    echo "  WARNING: Audio intervention may have failed (check Piper model)"
    tail -10 /tmp/intervention.log
else
    echo "  Audio intervention started (PID: $INTERVENTION_PID)"
fi

echo "[7/8] Starting system monitor..."
python -m src.core.system_monitor >> /tmp/system.log 2>&1 &
SYSTEM_PID=$!
sleep 2

echo "[8/8] Starting dashboard..."
python -m src.viz.judges_window >> /tmp/dashboard.log 2>&1 &
DASHBOARD_PID=$!
sleep 3

if ! kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo "  ERROR: Dashboard failed to start!"
    tail -20 /tmp/dashboard.log
    exit 1
fi
echo "  Dashboard started (PID: $DASHBOARD_PID)"

# ===== SUMMARY =====
echo ""
echo "========================================="
echo "Pipeline Started Successfully!"
echo "========================================="
echo "Components:"
echo "  Audio Capture:      PID $AUDIO_PID"
echo "  Speech Recognition: PID $SPEECH_PID"
echo "  Content Analyzer:   PID $ANALYZER_PID"
echo "  Audio Intervention: PID $INTERVENTION_PID"
echo "  System Monitor:     PID $SYSTEM_PID"
echo "  Dashboard:          PID $DASHBOARD_PID"
echo ""
echo "Dashboard: http://localhost:8080"
echo ""
echo "Logs:"
echo "  tail -f /tmp/audio.log"
echo "  tail -f /tmp/speech.log"
echo "  tail -f /tmp/analyzer.log"
echo "  tail -f /tmp/intervention.log"
echo "  tail -f /tmp/system.log"
echo "  tail -f /tmp/dashboard.log"
echo ""
echo "To stop: pkill -f 'src.core\|src.viz'"
echo "========================================="

# Memory check
FREE_MEM=$(free -m | awk '/^Mem:/{print $7}')
echo "Available memory: ${FREE_MEM}MB"