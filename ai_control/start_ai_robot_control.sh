#!/bin/bash

# ============================================================
# START AI ROBOT CONTROL WITH WEB INTERFACE
# ============================================================

set -e

echo ""
echo "=========================================="
echo "  🤖 AI ROBOT CONTROL SYSTEM"
echo "=========================================="
echo ""

# Activate Edge Impulse virtual environment
if [ -d "$HOME/ei-venv" ]; then
    echo "🔍 Activating Edge Impulse virtual environment..."
    source "$HOME/ei-venv/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found at ~/ei-venv"
    echo "   Using system Python..."
fi

# Kiểm tra camera (relaxed check)
echo "🔍 Checking camera..."
if libcamera-hello --list-cameras &>/dev/null; then
    echo "✅ Camera OK"
elif [ -e "/dev/video0" ]; then
    echo "⚠️  Camera device exists (/dev/video0)"
    echo "   Will try to use it anyway..."
else
    echo "⚠️  Camera check failed, but will try anyway"
fi

echo "🔍 Checking Arduino USB serial port /dev/ttyACM0..."
UART_PORT="/dev/ttyACM0"

if [[ ! -e "$UART_PORT" ]]; then
    echo "❌ Arduino USB serial port $UART_PORT not found!"
    exit 1
fi

if [[ ! -r "$UART_PORT" || ! -w "$UART_PORT" ]]; then
    echo "❌ No read/write permission for $UART_PORT. Add your user to the dialout group, then log in again."
    exit 1
fi
export UART_PORT
echo "✅ Arduino USB serial port: $UART_PORT"

# Kiểm tra model
MODEL="/home/quan/edge-impulse-linux-runner/bien_bao_pi-linux-aarch64-v5-impulse-#3.eim"
echo "🔍 Checking AI model..."
if [ -f "$MODEL" ]; then
    echo "✅ Model found"
else
    echo "⚠️  Model not found: $MODEL"
    echo "   Update MODEL_PATH in realtime_detect_with_control.py"
fi

echo ""
echo "=========================================="
echo "  🎛️  SERVO CONFIGURATION"
echo "=========================================="
echo ""
echo "Current defaults:"
echo "  - LEFT:   1500 µs"
echo "  - CENTER: 2300 µs"
echo "  - RIGHT:  2600 µs"
echo ""

# Prompt for servo values with defaults
read -p "Enter SERVO_LEFT pulse (1500-2600) [1500]: " SERVO_LEFT
SERVO_LEFT=${SERVO_LEFT:-1500}

read -p "Enter SERVO_CENTER pulse (1500-2600) [2300]: " SERVO_CENTER
SERVO_CENTER=${SERVO_CENTER:-2300}

read -p "Enter SERVO_RIGHT pulse (1500-2600) [2600]: " SERVO_RIGHT
SERVO_RIGHT=${SERVO_RIGHT:-2600}

# Validate ranges
if [ "$SERVO_LEFT" -lt 1500 ] || [ "$SERVO_LEFT" -gt 2600 ]; then
    echo "❌ SERVO_LEFT out of range (1500-2600)"
    exit 1
fi

if [ "$SERVO_CENTER" -lt 1500 ] || [ "$SERVO_CENTER" -gt 2600 ]; then
    echo "❌ SERVO_CENTER out of range (1500-2600)"
    exit 1
fi

if [ "$SERVO_RIGHT" -lt 1500 ] || [ "$SERVO_RIGHT" -gt 2600 ]; then
    echo "❌ SERVO_RIGHT out of range (1500-2600)"
    exit 1
fi

echo ""
echo "✅ Servo configuration:"
echo "   LEFT:   $SERVO_LEFT µs"
echo "   CENTER: $SERVO_CENTER µs"
echo "   RIGHT:  $SERVO_RIGHT µs"

# Get IP address
IP=$(hostname -I | awk '{print $1}')

echo ""
echo "=========================================="
echo "  Starting AI Robot Control..."
echo "=========================================="
echo ""
echo "📡 Access Web UI at:"
echo "   http://$IP:8080"
echo ""
echo "🎮 Controls:"
echo "   - Manual Mode: Use on-screen buttons"
echo "   - AI Auto Mode: Robot follows AI detection"
echo ""
echo "⚠️  Press Ctrl+C to stop"
echo ""
echo "=========================================="
echo ""

cd "$(dirname "$(readlink -f "$0")")"
exec python3 realtime_detect_with_control.py \
    --servo-left "$SERVO_LEFT" \
    --servo-center "$SERVO_CENTER" \
    --servo-right "$SERVO_RIGHT"
