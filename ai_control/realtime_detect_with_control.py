#!/usr/bin/env python3
"""
AI Detection + Web Control + Camera Stream
Tích hợp: Edge Impulse AI, GStreamer camera, UART robot control, Flask web UI
"""

from flask import Flask, Response, jsonify, request
from edge_impulse_linux.image import ImageImpulseRunner
import serial
import gi
import numpy as np
import threading
import time
import psutil
import os

import argparse
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/home/quan/edge-impulse-linux-runner/bien_bao_pi-linux-aarch64-v5-impulse-#3.eim"

# UART Configuration
UART_PORT = os.environ.get("UART_PORT", "/dev/ttyACM0")
BAUDRATE = 115200

# Camera & Stream
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
STREAM_FPS_TARGET = 20
AI_FPS_TARGET = 10
JPEG_QUALITY = 75
HTTP_PORT = 8080

# Servo settings (microseconds)
SERVO_MIN = 1500
SERVO_CENTER = 2300
SERVO_MAX = 2600
SERVO_LEFT = 1700   # Slight left adjustment
SERVO_RIGHT = 2500  # Slight right adjustment
# Motor PWM (0-50 according to Mega firmware)
MOTOR_PWM_MAX = 50
MOTOR_PWM_DEFAULT = 50

# AI Auto Mode settings
AI_CONFIDENCE_THRESHOLD = 0.6
AI_ACTION_COOLDOWN = 2.0

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# UART SETUP
# ============================================================

print("\n==============================================")
print(" Connecting to UART")
print("==============================================")

try:
    ser = serial.Serial(UART_PORT, BAUDRATE, timeout=0.1)
    time.sleep(1)
    print(f"✓ UART: {UART_PORT} @ {BAUDRATE}")
except Exception as e:
    print(f"ERROR: Cannot open UART: {e}")
    exit(1)

uart_lock = threading.Lock()
encoder_lock = threading.Lock()

# Control state
control_mode = "MANUAL"  # MANUAL or AUTO
current_servo_pulse = SERVO_CENTER
current_motor_pwm = 0
encoder_data = {"left": 0, "right": 0, "servo": SERVO_CENTER}
last_encoder_time = 0

def send_uart(command):
    """Send command via UART thread-safe"""
    try:
        with uart_lock:
            message = command + "\n"
            ser.write(message.encode())
            ser.flush()
        print(f"UART → {command}")
    except Exception as e:
        print(f"UART error: {e}")

def set_servo(pulse):
    """Set servo pulse (700-1500µs)"""
    global current_servo_pulse
    pulse = max(SERVO_MIN, min(SERVO_MAX, pulse))
    current_servo_pulse = pulse
    send_uart(f"P:{pulse}")

def drive(servo_pulse, pwm):
    """Drive with servo and motor PWM"""
    global current_servo_pulse, current_motor_pwm
    servo_pulse = max(SERVO_MIN, min(SERVO_MAX, servo_pulse))
    pwm = max(0, min(MOTOR_PWM_MAX, pwm))
    current_servo_pulse = servo_pulse
    current_motor_pwm = pwm
    send_uart(f"V,{servo_pulse},{pwm}")

def motor_stop():
    """Stop motor"""
    global current_motor_pwm
    current_motor_pwm = 0
    send_uart("S")

# ============================================================
# ENCODER READER THREAD
# ============================================================

# Encoder reading DISABLED - reserved for SLAM odometry (goal_pose_drive.py)
# def encoder_reader_thread(): [REMOVED]

# LOAD AI MODEL
# ============================================================

print("\n==============================================")
print(" Loading AI Model")
print("==============================================")

try:
    runner = ImageImpulseRunner(MODEL_PATH)
    model_info = runner.init()
    
    model_params = model_info.get("model_parameters", {})
    MODEL_WIDTH = int(model_params.get("image_input_width", 96))
    MODEL_HEIGHT = int(model_params.get("image_input_height", 96))
    labels = model_params.get("labels", [])
    
    print(f"Model: {MODEL_PATH.split('/')[-1]}")
    print(f"Input: {MODEL_WIDTH}x{MODEL_HEIGHT}")
    print(f"Labels: {labels}")
    
except Exception as e:
    print(f"ERROR: {e}")
    motor_stop()
    ser.close()
    exit(1)

# ============================================================
# CALCULATE CROP
# ============================================================

stream_aspect = STREAM_WIDTH / STREAM_HEIGHT
model_aspect = MODEL_WIDTH / MODEL_HEIGHT

if stream_aspect > model_aspect:
    crop_height = STREAM_HEIGHT
    crop_width = int(crop_height * model_aspect)
    crop_left = (STREAM_WIDTH - crop_width) // 2
    crop_right = STREAM_WIDTH - crop_width - crop_left
    crop_top = 0
    crop_bottom = 0
else:
    crop_width = STREAM_WIDTH
    crop_height = int(crop_width / model_aspect)
    crop_top = (STREAM_HEIGHT - crop_height) // 2
    crop_bottom = STREAM_HEIGHT - crop_height - crop_top
    crop_left = 0
    crop_right = 0

print(f"Stream: {STREAM_WIDTH}x{STREAM_HEIGHT} @ {STREAM_FPS_TARGET} FPS")
print(f"Crop: L={crop_left}, R={crop_right}, T={crop_top}, B={crop_bottom}")

# ============================================================
# GSTREAMER PIPELINE
# ============================================================

Gst.init(None)

pipeline_string = f"""
libcamerasrc !
video/x-raw,width={STREAM_WIDTH},height={STREAM_HEIGHT},framerate={STREAM_FPS_TARGET}/1 !
tee name=t

t. ! queue max-size-buffers=1 leaky=downstream !
videoconvert ! video/x-raw,format=I420 !
jpegenc quality={JPEG_QUALITY} !
appsink name=streamsink emit-signals=true max-buffers=1 drop=true sync=false

t. ! queue max-size-buffers=1 leaky=downstream !
videocrop left={crop_left} right={crop_right} top={crop_top} bottom={crop_bottom} !
videoscale ! videoconvert !
video/x-raw,width={MODEL_WIDTH},height={MODEL_HEIGHT},format=RGB !
appsink name=aisink emit-signals=true max-buffers=1 drop=true sync=false
"""

pipeline = Gst.parse_launch(pipeline_string)
stream_sink = pipeline.get_by_name("streamsink")
ai_sink = pipeline.get_by_name("aisink")

# ============================================================
# SHARED DATA
# ============================================================

latest_jpeg = None
latest_stream_id = 0
stream_condition = threading.Condition()

latest_ai_frame = None
latest_ai_id = 0
ai_condition = threading.Condition()

metrics_lock = threading.Lock()
stream_fps = 0.0
ai_fps = 0.0
inference_ms = 0.0
detections = []
top_label = ""
top_confidence = 0.0

stream_counter = 0
stream_counter_start = time.monotonic()
ai_counter = 0
ai_counter_start = time.monotonic()

# AI Auto mode state
last_ai_action_time = 0

# ============================================================
# CALLBACKS
# ============================================================

def on_stream_sample(sink):
    global latest_jpeg, latest_stream_id
    global stream_fps, stream_counter, stream_counter_start

    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR

    buffer = sample.get_buffer()
    jpeg = buffer.extract_dup(0, buffer.get_size())

    with stream_condition:
        latest_jpeg = jpeg
        latest_stream_id += 1
        stream_condition.notify_all()

    stream_counter += 1
    now = time.monotonic()
    elapsed = now - stream_counter_start

    if elapsed >= 1.0:
        with metrics_lock:
            stream_fps = stream_counter / elapsed
        stream_counter = 0
        stream_counter_start = now

    return Gst.FlowReturn.OK

def on_ai_sample(sink):
    global latest_ai_frame, latest_ai_id

    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR

    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)

    if not success:
        return Gst.FlowReturn.ERROR

    try:
        frame = np.frombuffer(map_info.data, dtype=np.uint8)
        frame = frame.reshape((MODEL_HEIGHT, MODEL_WIDTH, 3)).copy()
    finally:
        buffer.unmap(map_info)

    with ai_condition:
        latest_ai_frame = frame
        latest_ai_id += 1
        ai_condition.notify()

    return Gst.FlowReturn.OK

stream_sink.connect("new-sample", on_stream_sample)
ai_sink.connect("new-sample", on_ai_sample)

# ============================================================
# BOX MAPPING
# ============================================================

def map_box_to_stream(bb):
    x = float(bb.get("x", 0))
    y = float(bb.get("y", 0))
    w = float(bb.get("width", 0))
    h = float(bb.get("height", 0))

    scale_x = crop_width / MODEL_WIDTH
    scale_y = crop_height / MODEL_HEIGHT

    stream_x = crop_left + x * scale_x
    stream_y = crop_top + y * scale_y
    stream_w = w * scale_x
    stream_h = h * scale_y

    return {
        "label": bb.get("label", ""),
        "value": float(bb.get("value", 0)),
        "x": stream_x,
        "y": stream_y,
        "width": stream_w,
        "height": stream_h,
        "left_pct": stream_x / STREAM_WIDTH * 100.0,
        "top_pct": stream_y / STREAM_HEIGHT * 100.0,
        "width_pct": stream_w / STREAM_WIDTH * 100.0,
        "height_pct": stream_h / STREAM_HEIGHT * 100.0
    }

def is_in_detection_zone(box):
    """Check if box is in upper-right quadrant"""
    center_x = box["x"] + box["width"] / 2
    center_y = box["y"] + box["height"] / 2
    return (center_x > STREAM_WIDTH / 2) and (center_y < STREAM_HEIGHT / 2)

# ============================================================
# AI AUTO MODE ACTION
# ============================================================

def execute_ai_action(sign, confidence):
    """Execute robot action based on AI detection"""
    global last_ai_action_time, control_mode
    
    if control_mode != "AUTO":
        return
    
    now = time.monotonic()
    
    # Cooldown check
    if now - last_ai_action_time < AI_ACTION_COOLDOWN:
        return
    
    normalized_sign = sign.strip().lower().replace("-", "_").replace(" ", "_")
    
    if normalized_sign == "stop":
        motor_stop()
        last_ai_action_time = now
        print(f"\n🚦 AI ACTION: STOP ({confidence*100:.1f}%)")
        
    elif normalized_sign == "turn_left":
        drive(SERVO_LEFT, MOTOR_PWM_DEFAULT)
        last_ai_action_time = now
        print(f"\n🚦 AI ACTION: TURN LEFT ({confidence*100:.1f}%)")
        
    elif normalized_sign == "turn_right":
        drive(SERVO_RIGHT, MOTOR_PWM_DEFAULT)
        last_ai_action_time = now
        print(f"\n🚦 AI ACTION: TURN RIGHT ({confidence*100:.1f}%)")
        
    elif normalized_sign == "straight":
        drive(SERVO_CENTER, MOTOR_PWM_DEFAULT)
        last_ai_action_time = now
        print(f"\n🚦 AI ACTION: GO STRAIGHT ({confidence*100:.1f}%)")
        
    elif normalized_sign == "slow":
        drive(current_servo_pulse, 15)
        last_ai_action_time = now
        print(f"\n🚦 AI ACTION: SLOW DOWN ({confidence*100:.1f}%)")

# ============================================================
# AI WORKER
# ============================================================

def ai_worker():
    global ai_fps, inference_ms, detections, top_label, top_confidence
    global ai_counter, ai_counter_start

    last_frame_id = -1
    min_interval = 1.0 / AI_FPS_TARGET
    last_inference_time = 0.0

    while True:
        with ai_condition:
            ai_condition.wait_for(
                lambda: latest_ai_id != last_frame_id,
                timeout=1.0
            )
            frame_id = latest_ai_id
            frame = latest_ai_frame

        if frame is None:
            continue

        now = time.monotonic()
        if now - last_inference_time < min_interval:
            continue

        last_frame_id = frame_id
        last_inference_time = now

        try:
            start = time.perf_counter()
            features, _ = runner.get_features_from_image(frame)
            result = runner.classify(features)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            result_data = result.get("result", {})
            boxes = result_data.get("bounding_boxes", [])
            mapped_boxes = []
            best_label = ""
            best_value = 0.0

            for bb in boxes:
                value = float(bb.get("value", 0))
                mapped = map_box_to_stream(bb)
                
                if is_in_detection_zone(mapped):
                    mapped_boxes.append(mapped)
                    if value > best_value:
                        best_value = value
                        best_label = bb.get("label", "")

            classification = result_data.get("classification", {})
            if not boxes and classification:
                for label, value in classification.items():
                    value = float(value)
                    if value > best_value:
                        best_value = value
                        best_label = label

            with metrics_lock:
                detections = mapped_boxes
                inference_ms = elapsed_ms
                top_label = best_label
                top_confidence = best_value

            # Execute AI action if confidence is high enough
            if best_label and best_value >= AI_CONFIDENCE_THRESHOLD:
                execute_ai_action(best_label, best_value)

            ai_counter += 1
            now2 = time.monotonic()
            ai_elapsed = now2 - ai_counter_start

            if ai_elapsed >= 1.0:
                with metrics_lock:
                    ai_fps = ai_counter / ai_elapsed
                ai_counter = 0
                ai_counter_start = now2

        except Exception as e:
            print(f"AI error: {e}")
            time.sleep(0.1)

threading.Thread(target=ai_worker, daemon=True).start()

# ============================================================
# START CAMERA
# ============================================================

state = pipeline.set_state(Gst.State.PLAYING)
if state == Gst.StateChangeReturn.FAILURE:
    print("ERROR: Cannot start camera")
    runner.stop()
    motor_stop()
    ser.close()
    exit(1)

print("✓ Camera started\n")

# ============================================================
# SYSTEM INFO
# ============================================================

process = psutil.Process(os.getpid())
psutil.cpu_percent(interval=None)
process.cpu_percent(interval=None)

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except:
        return 0.0

# ============================================================
# VIDEO STREAM
# ============================================================

def generate_video():
    last_id = -1
    while True:
        with stream_condition:
            stream_condition.wait_for(
                lambda: latest_stream_id != last_id,
                timeout=1.0
            )
            current_id = latest_stream_id
            jpeg = latest_jpeg

        if jpeg is None:
            continue

        last_id = current_id
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Cache-Control: no-cache\r\n"
               b"\r\n" + jpeg + b"\r\n")

# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def index():
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>AI Robot Control</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #0a0e27;
            color: #fff;
            font-family: 'Segoe UI', Arial, sans-serif;
            padding: 10px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin: 15px 0;
            color: #4fc3f7;
            font-size: 24px;
        }}
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;  /* 50/50 split: stream left, controls right */
            min-width: 800px;  /* Prevent extreme squashing */
            gap: 20px;
            margin-bottom: 20px;
        }}
        .video-container {{
            position: relative;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        }}
        #video {{
            width: 100%;
            display: block;
        }}
        #boxes {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        .quadrant-guide {{
            position: absolute;
            top: 0;
            right: 0;
            width: 50%;
            height: 50%;
            border: 2px dashed rgba(79, 195, 247, 0.5);
            box-sizing: border-box;
        }}
        .quadrant-label {{
            position: absolute;
            top: 5px;
            right: 5px;
            background: rgba(79, 195, 247, 0.8);
            color: #000;
            padding: 3px 8px;
            font-size: 11px;
            font-weight: bold;
            border-radius: 4px;
        }}
        .box {{
            position: absolute;
            border: 3px solid #ff1744;
            box-shadow: 0 0 10px rgba(255,23,68,0.5);
        }}
        .box-label {{
            position: absolute;
            left: -3px;
            top: -28px;
            background: #ff1744;
            color: #fff;
            padding: 4px 10px;
            font-size: 13px;
            font-weight: bold;
            border-radius: 4px;
            white-space: nowrap;
        }}
        .control-panel {{
            width: 100%;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
        }}
        .mode-toggle {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .toggle-btn {{
            background: #374151;
            border: 2px solid #4fc3f7;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin: 0 5px;
        }}
        .toggle-btn.active {{
            background: #4fc3f7;
            color: #000;
        }}
        .controller {{
            display: grid;
            grid-template-rows: auto auto auto;
            gap: 12px;
            margin: 20px 0;
        }}
        .row {{
            display: flex;
            justify-content: center;
            gap: 12px;
        }}
        button {{
            width: 90px;
            height: 80px;
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 24px;
            font-weight: bold;
            background: #374151;
            cursor: pointer;
            user-select: none;
            touch-action: none;
        }}
        button:active {{
            transform: scale(0.95);
        }}
        .btn-forward {{ background: #059669; }}
        .btn-backward {{ background: #7c3aed; }}
        .btn-left {{ background: #2563eb; }}
        .btn-right {{ background: #2563eb; }}
        .btn-stop {{ background: #dc2626; width: 100%; }}
        .status-box {{
            background: rgba(0,0,0,0.3);
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 13px;
        }}
        .status-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
        }}
        .status-value {{
            color: #4fc3f7;
            font-weight: bold;
        }}
        .detection-info {{
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            padding: 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            margin: 15px 0;
        }}
        #detection {{
            color: #4fc3f7;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }}
        .metric {{
            background: rgba(255,255,255,0.05);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .metric-label {{
            color: #888;
            font-size: 10px;
            margin-bottom: 3px;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: bold;
            color: #4fc3f7;
        }}
        .metric-unit {{
            font-size: 11px;
            color: #888;
            margin-left: 2px;
        }}
        .emergency-stop {{
            background: #dc2626;
            color: white;
            border: 3px solid #fff;
            padding: 15px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
            margin-top: 15px;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7); }}
            50% {{ box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Robot Control + Camera Stream</h1>
        
        <div class="main-grid">
            <div class="video-container">
                <img id="video" src="/video">
                <div id="boxes"></div>
                <div class="quadrant-guide">
                    <div class="quadrant-label">Detection Zone</div>
                </div>
            </div>
            
            <div class="control-panel">
                <div class="mode-toggle">
                    <button class="toggle-btn active" id="manual-btn" onclick="setMode('MANUAL')">
                        🎮 MANUAL
                    </button>
                    <button class="toggle-btn" id="auto-btn" onclick="setMode('AUTO')">
                        🤖 AI AUTO
                    </button>
                </div>
                
                <div class="servo-control" id="manual-controls">
                    <div style="margin-bottom: 15px; text-align: center;">
                        <label style="display: block; margin-bottom: 8px; font-size: 14px; color: #4fc3f7;">
                            Servo Pulse (µs)
                        </label>
                        <input type="number" id="servo-input" 
                               min="1500" max="2600" step="50" value="2300"
                               style="width: 150px; padding: 10px; font-size: 16px; text-align: center; 
                                      border: 2px solid #4fc3f7; border-radius: 8px; 
                                      background: #1a1f3a; color: white;">
                        <button onclick="setServoPulse()" 
                                style="margin-left: 10px; padding: 10px 20px; font-size: 16px;
                                       background: #4fc3f7; color: #000; border: none; 
                                       border-radius: 8px; cursor: pointer; font-weight: bold;">
                            SET
                        </button>
                    </div>
                    <div class="controller">
                        <div class="row">
                            <button class="btn-forward" id="forward">▲</button>
                        </div>
                        <div class="row">
                            <button class="btn-backward" id="backward">▼</button>
                        </div>
                        <div class="row">
                            <button class="btn-stop" onclick="stopMotor()">■ STOP</button>
                        </div>
                    </div>
                </div>
                
                <div class="status-box">
                    <div class="status-row">
                        <span>Mode:</span>
                        <span class="status-value" id="mode">MANUAL</span>
                    </div>
                    <div class="status-row">
                        <span>Servo:</span>
                        <span class="status-value" id="servo">1100 µs</span>
                    </div>
                    <div class="status-row">
                        <span>Motor PWM:</span>
                        <span class="status-value" id="motor">0</span>
                    </div>
                    <div class="status-row">
                        <span>Encoder L/R:</span>
                        <span class="status-value" id="encoder">-- / --</span>
                    </div>
                </div>
                
                <button class="emergency-stop" onclick="emergencyStop()">
                    ⚠️ EMERGENCY STOP
                </button>
            </div>
        </div>
        
        <div class="detection-info">
            Detection: <span id="detection">Waiting...</span>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Stream FPS</div>
                <div class="metric-value">
                    <span id="streamfps">--</span><span class="metric-unit">fps</span>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">AI FPS</div>
                <div class="metric-value">
                    <span id="aifps">--</span><span class="metric-unit">fps</span>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Inference</div>
                <div class="metric-value">
                    <span id="infer">--</span><span class="metric-unit">ms</span>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">CPU Temp</div>
                <div class="metric-value">
                    <span id="temp">--</span><span class="metric-unit">°C</span>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">CPU Usage</div>
                <div class="metric-value">
                    <span id="cpu">--</span><span class="metric-unit">%</span>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Connection</div>
                <div class="metric-value">
                    <span id="connection" style="color: #22c55e;">●</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentMode = 'MANUAL';
        let motorTimer = null;
        
        async function setMode(mode) {{
            try {{
                const r = await fetch('/control/mode', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{mode: mode}})
                }});
                if (r.ok) {{
                    currentMode = mode;
                    document.getElementById('manual-btn').classList.toggle('active', mode === 'MANUAL');
                    document.getElementById('auto-btn').classList.toggle('active', mode === 'AUTO');
                    document.getElementById('manual-controls').style.opacity = mode === 'MANUAL' ? '1' : '0.3';
                    document.getElementById('manual-controls').style.pointerEvents = mode === 'MANUAL' ? 'auto' : 'none';
                }}
            }} catch(e) {{ console.error(e); }}
        }}
        
        async function setServoPulse() {{
            if (currentMode !== 'MANUAL') return;
            const pulse = parseInt(document.getElementById('servo-input').value);
            if (pulse < 1500 || pulse > 2600) {{
                alert('Servo pulse must be between 1500-2600 µs');
                return;
            }}
            try {{
                const r = await fetch('/control/servo/set', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{pulse: pulse}})
                }});
                if (r.ok) {{
                    const data = await r.json();
                    console.log('Servo set to:', data.pulse);
                }}
            }} catch(e) {{
                console.error(e);
            }}
        }}
        
        async function motorCommand(cmd) {{
            if (currentMode !== 'MANUAL') return;
            await fetch('/control/motor/' + cmd, {{method: 'POST'}});
        }}
        
        async function stopMotor() {{
            await fetch('/control/stop', {{method: 'POST'}});
        }}
        
        async function emergencyStop() {{
            await fetch('/control/emergency', {{method: 'POST'}});
            setMode('MANUAL');
        }}
        
        function setupHoldButton(id, cmd) {{
            const btn = document.getElementById(id);
            let timer = null;
            
            function start(e) {{
                e.preventDefault();
                if (currentMode !== 'MANUAL') return;
                motorCommand(cmd);
                timer = setInterval(() => motorCommand(cmd), 150);
            }}
            
            function stop(e) {{
                if (e) e.preventDefault();
                if (timer) {{
                    clearInterval(timer);
                    timer = null;
                }}
                motorCommand('stop');
            }}
            
            btn.addEventListener('pointerdown', start);
            btn.addEventListener('pointerup', stop);
            btn.addEventListener('pointerleave', stop);
            btn.addEventListener('pointercancel', stop);
        }}
        
        setupHoldButton('forward', 'forward');
        setupHoldButton('backward', 'backward');
        
        async function updateDetection() {{
            try {{
                const r = await fetch("/detections", {{cache:"no-store"}});
                const data = await r.json();
                
                const layer = document.getElementById("boxes");
                layer.innerHTML = "";
                
                data.boxes.forEach(b => {{
                    const box = document.createElement("div");
                    box.className = "box";
                    box.style.left = b.left_pct + "%";
                    box.style.top = b.top_pct + "%";
                    box.style.width = b.width_pct + "%";
                    box.style.height = b.height_pct + "%";
                    
                    const label = document.createElement("div");
                    label.className = "box-label";
                    label.innerText = b.label + " " + (b.value * 100).toFixed(0) + "%";
                    box.appendChild(label);
                    layer.appendChild(box);
                }});
                
                const detection = document.getElementById("detection");
                if (data.top_label) {{
                    const conf = (data.top_confidence * 100).toFixed(1);
                    detection.innerText = data.top_label + " (" + conf + "%)";
                    detection.style.color = conf > 60 ? "#4fc3f7" : "#ffa726";
                }} else {{
                    detection.innerText = "No detection";
                    detection.style.color = "#888";
                }}
                
                document.getElementById('connection').style.color = '#22c55e';
            }} catch(e) {{
                document.getElementById('connection').style.color = '#ef4444';
            }}
        }}
        
        async function updateStatus() {{
            try {{
                const r = await fetch("/status", {{cache:"no-store"}});
                const d = await r.json();
                document.getElementById("mode").innerText = d.mode;
                document.getElementById("servo").innerText = d.servo + " µs";
                document.getElementById("motor").innerText = d.motor_pwm;
                // Show SLAM reservation message for encoder
                if (d.encoder.message) {{
                    document.getElementById("encoder").innerText = d.encoder.message;
                }} else {{
                    document.getElementById("encoder").innerText = d.encoder.left + " / " + d.encoder.right;
                }}
            }} catch(e) {{}}
        }}
        
        async function updateMetrics() {{
            try {{
                const r = await fetch("/metrics", {{cache:"no-store"}});
                const d = await r.json();
                document.getElementById("streamfps").innerText = d.stream_fps;
                document.getElementById("aifps").innerText = d.ai_fps;
                document.getElementById("infer").innerText = d.inference_ms;
                document.getElementById("temp").innerText = d.temperature;
                document.getElementById("cpu").innerText = d.cpu;
            }} catch(e) {{}}
        }}
        
        setInterval(updateDetection, 100);
        setInterval(updateStatus, 200);
        setInterval(updateMetrics, 1000);
        updateDetection();
        updateStatus();
        updateMetrics();
        
        window.addEventListener('blur', () => {{ if (currentMode === 'MANUAL') stopMotor(); }});
        document.addEventListener('visibilitychange', () => {{ if (document.hidden && currentMode === 'MANUAL') stopMotor(); }});
    </script>
</body>
</html>
"""

@app.route("/video")
def video():
    return Response(
        generate_video(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store"}
    )

@app.route("/detections")
def detection_api():
    with metrics_lock:
        return jsonify({
            "boxes": detections,
            "top_label": top_label,
            "top_confidence": round(top_confidence, 4)
        })

@app.route("/status")
def status_api():
    with encoder_lock:
        enc = encoder_data.copy()
    return jsonify({
        "mode": control_mode,
        "servo": current_servo_pulse,
        "motor_pwm": current_motor_pwm,
        "encoder": {
            "left": "N/A",
            "right": "N/A",
            "servo": enc["servo"],
            "message": "Reserved for SLAM"
        }
    })

@app.route("/metrics")
def metrics_api():
    temp = get_cpu_temp()
    cpu = psutil.cpu_percent(interval=None)

    with metrics_lock:
        return jsonify({
            "stream_fps": round(stream_fps, 1),
            "ai_fps": round(ai_fps, 1),
            "inference_ms": round(inference_ms, 1),
            "temperature": round(temp, 1),
            "cpu": round(cpu, 1)
        })

@app.route("/control/mode", methods=["POST"])
def set_mode():
    global control_mode
    data = request.get_json()
    mode = data.get("mode", "MANUAL").upper()
    if mode in ["MANUAL", "AUTO"]:
        control_mode = mode
        if mode == "MANUAL":
            motor_stop()
        return jsonify({"ok": True, "mode": mode})
    return jsonify({"ok": False}), 400

@app.route("/control/servo/left", methods=["POST"])
def servo_left():
    if control_mode == "MANUAL":
        set_servo(current_servo_pulse - 50)
    return jsonify({"ok": True})

@app.route("/control/servo/right", methods=["POST"])
def servo_right():
    if control_mode == "MANUAL":
        set_servo(current_servo_pulse + 50)
    return jsonify({"ok": True})

@app.route("/control/servo/set", methods=["POST"])
def servo_set():
    if control_mode == "MANUAL":
        data = request.get_json()
        pulse = data.get("pulse", SERVO_CENTER)
        pulse = max(SERVO_MIN, min(SERVO_MAX, int(pulse)))
        set_servo(pulse)
        return jsonify({"ok": True, "pulse": pulse})
    return jsonify({"ok": False}), 400

@app.route("/control/motor/<cmd>", methods=["POST"])
def motor_control(cmd):
    if control_mode == "MANUAL":
        if cmd == "forward":
            drive(current_servo_pulse, MOTOR_PWM_DEFAULT)
        elif cmd == "backward":
            # Send backward command: B,servo,pwm
            send_uart(f"B,{current_servo_pulse},{MOTOR_PWM_DEFAULT}")
            global current_motor_pwm
            current_motor_pwm = MOTOR_PWM_DEFAULT
        elif cmd == "stop":
            motor_stop()
    return jsonify({"ok": True})

@app.route("/control/stop", methods=["POST"])
def stop():
    motor_stop()
    return jsonify({"ok": True})

@app.route("/control/emergency", methods=["POST"])
def emergency():
    global control_mode
    control_mode = "MANUAL"
    motor_stop()
    return jsonify({"ok": True})

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Robot Control System')
    parser.add_argument('--servo-left', type=int, default=1700,
                        help='Servo LEFT pulse in microseconds (1500-2600)')
    parser.add_argument('--servo-center', type=int, default=2300,
                        help='Servo CENTER pulse in microseconds (1500-2600)')
    parser.add_argument('--servo-right', type=int, default=2500,
                        help='Servo RIGHT pulse in microseconds (1500-2600)')
    
    args = parser.parse_args()
    
    # Validate ranges
    if not (1500 <= args.servo_left <= 2600):
        parser.error("--servo-left must be between 1500 and 2600")
    if not (1500 <= args.servo_center <= 2600):
        parser.error("--servo-center must be between 1500 and 2600")
    if not (1500 <= args.servo_right <= 2600):
        parser.error("--servo-right must be between 1500 and 2600")
    
    # Update servo constants by modifying globals()
    import sys
    current_module = sys.modules[__name__]
    current_module.SERVO_LEFT = args.servo_left
    current_module.SERVO_CENTER = args.servo_center
    current_module.SERVO_RIGHT = args.servo_right
    
    print("==============================================")
    print(" AI Robot Control System")
    print("==============================================")
    print(f"Web UI: http://<PI-IP>:{HTTP_PORT}")
    print(f"AI Confidence: {AI_CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"AI Cooldown: {AI_ACTION_COOLDOWN}s")
    print(f"Servo Config: LEFT={args.servo_left} CENTER={args.servo_center} RIGHT={args.servo_right}")
    print("==============================================\n")

    # Initialize robot state
    motor_stop()
    set_servo(SERVO_CENTER)

    try:
        app.run(
            host="0.0.0.0",
            port=HTTP_PORT,
            threaded=True,
            use_reloader=False
        )
    finally:
        motor_stop()
        pipeline.set_state(Gst.State.NULL)
        runner.stop()
        ser.close()
        print("\n✓ System stopped safely")
