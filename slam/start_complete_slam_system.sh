#!/bin/bash

echo "=========================================="
echo "🤖 Starting Complete SLAM System"
echo "=========================================="
echo ""
echo "This will start:"
echo "  1. RPLiDAR node"
echo "  2. Robot TF transforms (base_link -> laser)"
echo "  3. Cartographer SLAM"
echo "  4. Foxglove Bridge (for visualization)"
echo ""
echo "Connect Foxglove Studio to: ws://$(hostname -I | awk '{print $1}'):8765"
echo "Fixed Frame: 'map' or 'odom'"
echo "Topics: /scan, /map, /tf, /tf_static"
echo ""
echo "Press Ctrl+C to stop all"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source ROS2
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
unset RMW_IMPLEMENTATION
unset CYCLONEDDS_URI

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping all nodes..."
    pkill -f rplidar_composition
    pkill -f cartographer
    pkill -f foxglove_bridge
    pkill -f static_transform_publisher
    echo "All nodes stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Use the stable USB device ID so ttyUSB0/ttyUSB1 changes do not matter.
LIDAR_PORT="/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"

# Check LiDAR
if [ ! -e "$LIDAR_PORT" ]; then
    echo "❌ LiDAR not found at $LIDAR_PORT"
    echo "   Available serial ports:"
    ls -1 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "   (none)"
    exit 1
fi

echo "✅ LiDAR found at $LIDAR_PORT -> $(readlink -f "$LIDAR_PORT")"

# Kill any existing rplidar processes
echo "Checking for existing RPLiDAR processes..."
pkill -f rplidar_composition
sleep 1

# Start LiDAR in background
echo "▶ Starting RPLiDAR..."
echo "   (Motor should start spinning...)"
ros2 run rplidar_ros rplidar_composition \
  --ros-args -p serial_port:="$LIDAR_PORT" -p frame_id:=laser \
  -p serial_baudrate:=115200 &
LIDAR_PID=$!

# Wait for LiDAR motor to spin up
echo "   Waiting for LiDAR motor (5 seconds)..."
sleep 5

# Check if scan data is coming
echo "   Checking /scan topic..."
timeout 3 ros2 topic echo /scan --once > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ LiDAR is publishing data"
else
    echo "   ⚠️  No scan data yet (may need more time or check power)"
fi

# Start robot TF in background
echo "▶ Starting robot TF transforms..."
ros2 run tf2_ros static_transform_publisher 0 0 0.1 0 0 3.14159 base_link laser &
TF_PID=$!
sleep 1

# Start Cartographer in background
echo "▶ Starting Cartographer SLAM..."
ros2 launch "$SCRIPT_DIR/launch/cartographer_optimized.launch.py" &
CARTO_PID=$!
sleep 5

# Start Foxglove Bridge in foreground
echo "▶ Starting Foxglove Bridge..."
echo ""
ros2 launch foxglove_bridge foxglove_bridge_launch.xml

# Wait for Ctrl+C
wait
