#!/usr/bin/env python3
"""
Autonomous navigation to clicked point in Foxglove
Subscribe to /clicked_point and navigate robot to that position
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
import math

class GotoClickedPoint(Node):
    def __init__(self):
        super().__init__('goto_clicked_point')
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscribe to clicked point
        self.clicked_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_callback,
            10
        )
        
        # Publish goal to goal_pose_drive.py
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )
        
        self.get_logger().info('🎯 Goto Clicked Point node started')
        self.get_logger().info('📍 Click on the map in Foxglove to send robot there')
        
    def clicked_callback(self, msg):
        """When user clicks on map, send goal to navigation"""
        x = msg.point.x
        y = msg.point.y
        frame = msg.header.frame_id
        
        self.get_logger().info(f'\n📍 Clicked point received:')
        self.get_logger().info(f'   Position: ({x:.3f}, {y:.3f}) in frame "{frame}"')
        
        # Get current robot position
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
            
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            
            # Calculate distance and angle
            dx = x - robot_x
            dy = y - robot_y
            distance = math.sqrt(dx*dx + dy*dy)
            angle_to_goal = math.atan2(dy, dx)
            
            self.get_logger().info(f'🤖 Current robot position: ({robot_x:.3f}, {robot_y:.3f})')
            self.get_logger().info(f'📏 Distance to goal: {distance:.3f} m')
            self.get_logger().info(f'🧭 Angle to goal: {math.degrees(angle_to_goal):.1f}°')
            
        except TransformException as e:
            self.get_logger().warning(f'Could not get robot position: {e}')
            robot_x = 0.0
            robot_y = 0.0
            angle_to_goal = 0.0
        
        # Create goal message
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        
        # Position
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        
        # Orientation (face toward goal)
        qz = math.sin(angle_to_goal * 0.5)
        qw = math.cos(angle_to_goal * 0.5)
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        
        # Publish goal
        self.goal_pub.publish(goal)
        
        self.get_logger().info(f'✅ Goal sent to navigation!')
        self.get_logger().info(f'🚗 Robot will drive to ({x:.3f}, {y:.3f})')
        self.get_logger().info('')

def main():
    rclpy.init()
    node = GotoClickedPoint()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n👋 Shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
