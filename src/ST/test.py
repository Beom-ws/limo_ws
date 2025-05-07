import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

class VideoLaneDetection(Node):
    def __init__(self):
        super().__init__('video_lane_detection')
        
        # ROS2 퍼블리셔 및 서브스크라이버 설정
        self.image_subscriber = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.steering_angle_publisher = self.create_publisher(Int32, 'steering_angle', 10)
        
        # CvBridge 객체 생성
        self.bridge = CvBridge()
        
        # 조향각 초기값 설정
        self.current_steering_angle = 90

    def image_callback(self, msg):
        try:
            # ROS2 Image 메시지를 OpenCV 이미지로 변환
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 프레임 전처리 및 조향각 계산
            processed_frame, steering_angle = self.process_frame(frame)
            
            # 조향각 퍼블리시
            self.publish_steering_angle(steering_angle)
            
            # 처리된 프레임 표시
            cv2.imshow("Processed Frame", processed_frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def process_frame(self, frame):
        n_h, n_w = frame.shape[:2]
        cropped_frame = frame[int(n_h * (1 / 2)):, :]

        # 그레이 변환 및 이진화
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        binary_frame = (gray > 200).astype(np.uint8)
        
        blob_image = self.blob(binary_frame)
        
        # 차선 중심값 탐색
        left_centroid, right_centroid = self.detect_lane_centroids(blob_image)
        
        # 조향각 계산
        if left_centroid and right_centroid:
            steering_angle = self.compute_steering_angle_from_centroids(left_centroid, right_centroid, blob_image.shape[1])
        elif right_centroid:
            steering_angle = self.steering_angle_from_one_lane(blob_image.shape[1], right_centroid, True)
        elif left_centroid:
            steering_angle = self.steering_angle_from_one_lane(blob_image.shape[1], left_centroid, False)
        else:
            steering_angle = 90
        
        output_frame = self.visualize(cropped_frame, steering_angle, left_centroid, right_centroid)
        return output_frame, steering_angle

    def blob(self, frame):
        kernel = np.ones((11, 11), np.uint8)
        erode = cv2.erode(frame, kernel)
        dilate = cv2.dilate(erode, kernel)
        return dilate

    def detect_lane_centroids(self, binary_image):
        height, width = binary_image.shape
        left_half, right_half = binary_image[:, :width // 4], binary_image[:, 3*(width // 4):]
        
        left_moments = cv2.moments(left_half)
        right_moments = cv2.moments(right_half)
        
        left_centroid = (int(left_moments['m10'] / left_moments['m00']), int(left_moments['m01'] / left_moments['m00'])) if left_moments['m00'] > 0 else None
        right_centroid = (int(right_moments['m10'] / right_moments['m00']) + 3*(width // 4), int(right_moments['m01'] / right_moments['m00'])) if right_moments['m00'] > 0 else None
        
        return left_centroid, right_centroid
    
    def compute_steering_angle_from_centroids(self, left_centroid, right_centroid, frame_width):
        mid_x = (left_centroid[0] + right_centroid[0]) // 2
        frame_center_x = frame_width // 2
        offset = mid_x - frame_center_x
        angle_to_mid_radian = math.atan(offset / 240)
        return int(angle_to_mid_radian * 180.0 / math.pi) + 90

    def steering_angle_from_one_lane(self, frame_width, centroid, is_right):
        frame_center_x = frame_width // 2 #320 (폭:640/높이:240)
        lane_x = centroid[0]
        offset = lane_x - (frame_center_x+280) if is_right else (frame_center_x-280) - lane_x #40과 600위치        
        base_angle = math.degrees(math.atan(offset / 240))
        steering_angle = 90 + base_angle if is_right else 90 - base_angle
        return int(steering_angle)

    def visualize(self, frame, steering_angle, left_centroid=None, right_centroid=None):
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        center_y = frame_height
        
        if left_centroid:
            cv2.circle(frame, left_centroid, 5, (255, 255, 0), -1)
        if right_centroid:
            cv2.circle(frame, right_centroid, 5, (0, 255, 0), -1)
        
        steering_radian = (steering_angle - 90) * (math.pi / 180.0)
        line_length = 200
        end_x = int(frame_center_x + line_length * math.sin(steering_radian))
        end_y = int(center_y - line_length * math.cos(steering_radian))
        
        cv2.line(frame, (frame_center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Steering Angle: {steering_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def publish_steering_angle(self, steering_angle):
        msg = Int32()
        msg.data = steering_angle
        self.steering_angle_publisher.publish(msg)
        self.get_logger().info(f"Published Steering Angle: {steering_angle}")


def main(args=None):
    rclpy.init(args=args)
    node = VideoLaneDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()