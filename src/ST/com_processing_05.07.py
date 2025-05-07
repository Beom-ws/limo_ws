import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection')

        # CVBridge 초기화
        self.br = CvBridge()

        # 카메라 데이터 구독
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            rclpy.qos.qos_profile_sensor_data
        )

        # 조향각 퍼블리시
        self.steering_angle_publisher = self.create_publisher(Int32, 'steering_angle', 10)

        # 조향각 초기값 설정
        self.current_steering_angle = 90

        # 이전 값을 저장하여 x값의 범위가 너무 벗어나는것을 잡음
        self.prev_left_x = None
        self.prev_right_x = None

        self.get_logger().info("Lane Detection Node Initialized")

    def image_callback(self, msg):
        """카메라 토픽에서 받은 이미지를 처리"""
        try:
            # ROS 2 이미지 메시지를 OpenCV 형식으로 변환
            image = self.br.imgmsg_to_cv2(msg, 'bgr8')

            # 프레임 전처리 및 조향각 계산
            processed_frame, steering_angle = self.process_frame(image)

            # 조향각 퍼블리시
            self.publish_steering_angle(steering_angle)

            # 시각화
            cv2.imshow("Lane Detection", processed_frame)
            # cv2.imshow("orignal", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def process_frame(self, frame):
        """프레임 전처리 및 조향각 계산"""
        n_h, n_w = frame.shape[:2]
        cropped_frame = frame[int(n_h * (1 / 2)):, :]

        # 그레이 변환 및 이진화
        gray = self.gray(cropped_frame)
        binary_frame = self.binary_image(gray)
        # cv2.imshow('binary_frame',binary_frame*255)


        e_frame = self.blob(binary_frame)
        cv2.imshow('e_frame',e_frame*255)

        # 차선 중심값 탐색
        left_centroid, right_centroid = self.detect_lane_centroids(e_frame)

        # 조향각 계산
        if left_centroid and right_centroid:
            steering_angle = self.compute_steering_angle_from_centroids(left_centroid, right_centroid, e_frame.shape[1])
        elif right_centroid:
            steering_angle = self.detect_right_one_lane_point(right_centroid, e_frame.shape[1])
        elif left_centroid:
            steering_angle = self.detect_left_one_lane_point(left_centroid, e_frame.shape[1])
        else:
            steering_angle = 90  # 기본값 유지

        # # 🔹 이전 프레임과 비교하여 변화량 제한
        # max_angle_change = 5
        # steering_angle = max(self.current_steering_angle - max_angle_change, 
        #                      min(self.current_steering_angle + max_angle_change, steering_angle))

        # # 🔹 현재 조향각 저장
        # self.current_steering_angle = steering_angle  

        # 시각화
        output_frame = self.visualize(cropped_frame, steering_angle, left_centroid, right_centroid)

        return output_frame, steering_angle

    def gray(self, frame):
        """그레이스케일 변환"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def binary_image(self, image):
        """이진화"""
        return (image > 200).astype(np.uint8)

    def blob(self, image):
        kernel_erode = np.ones((5,5), np.uint8)  # 51x51 크기의 커널 생성
        kernel_dilate = np.ones((5,5), np.uint8)   # 5x5 크기의 커널 생성

        e_i = cv2.erode(image, kernel_erode)  # 침식 연산
        d_i = cv2.dilate(e_i, kernel_dilate)
        return d_i

    # def detect_lane_centroids(self, binary_image):
    #     """왼쪽 및 오른쪽 차선 중심값 계산"""
    #     height, width = binary_image.shape
    #     left_half, right_half = binary_image[:, :width // 2], binary_image[:, width // 2:]

    #     pixel_sum_left, pixel_sum_right = np.sum(left_half, axis=0), np.sum(right_half, axis=0)
    #     pixel_sum_left[pixel_sum_left < 20] = 0
    #     pixel_sum_right[pixel_sum_right < 20] = 0

    #     left_x = np.argmax(pixel_sum_left) + 1 if np.sum(pixel_sum_left) > 0 else None
    #     right_x = np.argmax(pixel_sum_right) + width // 2 if np.sum(pixel_sum_right) > 0 else None

    #     return left_x, right_x

    def detect_lane_centroids(self, binary_image):
        """왼쪽 및 오른쪽 차선 중심값 계산 (변화량 제한 적용)"""
        height, width = binary_image.shape
        left_half, right_half = binary_image[:, :width // 2], binary_image[:, width // 2:]

        pixel_sum_left, pixel_sum_right = np.sum(left_half, axis=0), np.sum(right_half, axis=0)
        pixel_sum_left[pixel_sum_left < 20] = 0
        pixel_sum_right[pixel_sum_right < 20] = 0

        left_x = np.argmax(pixel_sum_left) + 1 if np.sum(pixel_sum_left) > 0 else None
        right_x = np.argmax(pixel_sum_right) + width // 2 if np.sum(pixel_sum_right) > 0 else None

        # 변화량 제한 (이전 프레임 대비 ±10 이상 변화 시 조정)
        if self.prev_left_x is not None and left_x is not None:
            if abs(left_x - self.prev_left_x) > 10:
                if left_x > self.prev_left_x :
                    left_x = self.prev_left_x + 10 
                else :
                    left_x = self.prev_left_x - 10

        if self.prev_right_x is not None and right_x is not None:
            if abs(right_x - self.prev_right_x) > 10:
                if right_x > self.prev_right_x :
                    right_x = self.prev_right_x + 10
                else : 
                    right_x = self.prev_right_x - 10

        # 현재 값 저장
        self.prev_left_x = left_x
        self.prev_right_x = right_x

        return left_x, right_x
        


    def compute_steering_angle_from_centroids(self, left_centroid, right_centroid, frame_width):
        """좌우 차선 중심값 기반 조향각 계산"""
        mid_x = (left_centroid + right_centroid) // 2
        frame_center_x = frame_width // 2
        offset = mid_x - frame_center_x
        angle_to_mid_radian = math.atan(offset / 240)  
        return int(angle_to_mid_radian * 180.0 / math.pi) + 90

    def detect_right_one_lane_point(self, right_centroid, frame_width):
        """오른쪽 차선만 감지될 때 조향각 계산"""
        right_standard_x = frame_width - 30
        offset = right_standard_x - right_centroid
        angle_to_mid_radian = math.atan(-offset / 120)
        return int(angle_to_mid_radian * 180.0 / math.pi) + 90

    def detect_left_one_lane_point(self, left_centroid, frame_width):
        """왼쪽 차선만 감지될 때 조향각 계산"""
        left_standard_x = 30
        offset = left_centroid - left_standard_x
        angle_to_mid_radian = math.atan(offset / 120)
        return int(angle_to_mid_radian * 180.0 / math.pi) + 90

    def visualize(self, frame, steering_angle, left_centroid=None, right_centroid=None):
        """결과 시각화"""
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        center_y = frame_height

        if left_centroid:
            cv2.circle(frame, (left_centroid, 230), 5, (255, 255, 0), -1)

        if right_centroid:
            cv2.circle(frame, (right_centroid, 230), 5, (0, 255, 0), -1)

        steering_radian = (steering_angle - 90) * (math.pi / 180.0)
        line_length = 200
        end_x = int(frame_center_x + line_length * math.sin(steering_radian))
        end_y = int(center_y - line_length * math.cos(steering_radian))

        cv2.line(frame, (frame_center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Steering Angle: {steering_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow('frame',frame)

        return frame

    def publish_steering_angle(self, steering_angle):
        """조향각 퍼블리시"""
        msg = Int32()
        msg.data = steering_angle
        self.steering_angle_publisher.publish(msg)
        self.get_logger().info(f"Published Steering Angle: {steering_angle}")

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
