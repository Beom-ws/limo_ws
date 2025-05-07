# import cv2
# import numpy as np
# import math
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Int32
# import time


# class VideoLaneDetection(Node):
#     def __init__(self, video_path):
#         super().__init__('video_lane_detection')

#         # 동영상 경로
#         self.video_path = '/home/wego/bag_files/1.avi'

#         # 조향각 퍼블리셔 설정
#         self.steering_angle_publisher = self.create_publisher(Int32, 'steering_angle', 10)

#     def process_video(self):
#         """동영상 처리"""
#         cap = cv2.VideoCapture(self.video_path)
#         if not cap.isOpened():
#             self.get_logger().error(f"비디오 열람 실패: {self.video_path}")
#             return

#         self.get_logger().info("비디오 처리 시작..")
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # 프레임 전처리 및 조향각 계산
#             processed_frame, steering_angle = self.process_frame(frame)

#             # 조향각 퍼블리시
#             self.publish_steering_angle(steering_angle)

#             # 처리된 프레임 표시
#             cv2.imshow("left_bottom", processed_frame)
            
#             if cv2.waitKey(0) & 0xFF == ord('q'):  # 'q' 키로 종료
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         self.get_logger().info("비디오 종료.")

#     def process_frame(self, frame):
#         """프레임 전처리 및 조향각 계산""" 
#         ###### 가로=640 / 세로=480 #########
#         # 상단 1/3 잘라내기
#         n_h, n_w = frame.shape[:2]
#         cropped_frame = frame[int(n_h * (2 / 3)):, int(n_w * (1/2)):]
#         cv2.imshow("cropped_frame", cropped_frame)

#         gray_image = self.gray_image(cropped_frame)
#         cv2.imshow("gray", gray_image)
        
#         # 이진화
#         binary_frame = self.binary_image(gray_image)
#         cv2.imshow("binary_frame",binary_frame)

#         # 차선 중심값 및 조향각 계산
#         right_centroid = self.detect_lane_centroids(binary_frame)
#         print(right_centroid)
        
#         # bottom_point = self.detect_lane_bottom_point(binary_frame)

#         # 조향각 계산
#         # if left_centroid and right_centroid:
#         #     steering_angle = self.compute_steering_angle_from_centroids(left_centroid, right_centroid, binary_frame.shape[1])
#         # else:
#         #     steering_angle = self.compute_steering_angle_from_bottom_point(bottom_point, binary_frame.shape[1])
        
#         steering_angle = self.compute_steering_angle_from_centroids(right_centroid, binary_frame.shape[1])

#         # 시각화
#         output_frame = self.visualize(cropped_frame, steering_angle, right_centroid)
#         return output_frame, steering_angle

#     def gray_image(self, image):
#         """그레이스케일"""
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         return gray


#     def binary_image(self, image):
#         """이진화"""
#         binary = (image > 200).astype(np.uint8) * 255
#         return binary

#     def detect_lane_centroids(self, binary_image):
#         """오른쪽 차선 중심위치 계산"""
#         right_pixel = np.sum(binary_image, axis=0)
#         right_x = np.argmax(right_pixel)
#         return right_x

#     # def compute_steering_angle_from_centroids(self, right_centroid, frame_width):
#     #     """좌우 차선 중심값 기반 조향각 계산"""
#     #     frame_center_x = 310 #빼서 방향을 잡기 위한 기준점
#     #     offset = right_centroid - frame_center_x #좌회전 상황에서 -나옴 : 좌회전으로 사용가능
#     #     angle_to_mid_radian = math.atan(offset / 160) # 160 -> max는 480, 여기의 1/3
#     #     return int(angle_to_mid_radian * 180.0 / math.pi) + 90

#     # def compute_steering_angle_from_bottom_point(self, bottom_point, frame_width):
#     #     """하단 차선 점 기반 조향각 계산"""
#     #     if bottom_point:
#     #         bottom_x, _ = bottom_point
#     #         frame_center_x = frame_width // 2
#     #         offset = bottom_x - frame_center_x
#     #         angle_to_mid_radian = math.atan(-offset / 240)
#     #         return int(angle_to_mid_radian * 180.0 / math.pi) + 90
#     #     return 90

#     def visualize(self, frame, steering_angle, right_centroid=None):
#         """결과 시각화"""
#         frame_height, frame_width = frame.shape[:2]
#         frame_center_x = frame_width // 2
#         center_y = frame_height

#         # if bottom_point:
#         #     cv2.circle(frame, bottom_point, 5, (255, 0, 0), -1)

#         # if left_centroid:
#         #     cv2.circle(frame, left_centroid, 5, (255, 255, 0), -1)

#         right_xy = (right_centroid, 159)
#         if right_centroid:
#             cv2.circle(frame, right_xy, 5, (0, 255, 0), -1)

#         steering_radian = (steering_angle - 90) * (math.pi / 180.0)
#         line_length = 200
#         end_x = int(frame_center_x + line_length * math.sin(steering_radian))
#         end_y = int(center_y - line_length * math.cos(steering_radian))

#         cv2.line(frame, (frame_center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
#         cv2.putText(frame, f"Steering Angle: {steering_angle-90}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         return frame

#     def publish_steering_angle(self, steering_angle):
#         """조향각 퍼블리시"""
#         msg = Int32()
#         msg.data = steering_angle
#         self.steering_angle_publisher.publish(msg)
#         self.get_logger().info(f"pub하는 angle 값: {steering_angle}")


# def main(args=None):
#     rclpy.init(args=args)
#     video_path = "1.avi"  # 처리할 동영상 파일 경로
#     node = VideoLaneDetection(video_path)

#     try:
#         node.process_video()
#     except KeyboardInterrupt:
#         node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()



import cv2
import numpy as np
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


class VideoLaneDetection(Node):
    def __init__(self, video_path):
        super().__init__('video_lane_detection')

        # 동영상 경로
        self.video_path = video_path

        # 조향각 퍼블리셔 설정
        self.steering_angle_publisher = self.create_publisher(Int32, 'steering_angle', 10)

    def process_video(self):
        """동영상 처리"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.get_logger().error(f"비디오 열람 실패: {self.video_path}")
            return

        self.get_logger().info("비디오 처리 시작..")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 전처리 및 조향각 계산
            processed_frame, steering_angle = self.process_frame(frame)

            # 조향각 퍼블리시
            self.publish_steering_angle(steering_angle)

            # 처리된 프레임 표시
            cv2.imshow("Processed Frame", processed_frame)
            cv2.imshow("orig", frame)
            
            if cv2.waitKey(0) & 0xFF == ord('q'):  # 'q' 키로 종료
                break

        cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info("비디오 종료.")

    def process_frame(self, frame):
        """프레임 전처리 및 조향각 계산""" 
        h, w = frame.shape[:2]

        # ROI 설정 (하단 1/3 영역, 오른쪽 절반만)
        roi = frame[int(h * (2 / 3)):, int(w * (1/2)):]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 이진화 (적절한 임계값 설정 필요)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow('binary',binary)
        # 오른쪽 차선 중심 검출
        right_x = self.detect_lane_centroid(binary)

        # 차량 중심은 ROI의 x=0으로 설정
        car_x = 280  # ROI 좌측이 차량 기준점이므로 0으로 설정

        # 조향각 계산 (y 증가량은 고정값 사용)
        steering_angle = self.compute_steering_angle(right_x, car_x, roi.shape[1])

        # 시각화
        output_frame = self.visualize(roi, steering_angle, right_x)

        return output_frame, steering_angle

    def detect_lane_centroid(self, binary_image):
        """오른쪽 차선 중심 찾기"""
        print(binary_image)
        pixel_sum = np.sum(binary_image, axis=0)  # 각 열의 픽셀 합 구하기
        
        pixel_sum[pixel_sum < 1000] = 0
        print(pixel_sum)
        
        if (pixel_sum == 0).all():
            right_x = 320
        else :
            right_x = np.argmax(pixel_sum)  # 최대값을 갖는 x 좌표 찾기

        return right_x

    def compute_steering_angle(self, lane_x, car_x, frame_width):
        """조향각 계산"""
        offset = lane_x - car_x  # 차량 중심과 차선 중심의 차이
        y_delta = 320  # y 증가량 (임의의 고정값) -> 전체의 2/3
        
        # 기울기 계산
        if offset == 0:
            return 90  # 정면 유지
        
        angle_rad = math.atan(offset / y_delta)  # 라디안 값
        angle_deg = int(angle_rad * 180.0 / math.pi) + 90  # 도 단위 변환
        if angle_deg < 0:
            angle_deg = angle_deg * 2
        else :
            pass
        return angle_deg  # 90도를 기준으로 좌/우 방향 조정

    def visualize(self, frame, steering_angle, right_x=None):
        """결과 시각화"""
        h, w = frame.shape[:2]
        frame_center_x = 0  # 차량 중심은 ROI의 x=0
        center_y = h

        if right_x is not None:
            right_xy = (right_x, h - 1)
            cv2.circle(frame, right_xy, 5, (0, 255, 0), -1)  # 차선 중심점 표시

        # 조향각 선 그리기
        steering_radian = (steering_angle - 90) * (math.pi / 180.0) ## 라디안으로 설정
        line_length = 100
        end_x = int(frame_center_x + line_length * math.sin(steering_radian))
        end_y = int(center_y - line_length * math.cos(steering_radian))

        cv2.line(frame, (frame_center_x+160, center_y), (end_x+160, end_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Steering Angle: {steering_angle-90}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def publish_steering_angle(self, steering_angle):
        """조향각 퍼블리시"""
        msg = Int32()
        msg.data = steering_angle
        self.steering_angle_publisher.publish(msg)
        self.get_logger().info(f"Published Angle: {steering_angle}")

def main(args=None):
    rclpy.init(args=args)
    video_path = "/home/wego/bag_files/1.avi"
    node = VideoLaneDetection(video_path)

    try:
        node.process_video()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
