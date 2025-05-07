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
#             self.get_logger().error(f"Failed to open video: {self.video_path}")
#             return

#         self.get_logger().info("Starting video processing...")
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # 프레임 전처리 및 조향각 계산
#             processed_frame, steering_angle = self.process_frame(frame)

#             # 조향각 퍼블리시
#             self.publish_steering_angle(steering_angle)

#             # 처리된 프레임 표시
#             cv2.imshow("Processed Frame", processed_frame)
            
#             if cv2.waitKey(0) & 0xFF == ord('q'):  # 'q' 키로 종료
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         self.get_logger().info("Video processing completed.")

#     def process_frame(self, frame):
#         #######################여기부터
#         """프레임 전처리 및 조향각 계산""" 
#         ###### 가로=640 / 세로=480
#         # 상단 1/3 잘라내기
#         n_h, n_w = frame.shape[:2]
#         cropped_frame = frame[int(n_h * (1 / 3)):, :]

#         # YUV 변환 및 밝기 추출
#         YUV = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2YUV)
#         Y, _, _ = cv2.split(YUV)
#         Y_blur = cv2.GaussianBlur(Y, (9, 9), 1)

#         # 밝기 변환
#         Y_transformed = self.YUV_transform(Y_blur)
#         cv2.imshow("Y_transformed",Y_transformed)

#         # 이진화
#         binary_frame = self.binary_image(Y_transformed)
#         cv2.imshow("binary_frame",binary_frame)

#         # 팽창 침식
#         blob_image = self.morph(binary_frame)
#         cv2.imshow("blob_image",blob_image)

#         #엣지
#         edge_frame = self.edge_image(blob_image)
#         cv2.imshow("edge_frame",edge_frame)

#         #허프
#         Hough, Hough_frame = self.Hough(edge_frame,cropped_frame)
#         cv2.imshow("Hough_frame",Hough_frame)

#         # 차선 중심값 및 조향각 계산
#         left_centroid, right_centroid = self.detect_lane_centroids(binary_frame)
#         bottom_point = self.detect_lane_bottom_point(binary_frame)

#         # 조향각 계산
#         if left_centroid and right_centroid:
#             steering_angle = self.compute_steering_angle_from_centroids(left_centroid, right_centroid, binary_frame.shape[1])
#         else:
#             steering_angle = self.compute_steering_angle_from_bottom_point(bottom_point, binary_frame.shape[1])

#         ######################여기까지 러닝타임 0.01초

#         # 시각화
#         output_frame = self.visualize(cropped_frame, bottom_point, steering_angle, left_centroid, right_centroid)
#         return output_frame, steering_angle


#     def YUV_transform(self, Y):
#         """YUV 밝기 변환"""
#         Y_max = np.max(Y)
#         Y_max_3_5 = Y_max * (3 / 5)
#         Y_trans = np.where(
#             (Y > 0) & (Y < Y_max_3_5),
#             Y / 3,
#             np.where(
#                 (Y >= Y_max_3_5) & (Y < Y_max),
#                 (Y * 2) - Y_max,
#                 Y
#             )
#         )
#         return Y_trans.astype(np.uint8)

#     def binary_image(self, image):
#         """이진화"""
#         binary = (image > 100).astype(np.uint8) * 255
#         return binary

#     def edge_image(self, image):
#         """엣지 검출"""
#         edge = cv2.Canny(image,80,100)
#         return edge

#     def morph(self, image):
#         """blob의 팽창 침식"""
#         kernel = np.ones((9,9),np.uint8)
#         kernel_2 = np.ones((3,3),np.uint8)

#         erode = cv2.erode(image,kernel,iterations=1) #iterations : erode를 반복해서 몇번 실행시켜줄지 정함
#         dilate = cv2.dilate(erode,kernel_2,iterations=1) 
#         return dilate

#     def Hough(self, edge, frame):
#         lines = cv2.HoughLines(edge, 1, np.pi/180, 80)
#         if lines is not None:
#             for line in lines:
#                 rho, theta = line[0]
#                 cos, sin = np.cos(theta), np.sin(theta)
#                 cx, cy = rho * cos, rho * sin
#                 x1, y1 = int(cx + 1000 * (-sin)), int(cy + 1000 * cos)
#                 x2, y2 = int(cx + 1000 * sin), int(cy + 1000 * (-cos))
#                 # 원본 사진에 초록색 선으로 표시
#                 cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 1)
#         else:
#             print("No lines detected")
#         return edge, frame

#     def detect_lane_centroids(self, binary_image):
#         """왼쪽 및 오른쪽 차선 중심값 계산"""
#         height, width = binary_image.shape
#         left_half = binary_image[:, :width // 2]
#         right_half = binary_image[:, width // 2:]

#         left_moments = cv2.moments(left_half)
#         right_moments = cv2.moments(right_half)

#         left_centroid = (
#             int(left_moments['m10'] / left_moments['m00']),
#             int(left_moments['m01'] / left_moments['m00'])
#         ) if left_moments['m00'] > 0 else None

#         right_centroid = (
#             int(right_moments['m10'] / right_moments['m00']) + width // 2,
#             int(right_moments['m01'] / right_moments['m00'])
#         ) if right_moments['m00'] > 0 else None

#         return left_centroid, right_centroid

#     def detect_lane_bottom_point(self, binary_image):
#         """차선의 가장 하단 점 계산"""
#         height, width = binary_image.shape
#         bottom_half = binary_image[height // 2:, :]
#         moments = cv2.moments(bottom_half)
#         if moments['m00'] > 0:
#             bottom_x = int(moments['m10'] / moments['m00'])
#             bottom_y = height - 1
#             return (bottom_x, bottom_y)
#         return None

#     def compute_steering_angle_from_centroids(self, left_centroid, right_centroid, frame_width):
#         """좌우 차선 중심값 기반 조향각 계산"""
#         mid_x = (left_centroid[0] + right_centroid[0]) // 2
#         frame_center_x = frame_width // 2
#         offset = mid_x - frame_center_x
#         angle_to_mid_radian = math.atan(offset / 240) # 240 -> max는 480
#         return int(angle_to_mid_radian * 180.0 / math.pi) + 90

#     def compute_steering_angle_from_bottom_point(self, bottom_point, frame_width):
#         """하단 차선 점 기반 조향각 계산"""
#         if bottom_point:
#             bottom_x, _ = bottom_point
#             frame_center_x = frame_width // 2
#             offset = bottom_x - frame_center_x
#             angle_to_mid_radian = math.atan(-offset / 240)
#             return int(angle_to_mid_radian * 180.0 / math.pi) + 90
#         return 90

#     def visualize(self, frame, bottom_point, steering_angle, left_centroid=None, right_centroid=None):
#         """결과 시각화"""
#         frame_height, frame_width = frame.shape[:2]
#         frame_center_x = frame_width // 2
#         center_y = frame_height

#         if bottom_point:
#             cv2.circle(frame, bottom_point, 5, (255, 0, 0), -1)

#         if left_centroid:
#             cv2.circle(frame, left_centroid, 5, (255, 255, 0), -1)e

#         if right_centroid:
#             cv2.circle(frame, right_centroid, 5, (0, 255, 0), -1)

#         steering_radian = (steering_angle - 90) * (math.pi / 180.0)
#         line_length = 200
#         end_x = int(frame_center_x + line_length * math.sin(steering_radian))
#         end_y = int(center_y - line_length * math.cos(steering_radian))

#         cv2.line(frame, (frame_center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
#         cv2.putText(frame, f"Steering Angle: {steering_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         return frame

#     def publish_steering_angle(self, steering_angle):
#         """조향각 퍼블리시"""
#         msg = Int32()
#         msg.data = steering_angle
#         self.steering_angle_publisher.publish(msg)
#         self.get_logger().info(f"Published Steering Angle: {steering_angle}")


# def main(args=None):
#     rclpy.init(args=args)
#     video_path = "2.avi"  # 처리할 동영상 파일 경로
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


# ##########################################################################################################################################################
import cv2
import numpy as np
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import time


class VideoLaneDetection(Node):
    def __init__(self, video_path):
        super().__init__('video_lane_detection')

        # 동영상 경로
        self.video_path = '/home/wego/bag_files/1.avi'

        # 조향각 퍼블리셔 설정
        self.steering_angle_publisher = self.create_publisher(Int32, 'steering_angle', 10)

    def process_video(self):
        """동영상 처리"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.get_logger().error(f"Failed to open video: {self.video_path}")
            return

        self.get_logger().info("Starting video processing...")
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
            
            if cv2.waitKey(0) & 0xFF == ord('q'):  # 'q' 키로 종료
                break

        cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info("Video processing completed.")

    def process_frame(self, frame):
        #######################여기부터
        """프레임 전처리 및 조향각 계산""" 
        ###### 가로=640 / 세로=480 #########
        # 상단 1/3 잘라내기
        n_h, n_w = frame.shape[:2]
        cropped_frame = frame[int(n_h * (1 / 3)):, :]

        # YUV 변환 및 밝기 추출
        YUV = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2YUV)
        Y, _, _ = cv2.split(YUV)
        Y_blur = cv2.GaussianBlur(Y, (9, 9), 1)

        # 밝기 변환
        Y_transformed = self.YUV_transform(Y_blur)
        cv2.imshow("Y_transformed",Y_transformed)

        # 이진화
        binary_frame = self.binary_image(Y_transformed)
        cv2.imshow("binary_frame",binary_frame)

        # 팽창 침식
        blob_image = self.morph(binary_frame)
        cv2.imshow("blob_image",blob_image)

        #엣지
        edge_frame = self.edge_image(blob_image)
        cv2.imshow("edge_frame",edge_frame)

        #허프
        Hough, Hough_frame = self.Hough(edge_frame,cropped_frame)
        cv2.imshow("Hough_frame",Hough_frame)

        # 차선 중심값 및 조향각 계산
        left_centroid, right_centroid = self.detect_lane_centroids(binary_frame)
        bottom_point = self.detect_lane_bottom_point(binary_frame)

        # 조향각 계산
        if left_centroid and right_centroid:
            steering_angle = self.compute_steering_angle_from_centroids(left_centroid, right_centroid, binary_frame.shape[1])
        else:
            steering_angle = self.compute_steering_angle_from_bottom_point(bottom_point, binary_frame.shape[1])

        ######################여기까지 러닝타임 0.01초

        # 시각화
        output_frame = self.visualize(cropped_frame, bottom_point, steering_angle, left_centroid, right_centroid)
        return output_frame, steering_angle


    def YUV_transform(self, Y):
        """YUV 밝기 변환"""
        Y_max = np.max(Y)
        Y_max_3_5 = Y_max * (3 / 5)
        Y_trans = np.where(
            (Y > 0) & (Y < Y_max_3_5),
            Y / 3,
            np.where(
                (Y >= Y_max_3_5) & (Y < Y_max),
                (Y * 2) - Y_max,
                Y
            )
        )
        return Y_trans.astype(np.uint8)

    def binary_image(self, image):
        """이진화"""
        binary = (image > 100).astype(np.uint8) * 255
        return binary

    def edge_image(self, image):
        """엣지 검출"""
        edge = cv2.Canny(image,80,100)
        return edge

    def morph(self, image):
        """blob의 팽창 침식"""
        kernel = np.ones((9,9),np.uint8)
        kernel_2 = np.ones((3,3),np.uint8)

        erode = cv2.erode(image,kernel,iterations=1) #iterations : erode를 반복해서 몇번 실행시켜줄지 정함
        dilate = cv2.dilate(erode,kernel_2,iterations=1) 
        return dilate

    def Hough(self, edge, frame):
        lines = cv2.HoughLines(edge, 1, np.pi/180, 80)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                cos, sin = np.cos(theta), np.sin(theta)
                cx, cy = rho * cos, rho * sin
                x1, y1 = int(cx + 1000 * (-sin)), int(cy + 1000 * cos)
                x2, y2 = int(cx + 1000 * sin), int(cy + 1000 * (-cos))
                # 원본 사진에 초록색 선으로 표시
                cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 1)
        else:
            print("No lines detected")
        return edge, frame

    def detect_lane_centroids(self, binary_image):
        """왼쪽 및 오른쪽 차선 중심값 계산"""
        height, width = binary_image.shape
        left_half = binary_image[:, :width // 2]
        right_half = binary_image[:, width // 2:]

        left_moments = cv2.moments(left_half)
        right_moments = cv2.moments(right_half)

        left_centroid = (
            int(left_moments['m10'] / left_moments['m00']),
            int(left_moments['m01'] / left_moments['m00'])
        ) if left_moments['m00'] > 0 else None

        right_centroid = (
            int(right_moments['m10'] / right_moments['m00']) + width // 2,
            int(right_moments['m01'] / right_moments['m00'])
        ) if right_moments['m00'] > 0 else None

        return left_centroid, right_centroid

    def detect_lane_bottom_point(self, binary_image):
        """차선의 가장 하단 점 계산"""
        height, width = binary_image.shape
        bottom_half = binary_image[height // 2:, :]
        moments = cv2.moments(bottom_half)
        if moments['m00'] > 0:
            bottom_x = int(moments['m10'] / moments['m00'])
            bottom_y = height - 1
            return (bottom_x, bottom_y)
        return None

    def compute_steering_angle_from_centroids(self, left_centroid, right_centroid, frame_width):
        """좌우 차선 중심값 기반 조향각 계산"""
        mid_x = (left_centroid[0] + right_centroid[0]) // 2
        frame_center_x = frame_width // 2
        offset = mid_x - frame_center_x
        angle_to_mid_radian = math.atan(offset / 240) # 240 -> max는 480
        return int(angle_to_mid_radian * 180.0 / math.pi) + 90

    def compute_steering_angle_from_bottom_point(self, bottom_point, frame_width):
        """하단 차선 점 기반 조향각 계산"""
        if bottom_point:
            bottom_x, _ = bottom_point
            frame_center_x = frame_width // 2
            offset = bottom_x - frame_center_x
            angle_to_mid_radian = math.atan(-offset / 240)
            return int(angle_to_mid_radian * 180.0 / math.pi) + 90
        return 90

    def visualize(self, frame, bottom_point, steering_angle, left_centroid=None, right_centroid=None):
        """결과 시각화"""
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        center_y = frame_height

        if bottom_point:
            cv2.circle(frame, bottom_point, 5, (255, 0, 0), -1)

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
        """조향각 퍼블리시"""
        msg = Int32()
        msg.data = steering_angle
        self.steering_angle_publisher.publish(msg)
        self.get_logger().info(f"Published Steering Angle: {steering_angle}")


def main(args=None):
    rclpy.init(args=args)
    video_path = "1.avi"  # 처리할 동영상 파일 경로
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
