# # -*- coding: utf-8 -*-
# import rclpy  # ROS 2의 Python을 실행시키기 위한 라이브러리
# from rclpy.node import Node  # ROS 2 노드 생성 클래스

# import cv2  # OpenCV 라이브러리
# import numpy as np  # numpy 라이브러리
# from cv_bridge import CvBridge  # ROS 이미지 메시지와 OpenCV 이미지 변환 라이브러리
# import time  # 시간 측정을 위한 라이브러리

# from sensor_msgs.msg import Image  # ROS 센서 메시지 중 이미지 메시지
# from std_msgs.msg import Int32  # ROS 표준 메시지 중 정수 메시지

# # DetectLine 클래스 정의: 카메라의 입력 데이터를 처리하고 차선 인식 및 거리 계산을 수행
# class DetectLine(Node):
#     def __init__(self):
#         # CVBridge 초기화
#         self.br = CvBridge() # ROS 2 Image 메세지와 OpenCV 간 변환용

#         # 카메라 데이터 구독 설정
#         super().__init__('line_detect')  # 노드 이름을 'line_detect'로 생성 및 초기화
#         self.subscription = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',  # 카메라 데이터 토픽
#             self.image_callback,  # 콜백 함수
#             rclpy.qos.qos_profile_sensor_data  # QoS 설정
#         )
#         self.subscription  # 경고 방지를 위해 추가

#         # 결과 (기준 거리와 실제 거리의 오프셋)를 퍼블리시
#         self.dis_publisher = self.create_publisher(Int32, 'distance_y', 10)

#         # 디버깅용 이미지 퍼블리시
#         self.debug_publisher = self.create_publisher(Image, 'debug_image', 10)

#         # 타이머 설정 (0.1초마다 콜백 실행)
#         self.timer_ = self.create_timer(0.1, self.timer_callback)

#         # 파라미터 선언 및 초기화 (ROI 영역 설정)
#         self.declare_parameter('roi_x_l', 0)
#         self.declare_parameter('roi_x_h', 320)
#         self.declare_parameter('roi_y_l', 400)
#         self.declare_parameter('roi_y_h', 480)
#         self.roi_x_l = self.get_parameter('roi_x_l')
#         self.roi_x_h = self.get_parameter('roi_x_h')
#         self.roi_y_l = self.get_parameter('roi_y_l')
#         self.roi_y_h = self.get_parameter('roi_y_h')

#         # 파라미터 선언 및 초기화 (HSV 색상 범위 설정)
#         self.declare_parameter('lane_h_l', 0)
#         self.declare_parameter('lane_l_l', 90)
#         self.declare_parameter('lane_s_l', 100)
#         self.declare_parameter('lane_h_h', 60)
#         self.declare_parameter('lane_l_h', 220)
#         self.declare_parameter('lane_s_h', 255)

#         lane_h_l = self.get_parameter('lane_h_l')
#         lane_l_l = self.get_parameter('lane_l_l')
#         lane_s_l = self.get_parameter('lane_s_l')
#         lane_h_h = self.get_parameter('lane_h_h')
#         lane_l_h = self.get_parameter('lane_l_h')
#         lane_s_h = self.get_parameter('lane_s_h')

#         # HSV 색상 범위 사용해서 노란색 차선 마스크 설정
#         self.yellow_lane_low = np.array([lane_h_l.value, lane_l_l.value, lane_s_l.value])
#         self.yellow_lane_high = np.array([lane_h_h.value, lane_l_h.value, lane_s_h.value])

#         # 기준 거리 설정
#         self.declare_parameter('reference_distance', 170)
#         self.reference_distance = self.get_parameter('reference_distance')

#         # 디버깅 이미지 순서 설정 (0: ROI, 1: 마스킹, 2: 이미지)
#         self.declare_parameter('debug_image_num', 2)
#         self.debug_sequence = self.get_parameter('debug_image_num')

#         # 구독-퍼블리시 동기화 플래그
#         self.sub_flag = False

#     # YUV 채널 변환 함수
#     def YUV_transform(self, Y):
#         Y_max = np.max(Y)
#         Y_max_3_5 = Y_max * (3/5)

#         # 밝기 변환
#         Y_trans = np.where(
#             (Y > 0) & (Y < Y_max_3_5),
#             Y / 3,
#             np.where(
#                 (Y >= Y_max_3_5) & (Y < Y_max),
#                 (Y * 2) - Y_max,
#                 Y
#             )
#         )
#         Y_trans_uint8 = Y_trans.astype(np.uint8)  # uint8 형식으로 변환
#         return Y_trans_uint8

#     # 타이머 콜백 함수 (디버깅용 이미지 퍼블리시) / 주기적으로 디버깅 이미지를 펍하는 콜백 함수
#     def timer_callback(self):
#         if self.sub_flag:
#             if self.debug_sequence.value == 0:
#                 self.debug_publisher.publish(self.br.cv2_to_imgmsg(self.roi_, 'bgr8'))
#             elif self.debug_sequence.value == 1:
#                 self.debug_publisher.publish(self.br.cv2_to_imgmsg(self.mask_yellow, 'mono8'))
#             else:
#                 self.debug_publisher.publish(self.br.cv2_to_imgmsg(self.image_, 'bgr8'))

#     # 카메라 이미지 데이터를 처리하는 콜백 함수
#     def image_callback(self, msg):
#         st = time.time()  # 시작 시간 기록
#         self.image_ = self.br.imgmsg_to_cv2(msg, 'bgr8')  # ROS 2 메시지를 OpenCV 이미지로 변환

#         # # ROI 설정
#         # self.roi_ = self.image_[self.roi_y_l.value:self.roi_y_h.value,
#         #                         self.roi_x_l.value:self.roi_x_h.value]

#         # YUV 변환 및 밝기 처리
#         yuv = cv2.cvtColor(self.image_, cv2.COLOR_BGR2YUV)
#         Y, _, _ = cv2.split(yuv)
#         y_frame = self.YUV_transform(Y)

#         # 이미지의 무게중심 계산
#         M = cv2.moments(y_frame)
#         if M['m00'] > 0:
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#             cy = self.roi_y_l.value + cy

#             # 기준선과 검출 결과 시각화
#             self.image_ = cv2.line(self.image_,
#                                    (self.reference_distance.value, 0),
#                                    (self.reference_distance.value, 480),
#                                    (0, 255, 0),
#                                    5)
#             self.image_ = cv2.circle(self.image_, (cx, cy), 10, (255, 0, 0), -1)

#             # 기준 거리와 차선 중심의 오프셋 계산
#             distance_to_ref = self.reference_distance.value - cx
#         else:
#             # 차선을 찾지 못한 경우 0 퍼블리시
#             distance_to_ref = 0

#         # 디버깅용 출력
#         cv2.imshow("ROI", self.image_)
#         cv2.imshow("yuv", y_frame)
#         cv2.waitKey(1)

#         # 거리 오프셋 퍼블리시
#         dis = Int32()
#         dis.data = distance_to_ref
#         self.dis_publisher.publish(dis)

#         # 구독-퍼블리시 동기화 플래그 설정
#         self.sub_flag = True
        
#         et = time.time()  # 종료 시간 기록
#         print(et - st)  # 실행 시간 출력

# # 메인 함수
# def main(args=None):
#     rclpy.init(args=args)  # ROS 2 초기화

#     line_detect = DetectLine()  # DetectLine 노드 생성
#     rclpy.spin(line_detect)  # 노드 실행

#     # 종료 시 노드 삭제 및 ROS 2 종료
#     line_detect.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


##########################################################################################
# # -*- coding: utf-8 -*-
# import rclpy
# from rclpy.node import Node

# import cv2
# import numpy as np
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# from std_msgs.msg import Int32

# class DetectLine(Node):
#     def __init__(self):
#         super().__init__('line_detect')

#         # CVBridge 초기화
#         self.br = CvBridge()

#         # 카메라 데이터 구독
#         self.subscription = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.image_callback,
#             rclpy.qos.qos_profile_sensor_data
#         )

#         # 거리 오프셋 퍼블리시
#         self.dis_publisher = self.create_publisher(Int32, 'distance_y', 10)

#         # 파라미터 설정
#         self.initialize_parameters()

#     def initialize_parameters(self):
#         # ROI 영역 설정
#         self.declare_parameter('roi_x_l', 0)
#         self.declare_parameter('roi_x_h', 320)
#         self.declare_parameter('roi_y_l', 400)
#         self.declare_parameter('roi_y_h', 480)
#         self.roi_x_l = self.get_parameter('roi_x_l').value
#         self.roi_x_h = self.get_parameter('roi_x_h').value
#         self.roi_y_l = self.get_parameter('roi_y_l').value
#         self.roi_y_h = self.get_parameter('roi_y_h').value

#         # 기준 거리 설정
#         self.declare_parameter('reference_distance', 160)
#         self.reference_distance = self.get_parameter('reference_distance').value

#     def process_image(self, image):
#         try:
#             # ROI 설정
#             # roi = image[self.roi_y_l:self.roi_y_h, self.roi_x_l:self.roi_x_h]
#             # if roi.size == 0:
#             #     raise ValueError("ROI is empty")

#             # YUV 변환 및 밝기 처리
#             yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(yuv)
#             binary = self.YUV_transform(Y)

#             # 무게중심 계산
#             M = cv2.moments(binary)
#             if M['m00'] > 0:
#                 cx = int(M['m10'] / M['m00'])
#                 distance_to_ref = self.reference_distance - cx
#             else:
#                 distance_to_ref = 0

#             return binary, distance_to_ref

#         except Exception as e:
#             self.get_logger().error(f"Image processing failed: {e}")
#             return None, 0

#     def YUV_transform(self, Y):
#         Y_max = np.max(Y)
#         Y_max_3_5 = Y_max * (3/5)

#         # 밝기 변환
#         Y_trans = np.where(
#             (Y > 0) & (Y < Y_max_3_5),
#             Y / 3,
#             np.where(
#                 (Y >= Y_max_3_5) & (Y < Y_max),
#                 (Y * 2) - Y_max,
#                 Y
#             )
#         )
#         Y_trans_uint8 = Y_trans.astype(np.uint8)  # uint8 형식으로 변환
#         return Y_trans_uint8

#     def image_callback(self, msg):
#         # ROS 메시지를 OpenCV 이미지로 변환
#         image = self.br.imgmsg_to_cv2(msg, 'bgr8')

#         # 이미지 처리
#         binary, distance_to_ref = self.process_image(image)

#         # 거리 오프셋 퍼블리시
#         dis = Int32()
#         dis.data = distance_to_ref
#         self.dis_publisher.publish(dis)

#         # # 디버깅용 시각화
#         # if roi is not None:
#         self.visualize(binary, distance_to_ref)

#     def visualize(self, binary, distance):
#         # 디버깅용 시각화
#         vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
#         cv2.putText(vis, f"Offset: {distance}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Binary ROI", vis)
#         cv2.waitKey(1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = DetectLine()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


####################################################
# #호모모피 필터 적용 -> 너무 무거움
# # -*- coding: utf-8 -*-
# import rclpy
# from rclpy.node import Node

# import cv2
# import numpy as np
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# from std_msgs.msg import Int32


# class DetectLine(Node):
#     def __init__(self):
#         super().__init__('line_detect')

#         # CVBridge 초기화
#         self.br = CvBridge()

#         # 카메라 데이터 구독
#         self.subscription = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.image_callback,
#             rclpy.qos.qos_profile_sensor_data
#         )

#         # 거리 오프셋 퍼블리시
#         self.dis_publisher = self.create_publisher(Int32, 'distance_y', 10)

#         # 기준 거리 설정
#         self.declare_parameter('reference_distance', 160)
#         self.reference_distance = self.get_parameter('reference_distance').value

#     def process_image(self, image):
#         try:
#             # Homomorphic 필터링 적용
#             filtered_image = self.apply_homomorphic_filter(image)

#             # YUV 변환 및 밝기 처리
#             yuv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(yuv)
#             binary = self.binarize(Y)

#             # 무게중심 계산
#             M = cv2.moments(binary)
#             if M['m00'] > 0:
#                 cx = int(M['m10'] / M['m00'])
#                 distance_to_ref = self.reference_distance - cx
#             else:
#                 distance_to_ref = 0

#             return binary, distance_to_ref , filtered_image

#         except Exception as e:
#             self.get_logger().error(f"Image processing failed: {e}")
#             return None, 0

#     def apply_homomorphic_filter(self, img):
#         try:
#             # YUV로 변환 후 Y 채널 분리
#             img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#             y = img_YUV[:, :, 0]

#             rows, cols = y.shape
#             imgLog = np.log1p(np.array(y, dtype='float') / 255)  # Log 변환

#             # Frequency 공간에서 필터링 수행
#             M, N = 2 * rows + 1, 2 * cols + 1
#             sigma = 10
#             (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
#             Xc, Yc = np.ceil(N / 2), np.ceil(M / 2)
#             gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2

#             LPF = np.exp(-gaussianNumerator / (2 * sigma * sigma))
#             HPF = 1 - LPF

#             LPF_shift = np.fft.ifftshift(LPF)
#             HPF_shift = np.fft.ifftshift(HPF)

#             img_FFT = np.fft.fft2(imgLog, (M, N))
#             img_LF = np.real(np.fft.ifft2(img_FFT * LPF_shift, (M, N)))
#             img_HF = np.real(np.fft.ifft2(img_FFT * HPF_shift, (M, N)))

#             # 스케일링 파라미터
#             gamma1, gamma2 = 0.3, 1.5
#             img_adjusting = gamma1 * img_LF[0:rows, 0:cols] + gamma2 * img_HF[0:rows, 0:cols]

#             # exp를 통해 다시 이미지로 변환
#             img_exp = np.expm1(img_adjusting)
#             img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))  # 정규화
#             img_out = np.array(255 * img_exp, dtype='uint8')

#             # Y 채널에 필터링된 이미지 적용
#             img_YUV[:, :, 0] = img_out
#             result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)

#             return result
            
#         except Exception as e:
#             self.get_logger().error(f"Homomorphic filtering failed: {e}")
#             return img

#     def binarize(self, Y):
#         Y_max = np.max(Y)
#         threshold = Y_max * 0.6
#         _, binary = cv2.threshold(Y, threshold, 255, cv2.THRESH_BINARY)
#         return binary

#     def image_callback(self, msg):
#         # ROS 메시지를 OpenCV 이미지로 변환
#         image = self.br.imgmsg_to_cv2(msg, 'bgr8')

#         # 이미지 처리
#         binary, distance_to_ref, filtered_image = self.process_image(image)

#         # 거리 오프셋 퍼블리시
#         dis = Int32()
#         dis.data = distance_to_ref
#         self.dis_publisher.publish(dis)

#         # 디버깅용 시각화
#         if binary is not None:
#             self.visualize(binary, distance_to_ref, filtered_image)

#     def visualize(self, binary, distance, filtered_image):
#         # 디버깅용 시각화
#         vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
#         cv2.putText(vis, f"Offset: {distance}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Binary Image", filtered_image)
#         cv2.waitKey(1)


# def main(args=None):
#     rclpy.init(args=args)
#     node = DetectLine()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()


###############################################
# # -*- coding: utf-8 -*-
# import rclpy
# from rclpy.node import Node

# import cv2
# import numpy as np
# import math
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# from std_msgs.msg import Int32

# class DetectLine(Node):
#     def __init__(self):
#         super().__init__('line_detect')

#         # CVBridge 초기화
#         self.br = CvBridge()

#         # 카메라 데이터 구독
#         self.subscription = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.image_callback,
#             rclpy.qos.qos_profile_sensor_data
#         )

#         # 거리 오프셋 퍼블리시
#         self.dis_publisher = self.create_publisher(Int32, 'distance_y', 10)

#         # 기준 거리 및 조향각 초기화
#         self.initialize_parameters()
#         self.current_steering_angle = 90

#     def initialize_parameters(self):
#         # 기준 거리 설정
#         self.declare_parameter('reference_distance', 160)
#         self.reference_distance = self.get_parameter('reference_distance').value

#     def process_image(self, image):
#         try:
#             intensity_image = self.intensity_image(image)

#             # # 허프 라인으로 선분 검출
#             # line_segments = self.detect_line_segments(intensity_image)

#             # binary 진행
#             binary_image = self.binary_image(intensity_image)

#             # 차선 평균화 및 차선 생성
#             lane_lines = self.average_slope_intercept(image, binary_image)

#             # 조향각 계산
#             steering_angle = self.compute_steering_angle(image, lane_lines)

#             # 조향각 안정화
#             stabilized_angle = self.stabilize_steering_angle(
#                 self.current_steering_angle, steering_angle, len(lane_lines)
#             )
#             self.current_steering_angle = stabilized_angle

#             # 조향 데이터 반환
#             return lane_lines, stabilized_angle

#         except Exception as e:
#             self.get_logger().error(f"Image processing failed: {e}")
#             return [], 90

#         ## 범위로 검출하는 코드
#         # lower_blue = np.array([70, 100, 100])
#         # upper_blue = np.array([140, 255, 255])
#         # mask = cv2.inRange(hsv, lower_blue, upper_blue)

#     def intensity_image(self, frame):
#         YUV = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
#         Y,_,_ = cv2.split(YUV)

#         # 밝기 변환 진행
#         Y_frame = self.YUV_transform(Y)
#         cv2.imshow("Y_frame", Y_frame)
#         return Y_frame

#     def YUV_transform(self, Y):
#         Y_max = np.max(Y)
#         Y_max_3_5 = Y_max * (3/5)

#         # 밝기 변환
#         Y_trans = np.where(
#             (Y > 0) & (Y < Y_max_3_5),
#             Y / 3,
#             np.where(
#                 (Y >= Y_max_3_5) & (Y < Y_max),
#                 (Y * 2) - Y_max,
#                 Y
#             )
#         )
#         Y_trans_uint8 = Y_trans.astype(np.uint8)  # uint8 형식으로 변환
#         return Y_trans_uint8

#     def binary_image(self, image):
#         binary = image > 100
#         return binary

#     # def region_of_interest(self, edges):
#     #     height, width = edges.shape
#     #     mask = np.zeros_like(edges)
#     #     polygon = np.array([[
#     #         (0, height * (1 / 2)),
#     #         (width, height * (1 / 2)),
#     #         (width, height),
#     #         (0, height),
#     #     ]], np.int32)
#     #     cv2.fillPoly(mask, polygon, 255)
#     #     masked_image = cv2.bitwise_and(edges, mask)
#     #     return masked_image

#     # def detect_line_segments(self, cropped_edges):
#     #     rho = 1
#     #     angle = np.pi / 180
#     #     min_threshold = 10
#     #     line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold,
#     #                                     np.array([]), minLineLength=15, maxLineGap=4)
#     #     return line_segments


#     def average_slope_intercept(self, frame, binary_image):
#         lane_lines = []
#         if line_segments is None:
#             return lane_lines

#         height, width, _ = frame.shape
#         left_fit = []
#         right_fit = []

#         boundary = 1 / 3
#         left_region_boundary = width * (1 - boundary)
#         right_region_boundary = width * boundary

#         for line_segment in line_segments:
#             for x1, y1, x2, y2 in line_segment:
#                 if x1 == x2:
#                     continue
#                 fit = np.polyfit((x1, x2), (y1, y2), 1)
#                 slope = fit[0]
#                 intercept = fit[1]
#                 if slope < 0 and x1 < left_region_boundary and x2 < left_region_boundary:
#                     left_fit.append((slope, intercept))
#                 elif slope > 0 and x1 > right_region_boundary and x2 > right_region_boundary:
#                     right_fit.append((slope, intercept))

#         if left_fit:
#             left_fit_average = np.average(left_fit, axis=0)
#             lane_lines.append(self.make_points(frame, left_fit_average))
#         if right_fit:
#             right_fit_average = np.average(right_fit, axis=0)
#             lane_lines.append(self.make_points(frame, right_fit_average))

#         return lane_lines

#     def make_points(self, frame, line):
#         height, width, _ = frame.shape
#         slope, intercept = line
#         y1 = height
#         y2 = int(y1 * 0.5)

#         x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
#         x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
#         return [[x1, y1, x2, y2]]

#     def compute_steering_angle(self, frame, lane_lines):
#         if not lane_lines:
#             return 90

#         height, width, _ = frame.shape
#         if len(lane_lines) == 1:
#             x1, _, x2, _ = lane_lines[0][0]
#             x_offset = x2 - x1
#         else:
#             _, _, left_x2, _ = lane_lines[0][0]
#             _, _, right_x2, _ = lane_lines[1][0]
#             mid = int(width / 2)
#             x_offset = (left_x2 + right_x2) / 2 - mid

#         y_offset = int(height / 2)
#         angle_to_mid_radian = math.atan(x_offset / y_offset)
#         angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
#         return angle_to_mid_deg + 90

#     def stabilize_steering_angle(self, curr_steering_angle, new_steering_angle,
#                                   num_of_lane_lines, max_angle_deviation_two_lines=5,
#                                   max_angle_deviation_one_lane=1):
#         max_angle_deviation = max_angle_deviation_two_lines if num_of_lane_lines == 2 else max_angle_deviation_one_lane
#         angle_deviation = new_steering_angle - curr_steering_angle
#         if abs(angle_deviation) > max_angle_deviation:
#             return int(curr_steering_angle + max_angle_deviation * angle_deviation / abs(angle_deviation))
#         return new_steering_angle

#     def image_callback(self, msg):
#         # ROS 메시지를 OpenCV 이미지로 변환
#         image = self.br.imgmsg_to_cv2(msg, 'bgr8')

#         # 이미지 처리
#         lane_lines, steering_angle = self.process_image(image)

#         # 조향각 퍼블리시
#         dis = Int32()
#         dis.data = steering_angle
#         self.dis_publisher.publish(dis)

#         # 디버깅용 시각화
#         self.visualize(image, lane_lines, steering_angle)

#     def visualize(self, frame, lane_lines, steering_angle):
#         self.draw_lines(frame, lane_lines)
#         cv2.putText(frame, f"Steering Angle: {steering_angle}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Lane Detection", frame)
#         cv2.waitKey(1)

#     def draw_lines(self, frame, lines, line_color=(0, 255, 0), line_width=10):
#         if lines:
#             for line in lines:
#                 for x1, y1, x2, y2 in line:
#                     cv2.line(frame, (x1, y1), (x2, y2), line_color, line_width)


# def main(args=None):
#     rclpy.init(args=args)
#     node = DetectLine()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()


#############################################################
#코드 수정 (중심값으로 진행)

# -*- coding: utf-8 -*-
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

        # 기준 거리 설정
        self.declare_parameter('threshold', 50)
        self.threshold = self.get_parameter('threshold').value

    def process_image(self, image):
        try:
            # n_w = int(image.shape[1])
            # n_h = int(image.shape[2])
            # image = image(:,:n_h*(1/3))
            n_h, n_w = image.shape[:2]

            # 상단 1/3 잘라내기
            cropped_image = image[:, :]  # 높이 기준 1/3 잘라내기

            # YUV 변환 후 밝기 성분 추출 및 변환
            intensity_image = self.intensity_image(image)
            cv2.imshow("intensity_image", intensity_image)

            # 이진화
            binary_image = self.binary_image(intensity_image)
            cv2.imshow("binary_image", binary_image)

            # 좌우 차선 중심값 계산
            left_centroid, right_centroid = self.detect_lane_centroids(binary_image)

            # 하단 차선 점 계산
            bottom_point = self.detect_lane_bottom_point(binary_image)

            # 조향각 계산
            if left_centroid and right_centroid:
                steering_angle = self.compute_steering_angle_from_centroids(
                    left_centroid, right_centroid, image.shape[1]
                )
            else:
                steering_angle = self.compute_steering_angle_from_bottom_point(
                    bottom_point, image.shape[1], self.threshold
                )


            # 조향각 안정화
            stabilized_angle = self.stabilize_steering_angle(
                self.current_steering_angle, steering_angle, 
                2 if (left_centroid and right_centroid) else 1
            )
            self.current_steering_angle = stabilized_angle

            return bottom_point, stabilized_angle, left_centroid, right_centroid

        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")
            return None, 90, None, None

    def intensity_image(self, frame):
        """YUV 변환 후 밝기 성분만 추출 및 변환"""
        YUV = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        Y, _, _ = cv2.split(YUV)
        Y_blur = cv2.GaussianBlur(Y,(9,9),1)

        # 밝기 변환
        Y_transformed = self.YUV_transform(Y_blur)
        return Y_transformed

    def YUV_transform(self, Y):
        """YUV 밝기 변환"""
        Y_max = np.max(Y)
        Y_max_3_5 = Y_max * (3 / 5)

        # 밝기 변환 공식
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
        binary = (image > 100).astype(np.uint8) * 255  # uint8 형식으로 변환
        return binary

    def detect_lane_centroids(self, binary_image):
        """왼쪽 및 오른쪽 차선의 중심값 계산"""
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
        mid_x = (left_centroid[0] + right_centroid[0]) // 2  # [0] = x
        frame_center_x = frame_width // 2
        offset = mid_x - frame_center_x
        angle_to_mid_radian = math.atan(offset / 240) # Y축의 반 크기 만큼을 Y증가량으로 사용, 증가량이 반대인 이유는 원하는 각도의 증가 방향이 달라서  
        return int(angle_to_mid_radian * 180.0 / math.pi) + 90

    def compute_steering_angle_from_bottom_point(self, bottom_point, frame_width, threshold=50):
        """하단 차선 점 기반 조향각 계산"""
        if bottom_point:
            bottom_x, _ = bottom_point
            frame_center_x = frame_width // 2
            offset = bottom_x - frame_center_x

            if abs(offset) > threshold:
                angle_to_mid_radian = math.atan(-offset / 240)
                return int(angle_to_mid_radian * 180.0 / math.pi) + 90
        return 90

    def stabilize_steering_angle(self, curr_angle, new_angle, num_lines, max_dev_two=5, max_dev_one=1):
        """조향각 안정화"""
        max_dev = max_dev_two if num_lines == 2 else max_dev_one
        deviation = new_angle - curr_angle
        if abs(deviation) > max_dev:
            return int(curr_angle + max_dev * deviation / abs(deviation))
        return new_angle

    def image_callback(self, msg):
        """ROS2 콜백 함수"""
        image = self.br.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow('original',image)

        n_h, n_w = image.shape[:2]

        # 상단 1/3 잘라내기
        cropped_image = image[int(n_h * (1 / 3)):, :]  # 높이 기준 1/3 잘라내기

        # 이미지 처리
        bottom_point, steering_angle, left_centroid, right_centroid = self.process_image(cropped_image)

        # 조향각 퍼블리시
        angle_msg = Int32()
        angle_msg.data = steering_angle
        self.steering_angle_publisher.publish(angle_msg)

        # 디버깅용 시각화
        self.visualize(cropped_image, bottom_point, steering_angle, left_centroid, right_centroid)

    def visualize(self, frame, bottom_point, steering_angle, left_centroid=None, right_centroid=None):
        """디버깅 시각화"""
        frame_height, frame_width, _ = frame.shape
        frame_center_x = frame_width // 2
        center_y = frame_height  # 이미지 하단

        if bottom_point:
            cv2.circle(frame, bottom_point, 5, (255, 0, 0), -1)
            # cv2.putText(frame, f"Bottom Point: {bottom_point}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if left_centroid:
            cv2.circle(frame, left_centroid, 5, (255, 255, 0), -1)
            # cv2.putText(frame, f"Left Centroid: {left_centroid}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if right_centroid:
            cv2.circle(frame, right_centroid, 5, (0, 255, 0), -1)
            # cv2.putText(frame, f"Right Centroid: {right_centroid}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 조향각에 따른 진행 방향 선 그리기
        steering_radian = (steering_angle - 90) * (math.pi / 180.0)  # 각도를 라디안으로 변환
        line_length = 200  # 선의 길이
        end_x = int(frame_center_x + line_length * math.sin(steering_radian))
        end_y = int(center_y - line_length * math.cos(steering_radian))  # Y축 방향 반전

        cv2.line(frame, (frame_center_x, center_y), (end_x, end_y), (0, 0, 255), 2)  # 빨간색 선
        # cv2.putText(frame, f"Steering Angle: {steering_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Lane Detection", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
