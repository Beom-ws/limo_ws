# import rclpy
# from rclpy.node import Node

# from geometry_msgs.msg import Twist
# from std_msgs.msg import Int32, Bool

# class LimoControl(Node):
#     def __init__(self):
#         super().__init__('limo_control')
        
#         # cmd_vel 펍 설정
#         self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
#         self.timer_period = 0.1  # seconds
#         self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
#         # Setting for subscriber of e_stop and lane_detection
#         # self.e_stop_subscription = self.create_subscription(
#         #     Bool,
#         #     'e_stop',
#         #     self.e_stop_callback,
#         #     10)
        
#         # distance_y 토픽 구독
#         self.distance_subscription = self.create_subscription(
#             Int32,
#             'distance_y',
#             self.distance_callback,
#             10)
        
#         # 오류 방지
#         # self.e_stop_subscription
#         self.distance_subscription
        
#         # # flag and input value of twisting
#         # self.e_stop_flag = True

#         # distance_y를 저장할 변수 (gap)
#         self.gap = 0

#         # parameter for default speed, p_gain for twist / Parameter 값 설정
#         self.declare_parameter('default_speed', 0.2)
#         self.declare_parameter('p_gain', 0.01)

#         # Parameter 값 가져오기
#         self.default_speed =self.get_parameter('default_speed')
#         self.p_gain =self.get_parameter('p_gain')

#     # def e_stop_callback(self, msg):
#     #     self.e_stop_flag = msg.data
    
#     def distance_callback(self, msg): 
#         self.gap = msg.data # 거리 차이 값 저장

#     def timer_callback(self):
#         # Twist 메세지 생성
#         msg = Twist()
#         msg.linear.x = self.default_speed.value # 전진 속도
#         msg.angular.z = self.gap * self.p_gain.value # 거리 차이에 따른 회전 속도

#         # if e_stop called
#         # if self.e_stop_flag :
#         #     msg.linear.x = 0.0
#         #     msg.angular.z = 0.0
                
#         # cmd_vel 퍼블리싱
#         self.publisher_.publish(msg)
        
# def main(args=None):
#     rclpy.init(args=args)
#     limo_control = LimoControl()

#     rclpy.spin(limo_control)

#     limo_control.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

    ###############################################################################33
    # e stop 제거 코드

# # -*- coding: utf-8 -*-
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from std_msgs.msg import Int32

# class LimoControl(Node):
#     def __init__(self):
#         super().__init__('limo_control')

#         # cmd_vel 퍼블리셔
#         self.publisher_ = self.create_publisher(Twist, 'limo/ack_cmd', 10)

#         # distance_y 구독
#         self.distance_subscription = self.create_subscription(
#             Int32,
#             'distance_y',
#             self.distance_callback,
#             10
#         )

#         # 파라미터 설정
#         self.declare_parameter('default_speed', 0.5)
#         self.declare_parameter('p_gain', 0.01)
#         self.default_speed = self.get_parameter('default_speed').value
#         self.p_gain = self.get_parameter('p_gain').value

#         self.gap = 0  # 거리 오프셋 값 저장

#         # 타이머 생성 (0.1초마다 콜백 실행)
#         self.timer = self.create_timer(0.1, self.timer_callback)

#     def distance_callback(self, msg):
#         # 거리 오프셋 값 저장
#         self.gap = msg.data

#     def timer_callback(self):
#         # Twist 메시지 생성
#         msg = Twist()
#         msg.linear.x = self.default_speed  # 전진 속도
#         msg.angular.z = self.gap * self.p_gain  # 회전 속도 계산

#         # cmd_vel 퍼블리시
#         self.publisher_.publish(msg)
#         self.get_logger().info(f'Publishing cmd_vel: linear.x={msg.linear.x}, angular.z={msg.angular.z}')

# def main(args=None):
#     rclpy.init(args=args)
#     limo_control = LimoControl()
#     rclpy.spin(limo_control)
#     limo_control.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


#################################################################
# e-stop 제거 및 ackermann mode 설정

# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive  # Ackermann 사용
from std_msgs.msg import Int32
import math

class LimoControl(Node):
    def __init__(self):
        super().__init__('limo_control')

        # ackermann_cmd 퍼블리셔
        self.publisher_ = self.create_publisher(AckermannDrive, 'limo/ack_cmd', 10)

        # # distance_y 구독 (거리 오프셋)
        # self.distance_subscription = self.create_subscription(
        #     Int32,
        #     'distance_y',
        #     self.distance_callback,
        #     10
        # )

        # steering_angle 구독 (조향각)
        self.angle_subscription = self.create_subscription(
            Int32,
            'steering_angle',
            self.angle_callback,
            10
        )

        # 파라미터 설정
        self.declare_parameter('default_speed', 0.5)  # 기본 속도 # max 1.1 예상
        self.declare_parameter('p_gain', 1.0)       # 거리에 따른 P 게인
        self.default_speed = self.get_parameter('default_speed').value
        self.p_gain = self.get_parameter('p_gain').value

        self.gap = 0        # 거리 오프셋 값 저장
        self.steering_angle = 0  # 조향각 값 저장

        # 타이머 생성 (0.1초마다 콜백 실행)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def distance_callback(self, msg):
        """거리 오프셋 값 저장"""
        self.gap = msg.data

    def angle_callback(self, msg):
        """조향각 값 저장"""
        self.steering_angle = msg.data
        self.steering_angle = (90-self.steering_angle) * (math.pi / 180) # 각도를 라디안으로 변환


    def timer_callback(self):
        """AckermannDrive 메시지 생성 및 퍼블리시"""
        # AckermannDrive 메시지 생성
        msg = AckermannDrive()
        msg.speed = self.default_speed  # 전진 속도
        msg.steering_angle = self.steering_angle * self.p_gain  # 조향 각도 계산

        # ackermann_cmd 퍼블리시
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing ackermann_cmd: speed={msg.speed}, steering_angle={msg.steering_angle}')

def main(args=None):
    rclpy.init(args=args)
    limo_control = LimoControl()
    rclpy.spin(limo_control)
    limo_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

