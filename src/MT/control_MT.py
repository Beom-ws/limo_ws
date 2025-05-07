import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive  # Ackermann 사용
from std_msgs.msg import Int32, Bool
import math

class LimoControl(Node):
    def __init__(self):
        super().__init__('limo_control')

        # ackermann_cmd 퍼블리셔
        self.publisher_ = self.create_publisher(AckermannDrive, 'limo/ack_cmd', 10)

        # Setting for subscriber of e_stop and lane_detection
        self.e_stop_subscription = self.create_subscription(
            Bool,
            'e_stop',
            self.e_stop_callback,
            10)

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

    def e_stop_callback(self, msg):
        """e_stop 값 저장"""
        self.e_stop_flag = msg.data

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

        # if e_stop called
        if self.e_stop_flag :
            msg.speed = 0.0
            msg.steering_angle = 0.0
                

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

