import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool ## True / False를 전달하는 메세지 타입
from sensor_msgs.msg import LaserScan
from math import pi

class Limo_estop(Node):
    def __init__(self):
        super().__init__('limo_e_stop')
        self.subscription = self.create_subscription(LaserScan,'/scan', self.pose_cb, rclpy.qos.qos_profile_sensor_data)

        self.publisher_ = self.create_publisher(Bool, 'e_stop', 10)
        
        self.lidar_msg = None
        self.lidar_flag = False
        
        self.create_timer(0.01, self.laser_callback)
        
    def pose_cb(self, msg):
        if msg is not None:
            self.lidar_msg = msg
            self.lidar_flag = True
            self.get_logger().info("라이더 스캔중...")
        else:
            self.get_logger().info("라이더 스캔값 없음")
    
    def laser_callback(self):
        if self.lidar_flag and self.lidar_msg is not None:
            obstacle_count = 0
            estop = Bool()
            
            for index, data in enumerate(self.lidar_msg.ranges): ## data는 해당 위치에서 장애물까지의 거리를 나타냄 / index는 스캔 각도의 몇번째 위치인지 
                degree = (self.lidar_msg.angle_min + self.lidar_msg.angle_increment*index) * (180/pi)
                if -30 < degree < 30 and 0 < data < 0.4: ## 40cm 이내면
                    obstacle_count += 1
            if obstacle_count < 10 : ## 장애물이 감지되지 않은 경우
                estop.data = False # 주행 가능
            else : ## 장애물이 감지된 경우
                estop.data = True # 정지
                
            self.publisher_.publish(estop)
            self.lidar_flag = False
            
def main(args=None):
    rclpy.init()

    limo_e_stop = Limo_estop()
    rclpy.spin(limo_e_stop)

    limo_e_stop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()