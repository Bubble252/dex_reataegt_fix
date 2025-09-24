import numpy as np
from scipy.spatial.transform import Rotation as R

class HandPoseKalmanFilter:
    """
    针对手部姿态追踪的卡尔曼滤波器
    状态向量: [x, y, z, vx, vy, vz, roll, pitch, yaw, w_roll, w_pitch, w_yaw, openness, d_openness]
    包含位置、速度、欧拉角、角速度、手指开合度及其变化率
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.1, measurement_noise_pos=1.0, 
                 measurement_noise_rot=0.5, measurement_noise_openness=0.01):
        """
        初始化卡尔曼滤波器
        
        Args:
            dt: 时间步长 (默认30fps)
            process_noise: 过程噪声强度
            measurement_noise_pos: 位置测量噪声
            measurement_noise_rot: 旋转测量噪声  
            measurement_noise_openness: 开合度测量噪声
        """
        self.dt = dt
        self.n_states = 14  # 状态维数
        self.n_measurements = 7  # 测量维数 (x,y,z,roll,pitch,yaw,openness)
        
        # 状态向量初始化
        self.x = np.zeros(self.n_states)  # [x,y,z,vx,vy,vz,roll,pitch,yaw,w_roll,w_pitch,w_yaw,openness,d_openness]
        
        # 协方差矩阵初始化
        self.P = np.eye(self.n_states) * 10.0
        
        # 状态转移矩阵
        self.F = np.eye(self.n_states)
        # 位置-速度关系
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt  
        self.F[2, 5] = dt  # z += vz * dt
        # 角度-角速度关系
        self.F[6, 9] = dt   # roll += w_roll * dt
        self.F[7, 10] = dt  # pitch += w_pitch * dt
        self.F[8, 11] = dt  # yaw += w_yaw * dt
        # 开合度-变化率关系
        self.F[12, 13] = dt # openness += d_openness * dt
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(self.n_states) * process_noise
        # 对速度和角速度添加更多噪声
        self.Q[3:6, 3:6] *= 2.0    # 线速度噪声
        self.Q[9:12, 9:12] *= 2.0  # 角速度噪声
        self.Q[13, 13] *= 0.5      # 开合度变化率噪声
        
        # 测量矩阵 (只测量位置、角度和开合度)
        self.H = np.zeros((self.n_measurements, self.n_states))
        self.H[0, 0] = 1  # 测量 x
        self.H[1, 1] = 1  # 测量 y  
        self.H[2, 2] = 1  # 测量 z
        self.H[3, 6] = 1  # 测量 roll
        self.H[4, 7] = 1  # 测量 pitch
        self.H[5, 8] = 1  # 测量 yaw
        self.H[6, 12] = 1 # 测量 openness
        
        # 测量噪声协方差矩阵
        self.R = np.diag([
            measurement_noise_pos,      # x噪声
            measurement_noise_pos,      # y噪声  
            measurement_noise_pos,      # z噪声
            measurement_noise_rot,      # roll噪声
            measurement_noise_rot,      # pitch噪声
            measurement_noise_rot,      # yaw噪声
            measurement_noise_openness  # openness噪声
        ])
        
        self.initialized = False
        
    def predict(self):
        """预测步骤"""
        # 状态预测
        self.x = self.F @ self.x
        
        # 角度归一化到 [-π, π]
        for i in [6, 7, 8]:  # roll, pitch, yaw
            self.x[i] = self._normalize_angle(self.x[i])
            
        # 开合度限制到 [0, 1]
        self.x[12] = np.clip(self.x[12], 0.0, 1.0)
        
        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement):
        """
        更新步骤
        
        Args:
            measurement: [x, y, z, roll, pitch, yaw, openness] 测量值
        """
        if measurement is None or np.any(np.isnan(measurement)):
            return
            
        z = np.array(measurement)
        
        # 角度归一化
        for i in [3, 4, 5]:  # roll, pitch, yaw in measurement
            z[i] = self._normalize_angle(z[i])
            
        # 开合度限制
        z[6] = np.clip(z[6], 0.0, 1.0)
        
        if not self.initialized:
            # 首次初始化
            self.x[0:3] = z[0:3]      # 位置
            self.x[6:9] = z[3:6]      # 角度
            self.x[12] = z[6]         # 开合度
            self.initialized = True
            return
            
        # 计算创新 (角度差需要特殊处理)
        y = z - self.H @ self.x
        for i in [3, 4, 5]:  # roll, pitch, yaw
            y[i] = self._angle_difference(z[i], self.x[i+3])
            
        # 创新协方差
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新
        self.x = self.x + K @ y
        
        # 角度归一化
        for i in [6, 7, 8]:
            self.x[i] = self._normalize_angle(self.x[i])
            
        # 开合度限制
        self.x[12] = np.clip(self.x[12], 0.0, 1.0)
        
        # 协方差更新
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P
        
    def get_state(self):
        """
        获取当前估计状态
        
        Returns:
            dict: 包含位置、欧拉角、开合度等信息
        """
        return {
            'position': self.x[0:3].copy(),           # [x, y, z]
            'velocity': self.x[3:6].copy(),           # [vx, vy, vz]  
            'euler_angles': self.x[6:9].copy(),       # [roll, pitch, yaw] (弧度)
            'euler_degrees': np.degrees(self.x[6:9]), # [roll, pitch, yaw] (度)
            'angular_velocity': self.x[9:12].copy(),  # [w_roll, w_pitch, w_yaw]
            'openness': self.x[12],                   # 开合度
            'openness_rate': self.x[13]               # 开合度变化率
        }
        
    def reset(self):
        """重置滤波器"""
        self.x.fill(0)
        self.P = np.eye(self.n_states) * 10.0
        self.initialized = False
        
    def _normalize_angle(self, angle):
        """将角度归一化到 [-π, π]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
        
    def _angle_difference(self, angle1, angle2):
        """计算两个角度间的最小差值"""
        diff = angle1 - angle2
        return np.arctan2(np.sin(diff), np.cos(diff))


class MultiHandKalmanFilter:
    """多手追踪的卡尔曼滤波器管理器"""
    
    def __init__(self, max_hands=2, **filter_kwargs):
        """
        初始化多手滤波器
        
        Args:
            max_hands: 最大手数
            **filter_kwargs: 传递给单个滤波器的参数
        """
        self.max_hands = max_hands
        self.filters = {}  # 存储每只手的滤波器
        self.filter_kwargs = filter_kwargs
        self.hand_ids = set()
        
    def update(self, hand_data):
        """
        更新多手数据
        
        Args:
            hand_data: dict, key为hand_id, value为测量数据
        """
        current_ids = set(hand_data.keys())
        
        # 移除不再检测到的手
        removed_ids = self.hand_ids - current_ids
        for hand_id in removed_ids:
            if hand_id in self.filters:
                del self.filters[hand_id]
        
        # 更新现有的手和创建新的滤波器
        for hand_id, measurement in hand_data.items():
            if hand_id not in self.filters:
                # 创建新的滤波器
                self.filters[hand_id] = HandPoseKalmanFilter(**self.filter_kwargs)
                
            # 预测和更新
            self.filters[hand_id].predict()
            self.filters[hand_id].update(measurement)
            
        self.hand_ids = current_ids
        
    def get_states(self):
        """获取所有手的状态"""
        return {hand_id: filter.get_state() 
                for hand_id, filter in self.filters.items()}
                
    def reset(self):
        """重置所有滤波器"""
        for filter in self.filters.values():
            filter.reset()
        self.filters.clear()
        self.hand_ids.clear()


# 便捷函数
def create_hand_filter(fps=30, pos_noise=1.0, rot_noise=0.5, openness_noise=0.01):
    """
    创建手部姿态滤波器的便捷函数
    
    Args:
        fps: 帧率
        pos_noise: 位置测量噪声
        rot_noise: 旋转测量噪声
        openness_noise: 开合度测量噪声
    """
    return HandPoseKalmanFilter(
        dt=1.0/fps,
        measurement_noise_pos=pos_noise,
        measurement_noise_rot=rot_noise,
        measurement_noise_openness=openness_noise
    )