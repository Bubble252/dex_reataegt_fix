"""
数据稳定化模块
用于处理手势追踪中的数据跳变问题

使用方法:
from data_stabilizer import DataStabilizer, RobustKalmanFilter

# 创建稳定器
stabilizer = DataStabilizer()

# 在您的主循环中使用
stable_pos, stable_orient, stable_gripper = stabilizer.stabilize_data(
    position, orientation, gripper
)
"""

import numpy as np
from collections import deque
import warnings

class DataStabilizer:
    """
    数据稳定化处理类
    
    主要功能:
    1. 检测位置和方向的跳变
    2. 提供多种稳定化方法
    3. 维护历史数据窗口
    4. 异常统计和监控
    """
    
    def __init__(self, window_size=5, jump_threshold=0.2, angle_jump_threshold=25, 
                 method="hybrid", verbose=False):
        """
        初始化数据稳定器
        
        Args:
            window_size: 历史数据窗口大小
            jump_threshold: 位置跳变阈值(米)
            angle_jump_threshold: 角度跳变阈值(度)
            method: 稳定化方法 ["median", "moving_average", "jump_detection", "hybrid"]
            verbose: 是否打印详细信息
        """
        self.window_size = window_size
        self.jump_threshold = jump_threshold
        self.angle_jump_threshold = np.radians(angle_jump_threshold)
        self.method = method
        self.verbose = verbose
        
        # 历史数据窗口
        self.position_history = deque(maxlen=window_size)
        self.orientation_history = deque(maxlen=window_size)
        self.gripper_history = deque(maxlen=window_size)
        
        # 上一次有效数据
        self.last_valid_position = None
        self.last_valid_orientation = None
        self.last_valid_gripper = None
        
        # 统计信息
        self.jump_count = 0
        self.total_frames = 0
        
        # 预定义权重（用于加权移动平均）
        self.weights = {
            3: np.array([0.2, 0.3, 0.5]),
            4: np.array([0.15, 0.25, 0.3, 0.3]),
            5: np.array([0.1, 0.15, 0.2, 0.25, 0.3]),
            6: np.array([0.08, 0.12, 0.15, 0.2, 0.22, 0.23]),
            7: np.array([0.07, 0.1, 0.13, 0.15, 0.18, 0.19, 0.18])
        }
        
        if self.verbose:
            print(f"数据稳定器初始化完成 - 方法: {method}, 窗口: {window_size}, 位置阈值: {jump_threshold}, 角度阈值: {angle_jump_threshold}度")
    
    def set_method(self, method):
        """动态切换稳定化方法"""
        if method in ["median", "moving_average", "jump_detection", "hybrid"]:
            self.method = method
            if self.verbose:
                print(f"切换到稳定化方法: {method}")
        else:
            raise ValueError("不支持的稳定化方法")
    
    def set_thresholds(self, jump_threshold=None, angle_jump_threshold=None):
        """动态调整阈值"""
        if jump_threshold is not None:
            self.jump_threshold = jump_threshold
        if angle_jump_threshold is not None:
            self.angle_jump_threshold = np.radians(angle_jump_threshold)
        
        if self.verbose:
            print(f"阈值已更新 - 位置: {self.jump_threshold}, 角度: {np.degrees(self.angle_jump_threshold)}度")
    
    def is_position_jump(self, new_pos, last_pos):
        """检测位置是否发生跳变"""
        if last_pos is None or new_pos is None:
            return False
        distance = np.linalg.norm(np.array(new_pos) - np.array(last_pos))
        return distance > self.jump_threshold
    
    def is_orientation_jump(self, new_orient, last_orient):
        """检测方向是否发生跳变"""
        if last_orient is None or new_orient is None:
            return False
        angle_diff = np.abs(np.array(new_orient) - np.array(last_orient))
        # 处理角度的周期性 (-π到π)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        return np.any(angle_diff > self.angle_jump_threshold)
    
    def median_filter(self, data_history):
        """中位数滤波"""
        if len(data_history) == 0:
            return None
        data_array = np.array(list(data_history))
        return np.median(data_array, axis=0)
    
    def moving_average(self, data_history, weights=None):
        """加权移动平均"""
        if len(data_history) == 0:
            return None
        
        history_len = len(data_history)
        if weights is None:
            # 使用预定义权重或等权重
            if history_len in self.weights:
                weights = self.weights[history_len]
            else:
                weights = np.ones(history_len) / history_len
        else:
            weights = weights[-history_len:]  # 截取对应长度
            weights = weights / np.sum(weights)  # 归一化
        
        data_array = np.array(list(data_history))
        return np.average(data_array, axis=0, weights=weights)
    
    def stabilize_data(self, position, orientation, gripper):
        """
        数据稳定化主函数
        
        Args:
            position: [x, y, z] 位置数组
            orientation: [roll, pitch, yaw] 方向数组(弧度)
            gripper: float 夹持器开合度
            
        Returns:
            tuple: (stable_position, stable_orientation, stable_gripper)
        """
        self.total_frames += 1
        
        # 检测跳变
        pos_jump = self.is_position_jump(position, self.last_valid_position)
        orient_jump = self.is_orientation_jump(orientation, self.last_valid_orientation)
        
        if pos_jump or orient_jump:
            self.jump_count += 1
            if self.verbose:
                print(f"检测到数据跳变! 位置跳变: {pos_jump}, 方向跳变: {orient_jump}, 累计: {self.jump_count}/{self.total_frames}")
        
        # 根据方法进行稳定化
        if self.method == "median":
            return self._median_stabilization(position, orientation, gripper)
        elif self.method == "moving_average":
            return self._moving_average_stabilization(position, orientation, gripper)
        elif self.method == "jump_detection":
            return self._jump_detection_stabilization(position, orientation, gripper, pos_jump, orient_jump)
        elif self.method == "hybrid":
            return self._hybrid_stabilization(position, orientation, gripper, pos_jump, orient_jump)
        else:
            # 默认返回原始数据
            return position, orientation, gripper
    
    def _median_stabilization(self, position, orientation, gripper):
        """中位数滤波稳定化"""
        self.position_history.append(position)
        self.orientation_history.append(orientation)
        self.gripper_history.append(gripper)
        
        stable_pos = self.median_filter(self.position_history)
        stable_orient = self.median_filter(self.orientation_history)
        stable_gripper = self.median_filter(self.gripper_history)
        
        # 更新有效数据
        self._update_valid_data(stable_pos, stable_orient, stable_gripper)
        return stable_pos, stable_orient, stable_gripper
    
    def _moving_average_stabilization(self, position, orientation, gripper):
        """加权移动平均稳定化"""
        self.position_history.append(position)
        self.orientation_history.append(orientation)
        self.gripper_history.append(gripper)
        
        stable_pos = self.moving_average(self.position_history)
        stable_orient = self.moving_average(self.orientation_history)
        stable_gripper = self.moving_average(self.gripper_history)
        
        self._update_valid_data(stable_pos, stable_orient, stable_gripper)
        return stable_pos, stable_orient, stable_gripper
    
    def _jump_detection_stabilization(self, position, orientation, gripper, pos_jump, orient_jump):
        """跳变检测稳定化"""
        # 位置处理
        if pos_jump and len(self.position_history) > 0:
            stable_pos = self.median_filter(self.position_history)
            if self.verbose:
                print(f"位置跳变替换: {position} -> {stable_pos}")
        else:
            stable_pos = position
            self.position_history.append(position)
        
        # 方向处理
        if orient_jump and len(self.orientation_history) > 0:
            stable_orient = self.median_filter(self.orientation_history)
            if self.verbose:
                print(f"方向跳变替换")
        else:
            stable_orient = orientation
            self.orientation_history.append(orientation)
        
        # 夹持器总是添加到历史
        self.gripper_history.append(gripper)
        stable_gripper = self.median_filter(self.gripper_history)
        
        self._update_valid_data(stable_pos, stable_orient, stable_gripper)
        return stable_pos, stable_orient, stable_gripper
    
    def _hybrid_stabilization(self, position, orientation, gripper, pos_jump, orient_jump):
        """混合方法稳定化（推荐）"""
        # 先添加到历史记录
        self.position_history.append(position)
        self.orientation_history.append(orientation)
        self.gripper_history.append(gripper)
        
        # 如果检测到跳变，使用中位数滤波
        if pos_jump or orient_jump:
            stable_pos = self.median_filter(self.position_history)
            stable_orient = self.median_filter(self.orientation_history)
            stable_gripper = self.median_filter(self.gripper_history)
            if self.verbose:
                print(f"应用混合稳定化: 位置跳变={pos_jump}, 方向跳变={orient_jump}")
        else:
            # 正常情况下使用轻微的移动平均
            stable_pos = self.moving_average(self.position_history)
            stable_orient = self.moving_average(self.orientation_history)
            stable_gripper = self.moving_average(self.gripper_history)
        
        self._update_valid_data(stable_pos, stable_orient, stable_gripper)
        return stable_pos, stable_orient, stable_gripper
    
    def _update_valid_data(self, position, orientation, gripper):
        """更新有效数据记录"""
        if position is not None:
            self.last_valid_position = position
        if orientation is not None:
            self.last_valid_orientation = orientation
        if gripper is not None:
            self.last_valid_gripper = gripper
    
    def get_statistics(self):
        """获取统计信息"""
        jump_rate = self.jump_count / max(self.total_frames, 1) * 100
        return {
            "total_frames": self.total_frames,
            "jump_count": self.jump_count,
            "jump_rate_percent": jump_rate,
            "current_method": self.method,
            "window_size": len(self.position_history)
        }
    
    def reset(self):
        """重置稳定器状态"""
        self.position_history.clear()
        self.orientation_history.clear()
        self.gripper_history.clear()
        self.last_valid_position = None
        self.last_valid_orientation = None
        self.last_valid_gripper = None
        self.jump_count = 0
        self.total_frames = 0
        if self.verbose:
            print("数据稳定器已重置")


class RobustKalmanFilter:
    """
    增强型卡尔曼滤波器封装
    在原有卡尔曼滤波器基础上增加异常检测功能
    """
    
    def __init__(self, base_filter, innovation_threshold=2.5, verbose=False):
        """
        初始化增强型卡尔曼滤波器
        
        Args:
            base_filter: 基础卡尔曼滤波器实例
            innovation_threshold: 创新阈值（用于异常检测）
            verbose: 是否打印详细信息
        """
        self.base_filter = base_filter
        self.innovation_threshold = innovation_threshold
        self.rejected_count = 0
        self.total_updates = 0
        self.verbose = verbose
        
        if self.verbose:
            print(f"增强型卡尔曼滤波器初始化 - 创新阈值: {innovation_threshold}")
    
    def set_innovation_threshold(self, threshold):
        """动态调整创新阈值"""
        self.innovation_threshold = threshold
        if self.verbose:
            print(f"创新阈值更新为: {threshold}")
    
    def predict(self):
        """预测步骤"""
        return self.base_filter.predict()
    
    def update(self, measurement):
        """
        带异常检测的更新步骤
        
        Args:
            measurement: 测量值 [x, y, z, roll, pitch, yaw, openness]
        """
        self.total_updates += 1
        
        try:
            # 获取当前预测状态
            predicted_state = self.base_filter.get_state()
            
            # 构造预测测量值
            predicted_measurement = np.array([
                predicted_state['position'][0], 
                predicted_state['position'][1], 
                predicted_state['position'][2],
                predicted_state['euler_radians'][0], 
                predicted_state['euler_radians'][1], 
                predicted_state['euler_radians'][2],
                predicted_state['openness']
            ])
            
            # 计算创新(innovation) - 测量值与预测值的差异
            innovation = measurement - predicted_measurement
            
            # 异常检测：检查位置和角度创新是否过大
            position_innovation = np.linalg.norm(innovation[:3])
            angle_innovation = np.linalg.norm(innovation[3:6])
            
            # 如果创新过大，认为是异常值
            if (position_innovation > self.innovation_threshold or 
                angle_innovation > self.innovation_threshold):
                
                self.rejected_count += 1
                if self.verbose:
                    print(f"拒绝异常测量: 位置创新={position_innovation:.3f}, "
                          f"角度创新={angle_innovation:.3f}, "
                          f"累计拒绝={self.rejected_count}/{self.total_updates}")
                # 不进行更新，只返回预测值
                return
            
            # 正常更新
            self.base_filter.update(measurement)
            
        except Exception as e:
            # 如果出现任何异常，跳过此次更新
            if self.verbose:
                print(f"卡尔曼更新异常: {e}")
            self.rejected_count += 1
    
    def get_state(self):
        """获取当前状态"""
        return self.base_filter.get_state()
    
    def get_statistics(self):
        """获取统计信息"""
        rejection_rate = self.rejected_count / max(self.total_updates, 1) * 100
        return {
            "total_updates": self.total_updates,
            "rejected_count": self.rejected_count,
            "rejection_rate_percent": rejection_rate,
            "innovation_threshold": self.innovation_threshold
        }
    
    def reset(self):
        """重置滤波器"""
        self.base_filter.reset()
        self.rejected_count = 0
        self.total_updates = 0
        if self.verbose:
            print("增强型卡尔曼滤波器已重置")


# 便捷工厂函数
def create_stabilizer(method="hybrid", window_size=5, jump_threshold=0.2, 
                     angle_jump_threshold=25, verbose=False):
    """
    创建数据稳定器的便捷函数
    
    Args:
        method: 稳定化方法
        window_size: 窗口大小
        jump_threshold: 位置跳变阈值(米)
        angle_jump_threshold: 角度跳变阈值(度)
        verbose: 详细输出
        
    Returns:
        DataStabilizer: 配置好的稳定器实例
    """
    return DataStabilizer(
        window_size=window_size,
        jump_threshold=jump_threshold,
        angle_jump_threshold=angle_jump_threshold,
        method=method,
        verbose=verbose
    )

def create_robust_kalman(base_filter, innovation_threshold=2.5, verbose=False):
    """
    创建增强型卡尔曼滤波器的便捷函数
    
    Args:
        base_filter: 基础卡尔曼滤波器
        innovation_threshold: 创新阈值
        verbose: 详细输出
        
    Returns:
        RobustKalmanFilter: 增强型卡尔曼滤波器实例
    """
    return RobustKalmanFilter(
        base_filter=base_filter,
        innovation_threshold=innovation_threshold,
        verbose=verbose
    )


# 使用示例（注释掉的代码，仅作参考）
if __name__ == "__main__":
    """
    使用示例
    """
    # 创建稳定器
    stabilizer = create_stabilizer(
        method="hybrid",
        window_size=5,
        jump_threshold=0.15,
        angle_jump_threshold=20,
        verbose=True
    )
    
    # 模拟一些数据
    test_positions = [
        [0.1, 0.2, 0.3],
        [0.12, 0.22, 0.32],  # 正常变化
        [0.8, 0.9, 1.0],     # 跳变
        [0.14, 0.24, 0.34],  # 回归正常
    ]
    
    test_orientations = [
        [0.1, 0.2, 0.3],
        [0.12, 0.22, 0.32],
        [0.14, 0.24, 0.34],
        [0.16, 0.26, 0.36],
    ]
    
    test_grippers = [0.5, 0.6, 0.7, 0.8]
    
    print("测试数据稳定化:")
    for i, (pos, orient, gripper) in enumerate(zip(test_positions, test_orientations, test_grippers)):
        stable_pos, stable_orient, stable_gripper = stabilizer.stabilize_data(pos, orient, gripper)
        print(f"帧 {i}: 原始位置={pos} -> 稳定位置={stable_pos}")
    
    # 显示统计信息
    stats = stabilizer.get_statistics()
    print(f"\n统计信息: {stats}")