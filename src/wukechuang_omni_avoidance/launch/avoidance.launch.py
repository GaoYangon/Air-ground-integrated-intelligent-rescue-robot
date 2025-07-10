from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取功能包路径
    pkg_dir = get_package_share_directory('Wukechuang_omni_avoidance')
    # 参数文件路径
    params_path = os.path.join(pkg_dir, 'config', 'params.yaml')
    
    return LaunchDescription([
        # 避障主节点
        Node(
            package='Wukechuang_omni_avoidance',
            executable='omni_avoidance_node',
            name='omni_avoidance',
            parameters=[params_path],  # 加载参数文件
            output='screen',  # 终端输出日志
            respawn=True,  # 节点崩溃后自动重启
            respawn_delay=2.0
        ),
        
        # MAVROS节点（飞控通信，串口需根据实际修改）
        Node(
            package='mavros',
            executable='mavros_node',
            namespace='mavros',
            parameters=[{
                'fcu_url': 'serial:///dev/ttyUSB0:57600',  # 飞控串口，需修改为实际路径
                'system_id': 1,
                'component_id': 100
            }],
            output='screen'
        )
    ])

