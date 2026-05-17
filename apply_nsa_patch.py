import os
import ultralytics

# 定位库路径
lib_path = os.path.dirname(ultralytics.__file__)
kf_file = os.path.join(lib_path, 'trackers', 'utils', 'kalman_filter.py')
bt_file = os.path.join(lib_path, 'trackers', 'basetrack.py')

def patch_kalman():
    with open(kf_file, 'r') as f:
        content = f.read()
    # 修改 project 函数签名，添加 confidence=1.0
    content = content.replace('def project(self, mean, covariance):', 
                              'def project(self, mean, covariance, confidence=1.0):')
    # 注入 NSA 核心数学公式: R = (1-c) * R
    nsa_logic = "        nsa_factor = (1.0 - confidence)\n        innovation_cov = np.diag(np.square(std)) * nsa_factor"
    if 'nsa_factor' not in content:
        content = content.replace('innovation_cov = np.diag(np.square(std))', nsa_logic)
    
    with open(kf_file, 'w') as f:
        f.write(content)
    print("KalmanFilter 签名与逻辑已修改")

def patch_basetrack():
    with open(bt_file, 'r') as f:
        content = f.read()
    # 在 BaseTrack 更新时传入 score
    # 寻找调用 project 的地方并加上 confidence 参数
    content = content.replace('self.kalman_filter.project(self.mean, self.covariance)', 
                              'self.kalman_filter.project(self.mean, self.covariance, confidence=self.score)')
    
    with open(bt_file, 'w') as f:
        f.write(content)
    print(" BaseTrack 调用链已打通")

if __name__ == "__main__":
    patch_kalman()
    patch_basetrack()
    print(" NSA Kalman 链路已全线贯通！请重启实验。")