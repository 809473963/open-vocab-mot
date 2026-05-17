from ultralytics import YOLO
import os

def train_medium_model():
    # 1. 加载官方预训练的 Medium 权重作为起点
    model = YOLO('yolov8m.pt') 

    # 2. 开始训练
    # 针对 RTX 4070 (8GB VRAM) 的优化配置
    results = model.train(
        data='mot17.yaml',      # 数据集配置文件
        epochs=50,               # 训练 50 轮
        imgsz=640,               # 训练建议先用 640，推理再用 1280 提准
        batch=8,                 # Medium模型更大，batch需从 16/24 调小到 8 以防显存溢出
        workers=4,               # 保持 4 个加载线程
        device=0,                # 使用你的 4070 GPU
        
        # --- 内存保护设置 ---
        cache=False,             # 必须设为 False！防止 24GB RAM 被 1024+ 分辨率图片撑爆
        # persistent_workers=True, # 保持工作进程，减少每个 Epoch 开始时的加载延迟
        
        # --- 命名与保存 ---
        project='runs/detect',
        name='yolov8m_mot17_finetune',
        exist_ok=True
    )

if __name__ == '__main__':
    # 建议运行前先执行 pkill -9 python 清理残留内存
    train_medium_model()