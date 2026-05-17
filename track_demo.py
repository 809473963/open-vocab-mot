import cv2
from ultralytics import YOLO
import os

# 1. 加载模型
model_path = os.path.expanduser('~/cv_project/runs/detect/runs/detect/yolov8m_mot17_finetune/weights/best.pt')
model = YOLO(model_path)

# 2. 配置跟踪
tracker_config = "botsort.yaml" 
output_txt = "MOT17-04-results.txt"
output_video = "tracking_stable_demo.mp4"

# 运行跟踪
# stream=True 对长视频非常友好，能防止内存溢出
results = model.track(
    source='data/MOT17/train/MOT17-04-FRCNN/img1', 
    stream=True, 
    tracker=tracker_config,
    conf=0.2, 
    iou=0.5, 
    persist=True, 
    imgsz=1280,
    augment=True 
)

video_writer = None

print(" 正在同步生成结果文档与视频...")

with open(output_txt, "w") as f:
    for r in results:
        # --- 逻辑 A: 提取帧号 ---
        # 假设文件名是 000001.jpg，提取出数字 1
        frame_id = int(os.path.basename(r.path).split('.')[0])
        
        # --- 逻辑 B: 写入 MOT 格式结果 ---
        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            # xywh 是 [x_center, y_center, width, height]
            # 但 MOT 格式要求 [left, top, width, height]
            # xyxy 是 [x1, y1, x2, y2]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            for box_id, box, conf in zip(ids, boxes, confs):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                # 写入格式: <frame>, <id>, <x1>, <y1>, <w>, <h>, <conf>, -1, -1, -1
                f.write(f"{frame_id},{box_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n")
        
        # --- 逻辑 C: 写入视频帧 ---
        im_array = r.plot(labels=True, conf=False)
        if video_writer is None:
            h, w, _ = im_array.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_writer = cv2.VideoWriter(output_video, fourcc, 30, (w, h))
        
        video_writer.write(im_array)

if video_writer:
    video_writer.release()
    print(f" 处理完成！\nTXT结果: {os.path.abspath(output_txt)}\n视频结果: {os.path.abspath(output_video)}")