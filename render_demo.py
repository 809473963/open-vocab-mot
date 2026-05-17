import os
import cv2
from collections import defaultdict
from tqdm import tqdm

def main():
    # --- 1. 配置路径 ---
    result_file = "MOT17-04-results.txt"  
    img_dir = "data/MOT17/train/MOT17-04-FRCNN/img1/"
    output_video = "multimodal_tracking_demo_V2.mp4"
    
    # 核心修改：将刚才跑出来的 Top-3 ID 放入集合中
    target_ids = {711,1,352}  
    text_prompt = "A pedestrian viewed from behind, wearing a  backpack"

    # --- 2. 加载轨迹数据 ---
    print(" 正在解析时空轨迹网络...")
    tracks = defaultdict(list)
    with open(result_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            fid, tid = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = map(int, map(float, parts[2:6]))
            tracks[fid].append((tid, x, y, w, h))

    # --- 3. 初始化视频流 ---
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if not img_files: return

    first_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    height, width, _ = first_img.shape
    fps = 30  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # --- 4. 逐帧渲染视觉特效 ---
    print(" 正在渲染多模态追踪特效 (时空缝合版)...")
    for img_name in tqdm(img_files):
        fid = int(img_name.split('.')[0])
        img_path = os.path.join(img_dir, img_name)
        frame = cv2.imread(img_path)

        cv2.putText(frame, f"Open-Vocabulary MOT (Spatiotemporal Re-ID)", (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Query: '{text_prompt}'", (30, 90),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Semantic Linked IDs: {list(target_ids)}", (30, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 165, 255), 2)

        for tid, x, y, w, h in tracks.get(fid, []):
            if tid in target_ids:
                #  命中目标集合中的任何一个物理碎片，都进行高亮！
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                cv2.rectangle(frame, (x, y - 35), (x + 280, y), (0, 0, 0), -1)
                # 动态显示当前的物理 ID
                cv2.putText(frame, f"LOCKED (Physical ID: {tid})", (x + 5, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)

        out.write(frame)

    out.release()
    print(f"\n 终极视觉大片渲染完成！视频已保存为 '{output_video}'")

if __name__ == "__main__":
    main()