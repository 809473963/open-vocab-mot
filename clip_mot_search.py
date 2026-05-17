import os
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# ===== 增加这三行：强行关闭需要 libnvrtc 的高级注意力算子，使用标准数学回退 =====
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def load_tracks(result_file):
    """
    读取 MOT 轨迹文件，将数据按照 track_id 进行聚合
    返回格式: {track_id: {frame_id: [x, y, w, h]}}
    """
    tracks = defaultdict(dict)
    with open(result_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id, track_id = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            tracks[track_id][frame_id] = [x, y, w, h]
    return tracks

def main():
    # 1. 基础配置
    result_file = "MOT17-04-results.txt"  
    img_dir = "data/MOT17/train/MOT17-04-FRCNN/img1/" 
    text_prompt = "A cropped photo of a pedestrian viewed from behind, wearing a  backpack on the back."  
    
    print(f" 正在加载 CLIP 模型，设备: RTX 4070...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    #  第一重封锁：强制转为 FP32，彻底避开半精度的 JIT 即时编译
    model = model.float()

    # 2. 正向 + 负向文本特征编码
    # 正向：强调背影、肩带可见
    pos_prompt = "The back of a pedestrian walking away, backpack straps visible on shoulders, rear view"
    # 负向：正面行人、手提包
    neg_prompt = "A person facing the camera, carrying a handbag or shopping bag by hand"
    NEG_WEIGHT = 0.4   # 负向惩罚系数，可调

    print(f" 正向 Prompt: {pos_prompt}")
    print(f" 负向 Prompt: {neg_prompt}  (weight={NEG_WEIGHT})")

    with torch.no_grad(), torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        pos_inputs = clip.tokenize([pos_prompt]).to(device)
        neg_inputs = clip.tokenize([neg_prompt]).to(device)
        pos_features = model.encode_text(pos_inputs)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features = model.encode_text(neg_inputs)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)

    # 3. 聚合轨迹数据
    tracks = load_tracks(result_file)
    print(f" 成功加载 {len(tracks)} 条历史轨迹。")

    # 4. 图像特征提取与余弦匹配
    track_scores = {}
    print(" 正在跨模态对齐轨迹与文本指令...")
    
    for track_id, frame_data in tqdm(tracks.items()):
        frame_ids = sorted(list(frame_data.keys()))
        # 均匀采样 20 帧，覆盖更多角度/时段
        step = max(1, len(frame_ids) // 20)
        sampled_frames = frame_ids[::step][:20]
        
        image_embeddings = []
        for fid in sampled_frames:
            img_path = os.path.join(img_dir, f"{fid:06d}.jpg")
            if not os.path.exists(img_path): continue
            
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            x, y, w, h = map(int, frame_data[fid])
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.shape[0] < 30 or crop_img.shape[1] < 15: continue
                
            pil_img = Image.fromarray(crop_img)
            image_input = preprocess(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad(), torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                image_feature = model.encode_image(image_input)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                image_embeddings.append(image_feature)
        
        if not image_embeddings: continue
            
        track_image_features = torch.cat(image_embeddings)
        pos_sim = (track_image_features @ pos_features.T).squeeze(1)
        neg_sim = (track_image_features @ neg_features.T).squeeze(1)

        # 对抗评分：正向相似度 - 负向惩罚
        combined = pos_sim - NEG_WEIGHT * neg_sim

        # Top-5 均值：需要多帧持续高分才能命中
        top_k_sims, _ = combined.topk(min(5, len(combined)))
        track_scores[track_id] = top_k_sims.mean().item()

    # 5. 锁定目标
    # MIN_SCORE: 对抗评分的绝对下限，低于此值视为无意义匹配
    # REL_DROP:  后续候选比第一名低超过此值则截断（防止强行凑数）
    MIN_SCORE = 0.02
    REL_DROP  = 0.015
    MAX_RESULTS = 3

    if track_scores:
        sorted_ids = sorted(track_scores, key=track_scores.get, reverse=True)
        top1_score = track_scores[sorted_ids[0]]

        selected = []
        for tid in sorted_ids:
            if len(selected) >= MAX_RESULTS:
                break
            score = track_scores[tid]
            if score < MIN_SCORE:
                break   # 绝对阈值：分数太低直接停
            if top1_score - score > REL_DROP:
                break   # 相对阈值：与第一名差距过大则停
            selected.append(tid)

        print("\n" + "="*50)
        print(f"Prompt: '{pos_prompt}'")
        if selected:
            print(f" 找到 {len(selected)} 条匹配轨迹（阈值 min={MIN_SCORE}, drop={REL_DROP}）：")
            for i, tid in enumerate(selected):
                score = track_scores[tid]
                print(f"   [{i+1}] 轨迹 ID: {tid:<4}  对抗得分: {score:.4f}")
        else:
            print(" 未找到满足阈值的匹配轨迹，请调整 Prompt 或降低 MIN_SCORE。")
        print("="*50)

        with open("target_ids.txt", "w") as f:
            for tid in selected:
                f.write(f"{tid}\n")

if __name__ == "__main__":
    main()