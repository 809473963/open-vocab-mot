import os
import cv2
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from scipy import signal


def apply_mask(image, mask, color, alpha=0.5):
    """半透明荧光颜色叠加到 mask 区域"""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def butterworth_smooth(values, fps=30.0, fc=3.0):
    """零相位二阶 Butterworth 低通滤波（fc=3 Hz 运动学平滑）"""
    arr = np.array(values, dtype=float)
    if len(arr) < 10:
        return arr
    nyq = 0.5 * fps
    b, a = signal.butter(2, fc / nyq, btype='low')
    padlen = min(3 * max(len(a), len(b)), len(arr) - 1)
    if padlen <= 0:
        return arr
    return signal.filtfilt(b, a, arr, padlen=padlen)


def fft_gait_freq(h_values, fps=30.0):
    """FFT 步态频率分析，返回 (峰值频率 Hz, 是否为生理步态)
    
    判断标准：步态频带(1.0-2.5 Hz)内峰值 > 2 × 带外噪声基底均值。
    相比 3×全谱均值，此方法对俯视监控低振幅步态更鲁棒。
    """
    arr = np.array(h_values, dtype=float)
    if len(arr) < 30:
        return None, False
    h = signal.detrend(arr, type='linear')
    freqs = np.fft.rfftfreq(len(h), d=1.0 / fps)
    psd = np.abs(np.fft.rfft(h)) ** 2 / len(h)
    gait_mask = (freqs >= 1.0) & (freqs <= 2.5)
    if not np.any(gait_mask):
        return None, False
    idx = np.argmax(psd[gait_mask])
    f_peak = freqs[gait_mask][idx]
    peak_power = psd[gait_mask][idx]
    # 带外噪声基底（排除步态频段后的均值）
    noise_mask = ~gait_mask & (freqs > 0)
    noise_floor = np.mean(psd[noise_mask]) if np.any(noise_mask) else np.mean(psd)
    is_gait = peak_power > 2.0 * noise_floor
    return float(f_peak), bool(is_gait)


def precompute_dsp(tracks_by_id, target_ids, fps=30.0, fc=3.0):
    """
    对 target_ids 中每条轨迹进行：
    1. Butterworth 平滑 → smooth_coords[tid][fid] = [x, y, w, h]
    2. FFT 步态分析    → gait_info[tid] = {'freq': float|None, 'is_gait': bool}
    """
    smooth_coords = {}
    gait_info = {}
    for tid in target_ids:
        if tid not in tracks_by_id:
            continue
        fid_data = tracks_by_id[tid]
        frames = sorted(fid_data.keys())
        xs = np.array([fid_data[f][0] for f in frames], dtype=float)
        ys = np.array([fid_data[f][1] for f in frames], dtype=float)
        ws = np.array([fid_data[f][2] for f in frames], dtype=float)
        hs = np.array([fid_data[f][3] for f in frames], dtype=float)

        xs_s = butterworth_smooth(xs, fps, fc)
        ys_s = butterworth_smooth(ys, fps, fc)
        ws_s = butterworth_smooth(ws, fps, fc)
        hs_s = butterworth_smooth(hs, fps, fc)

        smooth_coords[tid] = {
            f: [xs_s[i], ys_s[i], ws_s[i], hs_s[i]]
            for i, f in enumerate(frames)
        }
        f_peak, is_gait = fft_gait_freq(hs, fps)
        gait_info[tid] = {'freq': f_peak, 'is_gait': is_gait}
    return smooth_coords, gait_info


def main():
    # --- 1. 配置路径 ---
    result_file = "MOT17-04-results.txt"
    img_dir = "data/MOT17/train/MOT17-04-FRCNN/img1/"
    output_video = "multimodal_sam_tracking.mp4"
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    FPS = 30.0

    # 从 CLIP 检索结果读取目标 ID（支持自动读取 target_ids.txt）
    if os.path.exists("target_ids.txt"):
        with open("target_ids.txt") as f:
            target_ids = set(int(line.strip()) for line in f if line.strip())
        print(f" 从 target_ids.txt 读取目标 ID: {target_ids}")
    else:
        target_ids = {395, 708, 711}
        print(f" 使用默认目标 ID: {target_ids}")

    text_prompt = "A pedestrian viewed from behind, wearing a black backpack"

    # --- 2. 载入 SAM ---
    print(" 正在将 SAM ViT-B 载入显存...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # --- 3. 解析轨迹（同时建立 by_frame 和 by_id 索引）---
    print(" 正在解析轨迹数据...")
    tracks_by_frame = defaultdict(list)   # {fid: [(tid, x, y, w, h), ...]}
    tracks_by_id    = defaultdict(dict)   # {tid: {fid: [x, y, w, h]}}
    with open(result_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            fid, tid = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            tracks_by_frame[fid].append((tid, int(x), int(y), int(w), int(h)))
            tracks_by_id[tid][fid] = [x, y, w, h]

    # --- 4. DSP 预计算 ---
    print(" 预计算 DSP（Butterworth 平滑 + FFT 步态分析）...")
    smooth_coords, gait_info = precompute_dsp(tracks_by_id, target_ids, fps=FPS)
    for tid in sorted(target_ids):
        info = gait_info.get(tid, {})
        freq_str = f"{info['freq']:.2f} Hz" if info.get('freq') else "N/A"
        gait_str = "生理步态✓" if info.get('is_gait') else "未检出"
        print(f"  ID {tid}: 步态峰值频率 = {freq_str}  ({gait_str})")

    # --- 5. 逐帧渲染 ---
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    first_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    height, width, _ = first_img.shape
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), int(FPS), (width, height))

    print(" 逐帧渲染 SAM + DSP 特效（需要一些时间）...")
    for img_name in tqdm(img_files):
        fid = int(img_name.split('.')[0])
        frame = cv2.imread(os.path.join(img_dir, img_name))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_targets = [t for t in tracks_by_frame.get(fid, []) if t[0] in target_ids]

        if current_targets:
            predictor.set_image(frame_rgb)
            for tid, rx, ry, rw, rh in current_targets:
                # 优先使用 Butterworth 平滑后的坐标，SAM prompt 更稳定
                if tid in smooth_coords and fid in smooth_coords[tid]:
                    sx, sy, sw, sh = smooth_coords[tid][fid]
                    x, y, w, h = int(sx), int(sy), int(sw), int(sh)
                else:
                    x, y, w, h = rx, ry, rw, rh

                input_box = np.array([x, y, x + w, y + h])
                masks, _, _ = predictor.predict(
                    point_coords=None, point_labels=None,
                    box=input_box, multimask_output=False,
                )
                frame = apply_mask(frame, masks[0], color=(0.0, 1.0, 1.0), alpha=0.55)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # 标签：ID + 步态频率
                info = gait_info.get(tid, {})
                if info.get('freq'):
                    freq_tag = f"{info['freq']:.2f}Hz {'✓' if info['is_gait'] else '~'}"
                else:
                    freq_tag = "N/A"
                label = f"SAM | ID:{tid} | Gait:{freq_tag}"
                label_w = len(label) * 10
                cv2.rectangle(frame, (x, max(0, y - 28)), (x + label_w, y), (0, 0, 0), -1)
                cv2.putText(frame, label, (x + 4, max(12, y - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

        # 其余行人暗色框
        for tid, x, y, w, h in tracks_by_frame.get(fid, []):
            if tid not in target_ids:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

        # HUD 顶部信息栏
        cv2.putText(frame, "Pipeline: YOLOv8 -> CLIP (Top-5 Mean) -> SAM + DSP Gait", (30, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(frame, f"Semantic Query: '{text_prompt}'", (30, 86),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"\n 完成！视频已保存为 '{output_video}'")


if __name__ == "__main__":
    main()