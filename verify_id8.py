import os
import cv2

# 配置
result_file = "MOT17-04-results.txt"
img_dir = "data/MOT17/train/MOT17-04-FRCNN/img1/"
target_id = 923  # 刚才 CLIP 找到的 ID
save_dir = "verify_result"

os.makedirs(save_dir, exist_ok=True)

# 提取 ID 8 的画面
count = 0
with open(result_file, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        frame_id, track_id = int(float(parts[0])), int(float(parts[1]))
        
        if track_id == target_id:
            x, y, w, h = map(int, map(float, parts[2:6]))
            img_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                # 裁剪目标
                crop_img = img[max(0, y):min(img.shape[0], y+h), max(0, x):min(img.shape[1], x+w)]
                # 保存图片
                cv2.imwrite(os.path.join(save_dir, f"frame_{frame_id}_id{target_id}.jpg"), crop_img)
                count += 1
                
                if count >= 5: # 保存 5 张看看就行
                    break

print(f"提取完成！去 '{save_dir}' 文件夹查看 ID {target_id} ")
