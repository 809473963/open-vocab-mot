import motmetrics as mm
import numpy as np
import os
from tqdm import tqdm # 增加进度条显示

def evaluate_mot(gt_file, res_file):
    print(f" 正在加载 GT 文件: {gt_file}")
    gt_data = mm.io.loadtxt(gt_file, fmt='mot16')
    print(f" 正在加载 Result 文件: {res_file}")
    res_data = mm.io.loadtxt(res_file, fmt='mot16')

    print(f" 数据加载成功。GT条数: {len(gt_data)}, 结果条数: {len(res_data)}")

    acc = mm.MOTAccumulator(auto_id=True)
    
    # 获取共有帧
    frames = np.union1d(
        gt_data.index.get_level_values(0).unique(),
        res_data.index.get_level_values(0).unique()
    )
    
    print(f" 开始计算 {len(frames)} 帧的匹配指标...")
    
    # 使用 tqdm 显示进度
    for frame in tqdm(frames, desc="计算中"):
        if frame not in gt_data.index.get_level_values(0):
            continue
            
        gt_frame = gt_data.loc[frame]
        res_frame = res_data.loc[frame] if frame in res_data.index.get_level_values(0) else []

        dist_matrix = mm.distances.iou_matrix(
            gt_frame.values[:, :4], 
            res_frame.values[:, :4] if len(res_frame) > 0 else [], 
            max_iou=0.5
        )

        acc.update(
            gt_frame.index.get_level_values(0).values, 
            res_frame.index.get_level_values(0).values if len(res_frame) > 0 else [], 
            dist_matrix
        )

    print(" 正在生成最终报表...")
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'precision', 'recall'], name='MOT17-04-Run')
    print("\n" + "="*50)
    print(summary)

if __name__ == "__main__":
    # 强制打印当前工作目录，防止相对路径坑人
    print(f" 当前运行目录: {os.getcwd()}")
    
    gt_path = os.path.expanduser('~/open-vocab-mot/data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt')
    res_path = 'MOT17-04-results.txt'
    
    # 显式检查文件是否存在
    if not os.path.exists(gt_path):
        print(f" 错误：找不到 GT 文件，请确认路径: {gt_path}")
    elif not os.path.exists(res_path):
        print(f" 错误：找不到结果文件，请确认路径: {res_path}")
    else:
        evaluate_mot(gt_path, res_path)