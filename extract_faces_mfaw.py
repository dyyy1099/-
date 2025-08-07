import os
import cv2
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm  # 用于进度条

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 初始化 MTCNN 人脸检测器
mtcnn = MTCNN(image_size=(1280, 720), device=device)

# 路径设置
root_dir = "/root/autodl-tmp/MMA-DFER-main/data/clip"  # 输入视频目录
write_dir = "/root/autodl-tmp/MMA-DFER-main/data/clips_faces"  # 输出帧目录


os.makedirs(write_dir, exist_ok=True)

# 获取所有视频文件列表
video_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

# 显示视频处理进度
for filename in tqdm(video_files, desc="总进度", unit="视频"):
    video_prefix = filename.split('.')[0]
    save_dir = os.path.join(write_dir, video_prefix)
    os.makedirs(save_dir, exist_ok=True)

    # 读取视频
    video_path = os.path.join(root_dir, filename)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nError: 无法打开视频文件 '{video_path}'，已跳过")
        continue  # 跳过无法打开的视频

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 显示帧处理进度
    frame_count = 0
    with tqdm(total=total_frames, desc=f"处理 {filename}", unit="帧", leave=False) as pbar:
        while True:
            ret, im = cap.read()
            if not ret:
                break  # 帧读取完毕
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            bbox = mtcnn.detect(im_rgb)

            if bbox[0] is not None:  # 检测到人脸
                # 合并所有人脸的边界框，取最大范围
                xs = [x[0] for x in bbox[0]]
                ys = [x[1] for x in bbox[0]]
                x2s = [x[2] for x in bbox[0]]
                y2s = [x[3] for x in bbox[0]]
                x1, y1, x2, y2 = min(xs), min(ys), max(x2s), max(y2s)

                # 调整为正方形边界框
                x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
                w = y2 - y1  # 高度
                h = x2 - x1  # 宽度
                if w > h:
                    diff = w - h
                    x2 += diff // 2
                    x1 -= diff // 2
                elif h > w:
                    diff = h - w
                    y2 += diff // 2
                    y1 -= diff // 2

                # 确保边界框不越界
                y1, x1 = max(0, y1), max(0, x1)
                y2, x2 = min(im.shape[0], y2), min(im.shape[1], x2)
                im_cropped = im[y1:y2, x1:x2, :]

            else:  # 未检测到人脸，裁剪中心正方形
                h, w = im.shape[:2]
                ss = min(h // 2, w // 2)
                center_h, center_w = h // 2, w // 2
                im_cropped = im[
                             center_h - ss // 2: center_h + ss // 2,
                             center_w - ss // 2: center_w + ss // 2,
                             :
                             ]

            # 保存裁剪后的帧
            save_path = os.path.join(save_dir, f"{frame_count}.jpg")
            cv2.imwrite(save_path, im_cropped)
            frame_count += 1
            pbar.update(1)
    cap.release()
    print(f"\n视频 {filename} 处理完成，共保存 {frame_count} 帧")