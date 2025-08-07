<h1>MMA-DFER: MultiModal Adaptation of unimodal models  for Dynamic Facial Expression Recognition in-the-wild代码复现

## 结构
AudioMAE/：AudioMAE model implementation and related components for audio processing<br>
annotation/：	标注数据<br>
dataloader/：视频帧、音频数据的加载及数据增强等<br>
models/：时间依赖融合模块、总体模型构建<br>
训练日志及结果/ 存放了在DFEW上的两折的训练结果，每一折大概需要训练9小时<br>
audiomae_pretrained.pth	预训练的音频模型权重<br>
mae_pretrain_vit_base.pth	预训练的MAE weights for vision transformer<br>
main.py	训练逻辑<br>
extract_faces_mfaw.py	使用MTCNN在视频中分帧提取人脸<br>
mp42wav.sh	视频中提取音频<br>
train_DFEW.sh	Training script for DFEW dataset<br>
train_MAFW.sh	Training script for MAFW dataset<br>
requirements.txt<br>
scheduler.py：学习率调度<br>

## 结果
set-1<br>
The best accuracy: 63.90431213378906<br>
UAR: 52.40<br>
WAR: 63.82<br>
<img width="1314" height="694" alt="image" src="https://github.com/user-attachments/assets/5a740c5f-2e4a-4184-aaa4-f7c8ea35b397" />
<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/13551130-2d48-4d20-be5f-495df6e11214" />

set-2<br>
The best accuracy: 61.2558708190918<br>
UAR: 49.45<br>
WAR: 61.04<br>
<img width="1314" height="694" alt="image" src="https://github.com/user-attachments/assets/ffecf156-39c3-4fea-9308-27e51eb9aa67" />
<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/61ab5023-9591-46ed-9b24-5a187b5bd560" />


