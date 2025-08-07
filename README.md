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
