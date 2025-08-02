# GAT-Net
Attention-Based Deep Convolutional Network for Speech Recognition under Multi-scene Noise Environment

#Step1 train the SSL model 
python3 /gemini/code/FT_SSL_XLSR.py \
  --wavdir /gemini/data-1/mytraindata/ \
  --train_label /gemini/data-1/config/train_estoi_PER_WER.txt \
  --val_label /gemini/data-1/config/val_estoi_PER_WER.txt \
  --pretrained_model_path /gemini/pretrain/wav2vec2-large-xlsr-53-chinese-zh-cn \
  --finetune_from_checkpoint /gemini/data-2/best_model_epoch6.pt \
  --outdir /gemini/output/ \
  --total_epochs 30
  
#Step2 Extract the SSL features
python3 /gemini/code/Extract_FT_SSL.py \
  --wavdir /gemini/data-1/mytraindata/ \
  --labelfile /gemini/data-1/config/val_estoi_PER_WER.txt \
  --pretrained_model_path /gemini/pretrain/wav2vec2-large-xlsr-53-chinese-zh-cn \
  --finetuned_ckpt /gemini/data-2/best_model_cont_epoch18.pt \
  --ssl_save_dir /gemini/output/ssl_embed_val \
  --out_csv /gemini/output/predict_val_results.csv
