export CUDA_VISIBLE_DEVICES=1

# 1s -> 2440 points

# MLP
python attack.py --model_type=MLP --data=Spectrum --data_path=data/psd.mat \
    --train_epochs=40 --learning_rate=1e-3 --delta=0.05 \
    --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --hiden=1024 \
    --exp_name=0527_0 \
    --resume=checkpoint/Spectrum/MLP/0527_0_hiden1024_in32out128_middle1024_lr0.01_bs32_epoch40/final_checkpoint.pth