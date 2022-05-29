export CUDA_VISIBLE_DEVICES=3

# 1s -> 2440 points

# # MLP
# python attack.py --model_type=MLP --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-3 --delta=0.05 \
#     --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --hiden=1024 \
#     --exp_name=0528_0 \
#     --resume=checkpoint/Spectrum/MLPMixer/0528_0_in32out128_lr0.01_bs32_blocks2_epoch40

# MLP-Mixer
# python attack.py --model_type=MLPMixer --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-3 --num_workers=4 \
#     --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --layer=2 \
#     --exp_name=0529_2 --delta=0.05 \
#     --resume=checkpoint/Spectrum/MLPMixer/0528_0_in32out128_lr0.01_bs32_blocks2_epoch40/final_checkpoint.pth


# informer
# python attack.py --model_type=informer --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-3 \
#     --seq_len=32 --label_len=16 --pred_len=128 --batch_size=32  \
#     --d_model=512 --n_heads=8 --e_layers=2 --d_layers=1 --d_ff=2048 \
#     --exp_name=0529_2 \
#     --resume=checkpoint/Spectrum/informer/0529_0_in32out128_en2de1_dmodel512dff1_nhead8_lr0.01_bs32_epoch40/valid_best_checkpoint.pth


# TCN
# python attack.py --model_type=TCN --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-3 --num_workers=4 \
#     --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --layer=2 --hiden=2048 \
#     --exp_name=0529_3 \
#     --resume=checkpoint/Spectrum/TCN/0529_1_in32out128_hiden2048_layer2_lr0.01_bs32_epoch40/valid_best_checkpoint.pth


# RNN
# python attack.py --model_type=rnn --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-3 --num_workers=4 \
#     --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --rnn_layers=2 \
#     --exp_name=0529_2 \
#     --resume=checkpoint/Spectrum/rnn/0529_0_in32out128_layer2_lr0.01_bs32_epoch40/valid_best_checkpoint.pth


# LSTM
# python attack.py --model_type=lstm --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-3 --num_workers=4 \
#     --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --rnn_layers=2 \
#     --exp_name=0529_2 \
#     --resume=checkpoint/Spectrum/lstm/0528_0_in32out128_layer2_lr0.01_bs32_epoch40/valid_best_checkpoint.pth