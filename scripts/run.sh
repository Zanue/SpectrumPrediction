export CUDA_VISIBLE_DEVICES=1

# 1s -> 2440 points

# MLP
# python main.py --model_type=MLP --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-2 \
#     --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --hiden=1024 \
#     --exp_name=0528_0


# MLP-Mixer
# python main.py --model_type=MLPMixer --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-2 --num_workers=4 \
#     --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --layer=2 \
#     --exp_name=0528_0


# informer
# python main.py --model_type=informer --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-2 \
#     --seq_len=32 --label_len=16 --pred_len=128 --batch_size=32  \
#     --d_model=512 --n_heads=8 --e_layers=2 --d_layers=1 --d_ff=2048 \
#     --exp_name=0529_1_amp --use_amp


# TCN
# python main.py --model_type=TCN --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-2 --num_workers=4 \
#     --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --layer=2 --hiden=2048 \
#     --exp_name=0529_1_amp --use_amp


# RNN
# python main.py --model_type=rnn --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-2 --num_workers=4 \
#     --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --rnn_layers=2 \
#     --exp_name=0529_1_amp --use_amp


# LSTM
# python main.py --model_type=lstm --data=Spectrum --data_path=data/psd.mat \
#     --train_epochs=40 --learning_rate=1e-2 --num_workers=4 \
#     --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --rnn_layers=2 \
#     --exp_name=0529_1_amp --use_amp