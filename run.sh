export CUDA_VISIBLE_DEVICES=1

python main.py --model_type=MLP --data=Spectrum --data_path=data/psd.mat \
    --train_epochs=100 --learning_rate=1e-2 \
    --seq_len=32 --label_len=0 --pred_len=128 --batch_size=32 --hiden=1024 \
    --exp_name=0501_1_hiden1024