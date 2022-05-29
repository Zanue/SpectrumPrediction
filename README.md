# Spectrum Prediction

## Contributions
1. Implemented multiple models for spectrum prediction ('TCN', 'informer', 'MLPMixer', 'rnn', 'lstm').
2. Calculated the params and latency of each model (params, latency).
3. Use automatic mixed precision training (torch.cuda.amp) to reduce the amount of model parameters and inference time while reduce part of the accuracy.
4. Simulated the attack behavior of the noise radio on the model prediction in the real complex environment (model attack).

## Results of Expriments 

in32out128, lr1e-2, bs=32, train_epochs=40, GTX3090
| Methods | Structure | MSE | ocpy_acc | false_alarm | missing_alarm | avarage inference time | avarage training time per epoch  | Training maximum memory usage | Params | avarage inference time with amp |
| --- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| MLPMixer | 2 blocks + 1 fc | 0.7032 | 0.6944 | 0.0782 | 0.2273 | 0.691ms  | 17.05s |  2579MiB | 65M | 0.912ms |
| RNN | 2 layers + 1 fc | 0.7634 | 0.6733 | 0.1173 | 0.2094 |  2.424ms | 18.73s |  2529MiB | 65M | 2.117ms |
| LSTM | 2 layers + 1 fc | 0.7787 | 0.6826 | 0.0211 | 0.2963 |  6.566ms | 19.92s |  3595MiB | 257M | 4.425ms |
| Informer | 2 en + 1 dec | 0.7018 | 0.6938 | 0.1220 | 0.1841 |  5.624ms | 31.26s |  2905MiB | 91M | 6.071ms |
| TCN | 2 layers + 1 fc | 0.7098 | 0.6904 | 0.0630 | 0.2466 |  3.344ms | 19.48s |  3221MiB | 209M | 3.453ms |


perturbation
| Methods | Structure |  ocpy_acc | false_alarm | missing_alarm | 
| --- | ----------- | ----------- | ----------- | ----------- | 
| MLPMixer | 2 blocks + 1 fc |  0.6945  | 0.0770 | 0.2284 |
| RNN | 2 layers + 1 fc |  0.6838 | 0.0000 | 0.3162 |
| LSTM | 2 layers + 1 fc |  0.6826 | 0.0203 | 0.2972 |
| Informer | 2 en + 1 dec |  0.6078 | 0.3045 | 0.0876 |
| TCN | 2 layers + 1 fc |  0.6900 | 0.0679 | 0.2421 |


in32out128, lr1e-2, bs=32, train_epochs=40, GTX3090, use_amp
| Methods | Structure | MSE | ocpy_acc | false_alarm | missing_alarm |
| --- | ----------- | ----------- | ----------- | ----------- | ----------- |
| MLPMixer | 2 blocks + 1 fc | 0.7033 | 0.6944 | 0.0784 | 0.2272 |
| RNN | 2 layers + 1 fc | 0.7665 | 0.6721 | 0.1157 | 0.2121 |
| LSTM | 2 layers + 1 fc | 0.8939 | 0.6838 | 0.0000 | 0.3162 |
| Informer | 2 en + 1 dec | 0.6939 | 0.6984 | 0.1075 | 0.1941 |
| TCN | 2 layers + 1 fc | 0.7075 | 0.6920 | 0.0636 | 0.2444 |