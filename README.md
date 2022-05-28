# Spectrum Prediction

Contributions:
1. Implemented multiple models for spectrum prediction ('TCN', 'informer', 'MLPMixer', 'rnn', 'lstm').
2. Calculated the params and latency of each model (params, latency).
3. Use automatic mixed precision training (torch.cuda.amp) to reduce the amount of model parameters and inference time while reduce part of the accuracy.
4. Simulated the attack behavior of the noise radio on the model prediction in the real complex environment (model attack).