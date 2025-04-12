# Deep-Autoencoder---NN1

The goal is to implement a deep autoencoder based on dense neural networks over MNIST and FMNIST databases:
Validate a couple of networks (3 layers at both encoder/decoder vs 5 layers) and study the influence of the projected dimension (15,30,50,100). 
Implement regularization techniques. Also, regularizing the encoder's output with a Lasso can be a good idea. 
For the best architecture found, implement a denoising autoencoder. Use additive zero-mean Gaussian noise of a given (tunable) variance as injected noise. Analyze the performance as a function of the variance. 
Use the Peak signal-to-noise ratio (PSNR) as a performance metric and, of course, visualization of the reconstructed/denoised signals versus the input to the encoder.
