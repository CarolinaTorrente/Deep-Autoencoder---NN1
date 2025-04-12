# Deep-Autoencoder---NN1

The goal is to implement a deep autoencoder based on dense neural networks over MNIST and FMNIST databases:
  - Validate a couple of network architectures:
    - 3 layers at both encoder/decoder
    - 5 layers at both encoder/decoder
  - Study the influence of the projected dimension:
    - Dimensions to test: 15, 30, 50, 100
  - Implement regularization techniques:
    - Consider regularizing the encoder's output using Lasso
  - For the best architecture found:
    - Implement a denoising autoencoder
    - Inject additive zero-mean Gaussian noise with tunable variance
    - Analyze performance as a function of the noise variance
  - Use the following performance metrics:
    - Peak Signal-to-Noise Ratio (PSNR)
    - Visualization of reconstructed/denoised signals versus encoder input


# Deep Autoencoder for MNIST and Fashion MNIST

## 1. Network Validation and Latent Dimension Study

- **Objective**: Validate networks with different architectures (3 vs. 5 layers in encoder/decoder) and study the influence of the latent dimension (15, 30, 50, 100).
- **Dataset**: 20,000 random samples from the original dataset.
- **Configurations Compared**:
  - Latent dimensions tested: 15, 30, 50, 100
  - Larger latent dimensions improve feature representation but increase complexity.
- **Training Details**:
  - **Loss Function**: Mean Squared Error (MSE)
  - **Optimizer**: Adam
  - **Batch Size**: 32
  - **Epochs**: Increased from 20 to 50 based on convergence behavior
- **Training Observations**:
  - Validation loss decreased logarithmically early on.
  - Increasing epochs yielded marginal improvement with significant cost.

### I. 3-Layer Architecture
- **Best Latent Dimension**: 100
- **Validation Loss**: 0.0079
- Larger latent dimensions better capture complex data patterns, but overfitting is a risk.

### II. 5-Layer Architecture
- **Best Latent Dimension**: 100
- **Validation Loss**: 0.0174
- Best performance among tested dimensions for 5-layer configuration.

## 2. Regularization Techniques

- **Goal**: Prevent overfitting observed in previous experiments.
- **Techniques Used**:
  - L1 (Lasso) and L2 (Ridge) regularization
  - Dropout, Batch Normalization, Layer Normalization

### Hyperparameter Search Strategy

- **Initial Search**:
  - Function: `initial_hyperparameter_search()`
  - Parameters: `encoding_dim`, `lasso_lambda`, `ridge_lambda`, `batch_size`
  - Dropout, batch norm, and layer norm disabled
- **Second Search (Regularization Search)**:
  - Function: `regularization_search()`
  - Search over `dropout_rate`, `batch_norm`, and `layer_norm`
  - Found dropout unnecessary (confirmed by Raschka’s quote)

### Best Hyperparameters (See Table 2)
- `'encoding_dim': 50`
- `'lasso_lambda': 0.0001`
- `'ridge_lambda': 0.0001`
- `'dropout_rate': 0.0`
- `'use_batch_norm': False`
- `'use_layer_norm': True`
- `'batch_size': 32`
- **Best Validation Loss**: 0.0640

### Additional Notes
- **Training**: Early stopping with patience parameter (5 epochs)
- **Conclusion**:
  - Layer normalization preferred over batch norm
  - No dropout needed for MNIST
  - Latent dimension of 50 balances compression and feature retention

## 3. Denoising Autoencoder Implementation

- **Noise Injection**: Additive zero-mean Gaussian noise with tunable variance.
- **Training**:
  - Input: Noisy images
  - Target: Clean images
  - Loss function: MSE
- **Performance Analysis**:
  - Evaluate performance across different noise variances
  - **Best Validation Loss**: 0.0145 at noise variance = 0.1 (See Graph 1)

## 4. PSNR Metric and Visualization

- **Metric Used**: Peak Signal-to-Noise Ratio (PSNR)
- **During Training/Validation**:
  - PSNR calculated per batch and averaged
- **Visualization**:
  - `visualize_reconstructed_images` shows input, clean, and reconstructed images
- **Noise Variance Analysis**:
  - `analyze_noise_variance_and_visualize` shows performance across variances
  - See Graph 2 for PSNR trends and image samples

---

# Fashion MNIST Results

## Part 1 (See Graph 3)
- **3-Layer Autoencoder**:
  - `encoding_dim = 50`
  - **Validation Loss**: 0.0122
- **5-Layer Autoencoder**:
  - `encoding_dim = 100`
  - **Validation Loss**: 0.0183
- Conclusion: Shallower network generalizes better for Fashion MNIST.

## Part 2 (See Graph 4)
- **Optimal Hyperparameters**:
  - `encoding_dim = 100`
  - Batch norm and layer norm enabled
  - `dropout = 0.2`
  - `batch_size = 128`
- Best Validation Loss: 0.1251

## Part 3 (See Graph 5)
- **Best Noise Variance**: 0.05 and 0.1
- **Validation Loss**: 0.0155
- Small noise acts as regularization and improves robustness.

## Part 4 (See Graphs 6 and 7)
- **Noise Variance = 0.05**:
  - Validation Loss: 0.0152
  - PSNR: 17.64
- **Noise Variance = 0.1**:
  - Validation Loss: 0.0155
  - PSNR: 17.61
- Gradual loss decrease and PSNR improvement confirm effective denoising.

---

# Final Conclusion

- **Architecture & Dataset Complexity**:
  - MNIST: deeper architectures, larger latent dimensions work well
  - Fashion MNIST: simpler models perform better
- **Regularization**:
  - Layer normalization and L1/L2 regularization effective
  - Dropout unnecessary
- **Denoising Autoencoder**:
  - Noise injection enhances robustness
  - Optimal noise levels key for balancing reconstruction quality
- **Best MNIST Model**:
  - **Validation Loss**: 0.0640
  - Combines simple architecture, strong regularization, and effective noise injection

