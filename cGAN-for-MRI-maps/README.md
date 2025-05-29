### MRI Image Synthesis with cGAN (U-Net + PatchGAN)

This project implements a conditional Generative Adversarial Network (cGAN) for synthesizing MRI parametric maps from weighted input images using a U-Net generator and PatchGAN discriminator.

#### Project Structure
```
├── model.py            - GAN architecture definitions
├── preprocessing.py    - Data loading and normalization
├── train.py            - Training pipeline
├── utils.py            - Training utilities
└── postprocessing.py   - Model evaluation and metrics
```

#### Key Features
1. **Generator**: U-Net architecture with skip connections
   - Encoder/decoder blocks with batch normalization
   - Transposed convolutions for upsampling
   - Tanh activation for output (-1 to 1 range)

2. **Discriminator**: PatchGAN classifier
   - LeakyReLU activations
   - Sigmoid output for real/fake classification

3. **cGAN Framework**:
   - Combined generator + discriminator
   - Loss: Binary cross-entropy + MAE (weighted 1:100)
   - Adam optimizer (lr=0.0002, β₁=0.5)

#### Usage

1. **Data Preparation**:
   ```python
   # Configure directories in preprocessing.py
   weighted_dir = "/path/to/weighted_images"
   map_dir = "/path/to/ground_truth_maps"
   ```

2. **Training**:
   ```bash
   python train.py
   ```
   - Automatically handles:
     * Data normalization
     * Model checkpointing
     * Progress visualization

3. **Evaluation**:
   ```python
   # In postprocessing.py
   evaluate_generator(trained_generator, weighted_dir, map_dir)
   ```
   - Computes SSIM and PSNR per channel
   - Handles denormalization automatically

#### Dependencies
- TensorFlow 2.x
- scikit-image
- NumPy
- Matplotlib
