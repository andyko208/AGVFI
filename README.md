
# Model Training and Evaluation

## Overview

This project implements and provides tools for training and evaluating Audio-Separated Video Frame Interpolation (ASVFI) models, specifically PerVFI and ASVFI. These models are designed to generate intermediate video frames with guidance from audio features, aiming to produce smooth and realistic video transitions. This repository contains the necessary scripts for model training, running inference to generate frames, and evaluating model performance using various metrics.

## Table of Contents
- [File Structure](#file-structure)
- [Prerequisites](#prerequisites)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)
- [Evaluating the Model](#evaluating-the-model)

## File Structure

```
.
├── main.py            # Main script for training the neural network.
├── inference.py       # Script for generating intermediate frames using a trained model.
├── evaluate.py        # Script for evaluating the model's performance on various metrics.
├── model.py           # Defines the core ASVFI model architecture, including sub-networks and attention mechanisms.
├── UNet.py            # Contains the implementation of 3D U-Net structures used in the model.
├── resnet_3D.py       # Implements 3D ResNet components, likely used as part of the video encoder.
├── dataset.py         # Handles dataset loading and preprocessing (e.g., UCF dataset).
├── utils.py           # Contains utility functions for tasks like video/audio processing (ffmpeg),
│                      # data preparation (CSV creation, train/test splits), and file operations.
│                      # Note: Some scripts here might be for one-time data setup.
├── tools.py           # Provides helper tools, e.g., calculating model parameters.
├── train.py           # Another script for training. (Note: This might be an alternative or older training
│                      # script. `main.py` is the primary one mentioned for training).
├── asvfi.yaml         # Conda environment file for setting up dependencies.
├── LICENSE            # License file for the project.
└── README.md          # Project documentation (this file).
```

## Prerequisites

- Python 3.8 or higher
- **Key Python Packages:**
  - PyTorch (torch)
  - TorchVision (torchvision)
  - TorchAudio (torchaudio)
  - NumPy
  - OpenCV (cv2)
  - Matplotlib
  - Pillow (PIL)
  - Pandas
- **Installation via Conda:**
  All required Python packages can be installed using the provided `asvfi.yml` file:
  ```bash
  conda env create -f asvfi.yml
  ```

## Training the Model

To train the ASVFI model, use the `main.py` script. This script allows for configuration of various training parameters, including optimizer settings and learning rates.

### Example Training Command
```bash
python main.py --optimizer adamw --lr 6e-4
```

### Training Arguments
Below is a list of command-line arguments you can use to configure the training process:
- `--start_epoch INT`: Specifies the epoch from which to start or resume training. Default: `0`.
- `--lr FLOAT`: Sets the learning rate for the optimizer. Default: `6e-4`.
- `--optimizer STR`: Defines the type of optimizer to use (e.g., 'adamw', 'adam'). Default: `'adamw'`.
- `--loss STR`: Configures the loss function. Default: `'1*L1'`.
- `--beta1 FLOAT`: Sets the beta1 parameter for Adam-based optimizers. Default: `0.9`.
- `--beta2 FLOAT`: Sets the beta2 parameter for Adam-based optimizers. Default: `0.99`.
- `--random_seed INT`: Provides a seed for random number generation, ensuring reproducibility. Default: `12345`.
- **Note on Batch Size:** The `batch_size` is internally set to `1` within the data loaders.

### Model Checkpoints and Outputs
- **Training Checkpoints:** Periodically saved to the `weights/asvfi/` directory. Checkpoint filenames include the optimizer type and learning rate (e.g., `weights/asvfi/adamw_6e-4.pth`).
- **Best Model:** The model checkpoint that achieves the best performance on the validation set is saved as `weights/asvfi/best_asvfi.pth`.
- **Loss Curves:** Plots illustrating training and validation loss are saved in the `figs/` directory, named according to the optimizer and learning rate.

## Running Inference

After training, the `inference.py` script can be used to generate intermediate video frames using a saved model checkpoint.

### Example Inference Command
```bash
python inference.py --optimizer adamw --lr 6e-4 --scale 2
```

### Inference Arguments
- `--scale INT` or `-s INT`: Defines the number of intermediate frames to generate between each pair of input frames. For example, `--scale 2` generates one intermediate frame. Default: `2`.
- `--optimizer STR`: Specifies the optimizer used for the trained model (e.g., 'adamw'). This helps in locating the correct model weights. Default: `'adam'`.
- `--lr FLOAT`: Indicates the learning rate of the trained model. This is also used to identify the model weights. Default: `0.0002`.
- `--epoch INT`: Specifies a particular epoch of the model to use for inference. Default: `0`. (Note: The current script primarily uses the optimizer and learning rate to construct the weight path; this argument might be for specific model versions or future enhancements.)

### Specifying the Trained Model for Inference
The inference script loads model weights from a path typically constructed as `weights/asvfi/<optimizer>_<lr>.pth` (e.g., `weights/asvfi/adamw_6e-4.pth`). Ensure that the `--optimizer` and `--lr` arguments correctly correspond to the desired trained model.

### Inference Output
- The script processes videos from the test set, as defined in `../data/UCF-101_split.csv`.
- Generated intermediate frames are saved in a structured directory format: `../data/ASVFI/scale/<scale>_<optimizer>_<lr>/<category>/<video_name>/`.
- These individual frames are then compiled into MP4 video files within their respective directories.

## Evaluating the Model

The `evaluate.py` script is used to assess the performance of a trained model on a test dataset, using various standard video quality metrics.

### Example Evaluation Command
```bash
python evaluate.py --optimizer adamw --lr 6e-4 --scale 2
```

### Evaluation Arguments
- `--scale INT` or `-s INT`: The scale factor used during inference, indicating the set of generated frames to be evaluated. Default: `2`.
- `--optimizer STR`: Specifies the optimizer of the trained model being evaluated. Default: `'adam'`.
- `--lr FLOAT`: The learning rate of the trained model under evaluation. Default: `0.0002`.
- `--epoch INT`: Designates the specific epoch of the model to evaluate. Default: `0`. (Note: Similar to inference, this helps in identifying the model and corresponding data.)

### Specifying Data for Evaluation
The evaluation script expects the generated frames/videos to be located in `../data/ASVFI/scale/<scale>_<optimizer>_<lr>/`. Ensure these arguments correctly point to the dataset produced by `inference.py`. Ground truth data for comparison is sourced from `../data/GT/scale/2` and `../data/UCF-101_imgs`.

### Evaluation Output
- **Console Output:** Key evaluation metrics (FloLPIPs, FVD, VFIPS, PSNR, SSIM) are printed directly to the console.
- **CSV Report:** A detailed report containing these metrics for each evaluated video is saved to `../ASVFI/evals/ASVFI_<optimizer>_<lr>.csv`.
