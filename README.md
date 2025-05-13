
# Model Training and Evaluation

This repository contains code for training, evaluating, and running inference on a PerVFI and ASVFI.

## File Structure

```
.
├── main.py          # Script for training the neural network
├── inference.py     # Script for generating intermediate frame
├── evaluate.py      # Script for evaluating the model's performance on the metrics
└── README.md        # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- Required Python packages (install via `asvfi.yml`):
  ```bash
  conda env create -f asvfi.yml
  ```

## Training the Model

To train the model, use the `main.py` script. You can specify the optimizer, learning rate, and other training parameters as command-line arguments.

### Example Command:
```bash
python main.py --optimizer=adamw --lr=2e-4
```

### Arguments:
- `--lr`: Learning rate (default: `2e-4`).
- `--batch_size`: Batch size for training (default: `1`).


## Running Inference

To run inference using a trained model, use the `inference.py` script.

### Example Command:
```bash
python inference.py 
```

## Evaluating the Model

To evaluate the model's performance on a test dataset, use the `evaluate.py` script.

### Example Command:
```bash
python evaluate.py 
```

The script will print the evaluation metrics (FloLPIPs, FVD, VFIPS, PSNR, SSIM)
