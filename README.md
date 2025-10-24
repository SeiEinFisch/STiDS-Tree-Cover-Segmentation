# STiDS-Tree-Cover-Segmentation

This project was done as part of the "Selected Topics in Datascience" seminar (M.Inf.2243) at the University of Göttingen for the summer semester 2025 (SoSe2025). The project is about a contrastive learning approach ([SimCLR](https://arxiv.org/abs/2002.05709)) for high-resolution tree cover segmentation using both real-world images and video game map images, to find out if realistic video game imagery can be used as a substitution/extension of real-world data.

# General idea

I decided to approach this tree cover segmentation with a contrastive learning deep learning model. I chose an Encoder-Decoder architecture like U-Net in which the encoder learns features from task-related unlabeled images. Then, the decoder is fine-tuned on a labeled high-resolution aerial image dataset to return a pixel-level classification for the tree crowns. For the encoder and decoder, I used ResNet50 pre-trained on ImageNet due to its already good performance on image segmentation and classification tasks. The encoder is additionally pre-trained with SimCLR so that the model can learn satellite images better.

<img width="2406" height="1016" alt="grafik" src="https://github.com/user-attachments/assets/9df245e4-dbe0-43e9-8019-07933e00a97c" />

# Setup

To ensure that this project runs smoothly, it is recommended to use the same environment that this project was designed in. The following packages and dependencies were used.

| Module/package | Version |
| ---------------|---------|
| Python        | 3.11.13 |
| pandas        | 2.2.3   |
| numpy         | 1.26.4  |
| torch         | 2.6.0+cu124 |
| scikit-learn  | 1.2.2   |
| matplotlib    | 3.7.2   |
| transformers  | 4.52.4  |

Additionally, I used Kaggle to run the code as the service offers the needed hardware requirements. Specifically, the P100 GPU with 16GB of VRAM was needed.

## Folder Structure

```
STiDS-Tree-Cover-Segmentation/
├── LICENSE                              # Apache License 2.0
├── README.md                           # Project documentation
├── config.py                          # Configuration settings and paths
├── main.py                           # Main training script
├── main.ipynb                        # Jupyter notebook for training (includes a full training loop that can be used)
├── simclr-resnet50-unet-original.ipynb  # Original Jupyter notebook containing all code. Was split up for better overview
├── tiling.ipynb                      # Notebook for preprocessing large images into tiles
│
├── data/                             # Data handling modules
│   ├── __init__.py
│   ├── dataloaders.py                # DataLoader creation functions
│   ├── datasets.py                   # Custom Dataset classes
│   └── transforms.py                 # SimCLR augmentations
│
├── models/                           # Model architectures
│   ├── __init__.py
│   ├── baseline_model.py             # Baseline ResNet50 without SimCLR
│   ├── decoders.py                   # U-Net decoder implementation
│   ├── encoders.py                   # ResNet50 encoder variants
│   ├── segmentation.py               # Complete segmentation models
│   └── simclr.py                    # SimCLR model and projection head
│
├── training/                        # Training and evaluation scripts
│   ├── __init__.py
│   ├── baseline_trainer.py          # Training functions for baseline model
│   ├── segmentation_trainer.py      # Training functions for SimCLR model
│   └── simclr_trainer.py            # SimCLR contrastive learning trainer
│
└── utils/                         # Utility functions
    ├── __init__.py
    ├── io_utils.py                 # Data saving utilities
    └── visualization.py            # Plotting and visualization functions
```

# Training

## Datasets

I used three separate datasets for this project:

The first dataset was a labeled high-resolution aerial image dataset from around the City of Göttingen offered by our supervisors. This dataset was used for fine-tuning the SimCLR encoder with images of resolution 256x256 pixels. (1000 images)

The second dataset was from the video game GTA V, also provided by our supervisors. This dataset was used for pretraining the SimCLR encoder model. (385 images)

The third dataset is from the [OAM-TCD](https://arxiv.org/abs/2407.11743) dataset, which contains diverse high-resolution tree segmentation images. These images with a resolution of 2048 x 2048 were cut into smaller patches of 256x256 to match the decoder size, since this dataset is used for the fine-tuned decoder model.  
Since the dataset was very large, I decided to only use the first 16,176 tiles. This amounts to only the first 252 images, but this was done to reduce computing time, as it already takes a long time with this many images.

**Note:** Data paths might need to be reconfigured since the project uses the Kaggle folder structure.

<img width="2192" height="666" alt="Dataset comparison" src="https://github.com/user-attachments/assets/d4289a1f-1f50-4d59-9f4d-be53d119ebac" />

## Training config

| Parameter             | Value      | Additional Note                  |
|----------------------|------------|--------------------------------|
| Random Seed          | 42         | Pre-set to a value to assist replication |
| Batch-size encoder   | 512        | Used gradient accumulation: 32 size × 8 |
| Learning rate encoder | 1e-3       |                                |
| Epochs encoder       | 50         |                                |
| NTXentLoss Temperature | 0.5      |                                |
| Batch-size decoder   | 16         |                                |
| Epochs decoder       | 50         |                                |
| Learning rate decoder | 1e-4       |                                |

# Results

## Using only real-world data

<img width="5370" height="1466" alt="training_comparison_20250828_080542" src="https://github.com/user-attachments/assets/63079edf-2265-4dff-ace5-de83967850ce" />

**Note:** Baseline model uses a ResNet50 Encoder without SimCLR.

## Including video game data

<img width="5370" height="1466" alt="training_comparison_20250828_120114" src="https://github.com/user-attachments/assets/299f9395-0527-4b56-8781-3ff2e68d7780" />

| Metric     | Baseline | SimCLR real-world only | SimCLR with video game data |
|------------|----------|-----------------------|-----------------------------|
| IoU        | **0.886** | 0.878                 | 0.796                       |
| Dice       | 0.404    | 0.396                 | **0.538**                   |
| Pixel acc  | 95%      | 94%                   | 93%                         |

## Conclusion

The evaluation performance of the baseline ResNet50 model is usually slightly better. This might be due to overfitting. At least in this experiment with the data we used, the video game dataset performs worse except in Dice score.

Further conclusions were presented in the seminar.

# AI-Usage card

<img width="1527" height="1916" alt="ai-usage-card(1)" src="https://github.com/user-attachments/assets/2f62e2e5-7499-4568-88e8-6616765bf2f9" />
