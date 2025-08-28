# STiDS-Tree-Cover-Segmentation

This project was done as part of the "Selected Topics in Datascience" seminar(M.Inf.2243) at the University of Göttingen for the summer semester 2025(SoSe2025). The project is about a contrastive learning approach ([SimCLR](https://arxiv.org/abs/2002.05709)) for high resolution tree cover segmentation using both real world images and video game map images, to find out if realistic video game imagery can be used as a subsitution/extension of real-world data.

# General idea

I decided to approach this tree cover segmentation with a contrastive learning deep learning model. I decided on a Encoder-Decoder architecture like U-net in which the encoder learns features from task related unlabeled images and then the decoder is fine-tuned on a labeled high resolution aerial image dataset to return a pixel-level classification for the tree crowns. For the encoder and decoder I used ResNet50 pre-trained on ImageNet due to it's already good performance on image segmentation and classification tasks. The encoder is additionally pre-trained with SimCLR so that the model can learn satellite images better.

<img width="2406" height="1016" alt="grafik" src="https://github.com/user-attachments/assets/9df245e4-dbe0-43e9-8019-07933e00a97c" />

# Setup

To ensure that this project runs smoothly it is recommended that the same environemnt that this project was designed in is used. The following packages and dependencies were used.




Additionally I used Kaggle to run the code as the service offers the needed hardware requirements. Specifically the P100 GPU with 16GB of VRAM was used.


# Training

## Datasets

I used three seperate datasets for this project. 

The first dataset was a labeled high resolution aerial image dataset from around the City of Göttingen offered to me by our supervisors. This dataset was used for fine tuning the image segmentation decoder with a resolution of 256x256 pixels. 

The second dataset was a dataset from the videogame GTA V, also offered to me by our supervisors. This dataset was used for pretraining the SimCLR encoder model.

The third dataset is from the [OAM-TCD](https://arxiv.org/abs/2407.11743) Dataset which gathered diverse high resolution tree segmentation images. These images with a resolution of 2048 x 2048 were cut into smaller patches of 256x256 to match the decoder size, since this dataset is used for the SimCLR encoder model.
\n Since the dataset was very large I decided to only use the first 1848 images that were cut up into 8 x 8 tiles which resulted in 118.272 images for pre-training.

Note:
Data paths might need to be reconfigured since it used the Kaggle folder structure.

<img width="2192" height="666" alt="Dataset comparison" src="https://github.com/user-attachments/assets/d4289a1f-1f50-4d59-9f4d-be53d119ebac" />

## Training config

| Parameter | Value     | Additional Note |
|------|-------------|-------|
| Random Seed  | 42 | pre-set to a value to assist replication |
| Batch-size encoder | 512  | Used gradient accumulation 32size x 8 | 
| Learning rate encoder | 1e-3 | |
| Epoches encoder | 50 | |
| NTXentLoss Temperature | 0.5 | |
| Batch-size decoder  | 16         |  | 
| Epoches decoder | 50 | |
| Learning rate decoder | 1e-4 | |

# Results

## Using only real-world data

<img width="5370" height="1466" alt="training_comparison_20250828_080542" src="https://github.com/user-attachments/assets/63079edf-2265-4dff-ace5-de83967850ce" />
Note: Baseline model is using a ResNet50 Encoder without SimCLR.
