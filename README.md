# README

# Dual-Input Action Recognition Framework

This project implements a **dual-input action recognition model** that fuses spatiotemporal video features with keypoint-based representations for robust human action classification. The framework integrates an R(2+1)D convolutional video branch with a key point branch, enhanced by velocity and positional encoding to capture temporal dynamics.

## Features

- Dual-input architecture: combines RGB video frames and skeletal keypoints
- Video backbone: R(2+1)D pretrained on Kinetics-400
- Keypoint branch supports velocity augmentation and positional encoding
- Achieves up to **94.52% validation accuracy** on [Workout/Exercise Video](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)
- Includes qualitative visualization pipeline with skeleton overlay and predicted labels
- Supports evaluation on custom datasets with flexible input formats

## Architecture Overview

- `img_branch`: extracts spatiotemporal features from input video
- `kp_branch`: extracts motion-aware keypoint embeddings
- Intermediate fusion: concatenates video and keypoint features
- Classification head: predicts action class based on fused representation

## 📂 Project Structure

While we do not know your specific settings, the following tree structure of the file system provides you with a general layout of what the structure of the project should be like.

```bash
dual-input-action-recognition/
├── raw_data/         # Where you store the raw data
│   └── split_data.py  # Script to split raw data into train, val, and test sets
├── checkpoints/         # Where you save model checkpoints
│   └── best_model.pth
├── processed_data/.     # Where the pre-processed data is stored
│   ├── workoutfitness-train/
│				├── ...
│		    ├── train_list.txt/
│   ├── workoutfitness-val/
│				├── ...
│		    ├── val_list.txt/
│   ├── workoutfitness-test/
│				├── ...
│		    ├── test.txt/
├── timesformer/         # Where you save model checkpoints
│   └── train_timesformer.py         # Training script for TimeSformer
│   └── checkpoints/     # Where TimeSformer checkpoints are saved
├── test_results/         # Inference results and visualization
├── dual_input_r2plus1d.py # The R2Plus1D+Velocity model
├── dual_input_r2plus1d_positional.py # The R2Plus1D+Velocity+Positional model
├── dual_input_dataset.py # The Dataset class
├── train_dual_input.py.  # Training script
├── extract_frames_and_keypoints_aug.py.  # Preprocess data
├── predict_and_visualize.py # Visualization script
├── requirements.txt
├── additional/         # Additional/future work done and other models
│   └── ...
└── README.md
```

## Set Up Environment

We assume you have set up a GCP Compute Engine instance with at least one Nvidia T4 GPU, 2 vCPUs, and 15GB RAM.

<aside>
💡

You may need to configure the paths inside all the Python scripts

</aside>

1. Create a conda environment:
    
    ```bash
    conda create -n dualinput python=3.10
    conda activate dualinput
    ```
    
2. Install dependencies, we have included requirements.txt in the submission:
    
    ```bash
    pip install -r requirements.txt
    ```
    

---

## Dataset Preparation

1. Download kaggle dataset
    1. link: [https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)
    2. Put the dataset into raw_data folder
2. Split data randomly into training, validation, and test sets by running the following command, 
    
    ```bash
    python split_data.py
    ```
    
3. Preprocess the videos into frames and extract estimated key points from them by running
    
    <aside>
    💡
    
    You may need to pass in arguments to run this command. You also need to run the script 3 times if you want to process data for all training, validation, and testing.
    
    1. —input_dir: The source path for data you want to process
    2. —out_dir: The target directory you want the processed data to be saved to
    3. —list_name: The meta information about the processed data 
    </aside>
    
    ```bash
    python extract_frames_and_keypoints_aug.py --input_dir="PATH TO TRAIN DATA" --out_dir="PATH TO DST DIR" --list_name="TXT FILE NAME FOR DATA"
    ```
    

---

## R2Plus1D Model Training

1. Update [`train_dual_input.py`](http://train.py) with desired model and checkpoint destination before running training script, then run
    
    ```bash
    python train_dual_input.py
    ```
    
2. Predict and visualize results, before you do it, you must move videos you want to visualize inside test folder
    
    ```bash
    mv "PATH TO TARGET VIDEO" "dual-input-action-recognition/test"
    python predict_and_visualize.py
    ```
    

---

## TimesFormer FineTuning

1. Update [train_timesformer.py](http://train.py) with desired TimesFormer series model and checkpoint destination before running training script, or you can choose to experiment with different pretrained model.
2. You can directly get the training info and evaluation results on both validation and test set by running
    
    ```bash
    python timesformer/train_timesformer.py
    ```
    

---

# Contribution

Mingyu Zhu (mz3062) implemented the following files:

1. split_data.py
2. dual_input_r2plus1d.py
3. dual_input_r2plus1d_positional.py
4. extract_frames_and_keypoints_aug.py
5. dual_input_dataset.py
6. train_dual_input.py
7. predict_and_visualize.py

Dieter Joubert (dj2574) implemented the following files:

1. additional/dual_input_timesformer.py
2. additional/dual_input_vit_improved.py
3. additional/dual_input_vit.py
4. additional/train_dual_input_oom.py

Wangshu Zhu (wz2708) implemented the followin files:

1. timesformer/train_timesformer.py

<aside>
💡

All members contributed equally to the paper and presentation.

</aside>

---

# Attribution

This project builds upon existing open-source implementations:

- **R(2+1)D Model:**
    
    We use the official implementation of R(2+1)D from [torchvision.models.video.r2plus1d_18](https://pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html) (Apache 2.0 License).
    
- **MediaPipe:**
    
    We use the [MediaPipe](https://mediapipe.dev/) Python API to extract human pose keypoints (Apache 2.0 License).
    
- **TimeSformer:**
    
    We use the official TimeSformer model from [facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer) (MIT License) for transformer-based video classification baseline.
    

We thank the original authors and contributors of these projects for making their implementations publicly available.

Please refer to their respective repositories and licenses for more details.