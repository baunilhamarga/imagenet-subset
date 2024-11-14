# ImageNet Subset

Generate a classification subset from the complete ImageNet dataset.

**Warning:** This process is memory intensive.

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
  - [Running the Script](#running-the-script)
- [NPZ Content](#npz-content)
  - [Resize Method](#resize-method)
- [Metadata Content](#metadata-content)
- [Citations](#citations)
- [License](#license)

## Introduction

This repository provides scripts to generate a classification subset from the complete ImageNet dataset. It allows you to create a custom subset for training or evaluating machine learning models on a smaller, more manageable dataset.

## Quick Start

### Prerequisites

- Python 3.x
- Required Python packages:
  - `numpy`
  - `Pillow`
  - `torchvision`
  - `tqdm`
  - `argparse`
  - `matplotlib` (optional)
  - `pandas` (optional)
- Sufficient disk space and memory (processing ImageNet can be resource-intensive).
- Access to the full ImageNet dataset (requires registration at the [ImageNet website](https://www.image-net.org/)).

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/baunilhamarga/imagenet-subset.git
   cd imagenet-subset
   ```

2. **Install Python dependencies:**

   You can install the required packages using `pip`:

   ```bash
   pip install numpy pillow torchvision argparse tqdm matplotlib pandas
   ```

### Dataset Preparation

1. **Download the ImageNet dataset:**

   - Register and download the complete ImageNet dataset from the [ImageNet website](https://www.image-net.org/).
   - Follow their instructions to obtain the training and validation data.

2. **Organize the Data Folder:**

   Place the `Data` folder inside the root directory of this repository or use the flag `--data_dir` on the [running step](#running-the-script) for other paths. The `Data` folder should have the following structure:

   ```
   .
   ├── Data
   |   └── CLS-LOC
   │       ├── train
   │       │   ├── n01440764
   │       │   │   ├── n01440764_18.JPEG
   │       │   │   └── ...
   │       │   └── ...
   │       └── val
   │           ├── ILSVRC2012_val_00000001.JPEG
   │           └── ...
   ```

   - **Training Images (`train`):** Organized into subfolders named after the WordNet IDs (WNIDs).
   - **Validation Images (`val`):** All images are in a single folder without subdirectories.

3. **Create Your Label Subset:**

   - Inside the `./aux_files` directory, create a text file containing your desired subset of labels (WNIDs), e.g., `imagenet30.txt`.
   - The file should contain one WNID per line. For example:

     ```
     n12267677
     n02690373
     n02701002
     ...
     ```

### Running the Script

Run the main script to generate the subset:

```bash
python main.py --subset_name imagenet30
```

- Replace `imagenet30` with the name of your subset file (without the `.txt` extension) if you are using a different file.
- You can add the flag `--data_dir` to choose a custom Data folder. The path should be to the folder contaning `train` and `val` subfolders, for example:
```bash
python main.py --subset_name imagenet30 --data_dir ../ILSVRC/Data/CLS-LOC/
```
- After processing, the script will generate a `.npz` file containing your ImageNet subset and a `.csv` metadata file inside the `./out_files` directory.

## NPZ Content

The generated NPZ file contains training and validation images and labels stored in NumPy arrays. The images are stored as `uint8` arrays with the shape `(num_images, 224, 224, 3)`. The labels are `int32` integers ranging from `0` to `num_labels - 1`, corresponding to the number of lines in your subset text file. The original unique image IDs from ILSVRC15 are also included.

**NPZ Keys:**

- `'X_train'`: Training images subset.
- `'y_train'`: Training labels subset.
- `'train_id'`: Training unique integer IDs.*
- `'X_val'`: Validation images subset.
- `'y_val'`: Validation labels subset.
- `'val_id'`: Validation unique integer IDs.*

\* You can use these IDs to track the exact JPEG image files from the original ImageNet dataset.

### Resize Method

To achieve images with consistent dimensions of 224x224 pixels:

- The shortest dimension of each image is rescaled to 256 pixels while maintaining the aspect ratio.
- A central crop of size 224x224 pixels is then extracted from the resized image.

## Metadata Content

A CSV file is generated containing metadata for your subset. This file includes the mapping between your subset labels and the original ImageNet labels.

Sample content of `imagenet30_cls_metadata.csv` for `imagenet30`:

```
subset_label,original_id,WNID,human_label
0,327,n12267677,acorn
1,230,n02690373,airliner
2,265,n02701002,ambulance
...
29,362,n09472597,volcano
```

- **subset_label**: Label index in your subset (from `0` to `num_labels - 1`).
- **original_id**: Original label ID from ImageNet.
- **WNID**: WordNet ID.
- **human_label**: Human-readable label.

## Citations

**ImageNet Original:**

```bibtex
@article{ILSVRC15,
  Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
  Title = {{ImageNet Large Scale Visual Recognition Challenge}},
  Year = {2015},
  journal   = {International Journal of Computer Vision (IJCV)},
  doi = {10.1007/s11263-015-0816-y},
  volume={115},
  number={3},
  pages={211-252}
}
```

**ImageNet30 Subset:**

```bibtex
@inproceedings{hendrycks2019selfsupervised,
  title={Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty},
  author={Dan Hendrycks and Mantas Mazeika and Saurav Kadavath and Dawn Song},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

**ImageNet100 Subset:**

```bibtex
@article{JMLR:v23:21-1155,
  author  = {Victor Guilherme Turrisi da Costa and Enrico Fini and Moin Nabi and Nicu Sebe and Elisa Ricci},
  title   = {solo-learn: A Library of Self-supervised Methods for Visual Representation Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {56},
  pages   = {1--6},
  url     = {http://jmlr.org/papers/v23/21-1155.html}
}
```

## License

[MIT License](LICENSE)

*Please ensure you comply with the ImageNet terms of use when distributing any part of the dataset.*
