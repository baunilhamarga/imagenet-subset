import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms
import argparse
from tqdm import tqdm
import csv

def generate_metadata(subset_file = './aux_files/imagenet30.txt', mapping_file = '../devkit/data/map_clsloc.txt', output_file = './out_files/metadata.csv'):
    """
    Generates a metadata CSV file for a subset of ImageNet classes.

    This function reads a subset file containing WordNet IDs (WNIDs) and a mapping file
    containing WNIDs, original labels, and human-readable labels. It then creates a metadata
    CSV file that includes the subset label, original label, WNID, and human-readable label
    for each class in the subset.

    Args:
        subset_file (str): Path to the file containing the subset WNIDs. Default is './aux_files/imagenet30.txt'.
        mapping_file (str): Path to the mapping file containing WNIDs, original labels, and human-readable labels. Default is '../devkit/data/map_clsloc.txt'.
        output_file (str): Path to the output CSV file where the metadata will be saved. Default is './out_files/metadata.csv'.

    Raises:
        FileNotFoundError: If the subset file or mapping file does not exist.
        IOError: If there is an error reading the files or writing the output file.

    Example:
        generate_metadata(
            subset_file='./aux_files/imagenet30.txt',
            mapping_file='../devkit/data/map_clsloc.txt',
            output_file='./out_files/metadata.csv'
        )
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the subset WNIDs and assign subset labels (0 to 29)
    with open(subset_file, 'r') as f:
        subset_wnids = [line.strip() for line in f if line.strip()]

    # Create a mapping from WNID to subset_label
    wnid_to_subset_label = {wnid: idx for idx, wnid in enumerate(subset_wnids)}

    # Read the mapping file and create a list of dictionaries
    mapping_data = []
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 2)
            if len(parts) < 3:
                continue  # Skip lines that don't have all three parts
            wnid, original_label, human_label = parts
            mapping_data.append({
                'WNID': wnid,
                'original_label': original_label,
                'human_label': human_label
            })

    # Filter the mapping data for WNIDs in our subset and add subset_label
    metadata = []
    for wnid in subset_wnids:
        # Find the corresponding entry in mapping_data
        entry = next((item for item in mapping_data if item['WNID'] == wnid), None)
        if entry:
            metadata.append({
                'subset_label': wnid_to_subset_label[wnid],
                'original_label': entry['original_label'],
                'WNID': wnid,
                'human_label': entry['human_label']
            })
        else:
            print(f"Warning: WNID {wnid} not found in mapping file.")

    # Write the metadata to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['subset_label', 'original_label', 'WNID', 'human_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in metadata:
            writer.writerow(data)

    print(f"Metadata CSV file has been generated at '{output_file}'.")

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

def generate_cls_subset(subset_name = 'imagenet30'):
    # Function to process a dataset (train or val)
    def process_dataset(split):
        images = []
        labels = []
        ids = []

        if split == 'train':
            imageset_file = '../ImageSets/CLS-LOC/train_cls.txt'
            annotations_dir = '../Annotations/CLS-LOC/train'
            images_dir = '../Data/CLS-LOC/train'
        else:
            imageset_file = '../ImageSets/CLS-LOC/val.txt'
            annotations_dir = '../Annotations/CLS-LOC/val'
            images_dir = '../Data/CLS-LOC/val'

        # Read the image list
        with open(imageset_file, 'r') as f:
            image_list = [line.strip().split() for line in f if line.strip()]

        for img_info in tqdm(image_list, desc=f"Processing {split} data"):
            img_rel_path = img_info[0]  # e.g., 'n01440764/n01440764_10026' or 'ILSVRC2012_val_00000001'
            img_id = int(img_info[1])   # ID according to the file

            # Determine the image filename and annotation path
            if split == 'train':
                wnid, img_name = img_rel_path.split('/')
                img_file = os.path.join(images_dir, wnid, img_name + '.JPEG')
            else:
                img_name = img_rel_path
                img_file = os.path.join(images_dir, img_name + '.JPEG')
                annotation_file = os.path.join(annotations_dir, img_name + '.xml')
                # Parse the annotation XML to get the label (WNID)
                try:
                    tree = ET.parse(annotation_file)
                    root = tree.getroot()
                    wnid = root.find('object').find('name').text
                except Exception as e:
                    raise RuntimeError(f"Error parsing XML for {img_file}: {e}")

            # Check if the WNID is in our subset
            if wnid in subset_wnids:
                # Open and preprocess the image
                try:
                    img = Image.open(img_file).convert('RGB')
                    img = preprocess(img)
                    img = np.array(img).astype(np.uint8)

                    # Append to lists
                    images.append(img)
                    labels.append(wnid_to_label[wnid])
                    ids.append(img_id)
                except Exception as e:
                    print(f"Error processing image {img_file}: {e}")
                    continue

        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels, dtype=np.int32)
        ids = np.array(ids, dtype=np.int32)

        return images, labels, ids

    subset_file_path = os.path.join('./aux_files/', subset_name + '.txt')
    
    # Load the subset WNIDs
    with open(subset_file_path, 'r') as f:
        subset_wnids = [line.strip() for line in f if line.strip()]

    # Create a mapping from WNID to label index (0 to N-1)
    wnid_to_label = {wnid: idx for idx, wnid in enumerate(subset_wnids)}

    # Process the train and validation datasets
    X_train, y_train, train_id = process_dataset('train')
    X_val, y_val, val_id = process_dataset('val')

    # Save to NPZ file
    output_path = f'./out_files/{subset_name}_cls.npz'
    np.savez(output_path,
            X_train=X_train,
            y_train=y_train,
            train_id=train_id,
            X_val=X_val,
            y_val=y_val,
            val_id=val_id)

    print(f"Processing complete. Data saved to '{output_path}'.")
    
    generate_metadata(subset_file=subset_file_path, output_file=f'./out_files/{subset_name}_metadata.csv')

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_name', type=str, default='imagenet30', help='Subset name.')
    
    args = parser.parse_args()
    subset_name = args.subset_name
    
    generate_cls_subset(subset_name)