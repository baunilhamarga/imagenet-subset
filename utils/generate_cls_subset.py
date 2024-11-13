import os
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
from tqdm import tqdm
import csv

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

def generate_metadata(subset_file='./aux_files/imagenet30.txt',
                      mapping_file='./aux_files/meta_clsloc.csv',
                      output_file='./out_files/imagenet30_metadata.csv'):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the subset WNIDs and assign subset labels
    with open(subset_file, 'r') as f:
        subset_wnids = [line.strip() for line in f if line.strip()]
    wnid_to_subset_label = {wnid: idx for idx, wnid in enumerate(subset_wnids)}

    # Read the mapping file
    mapping_data = []
    with open(mapping_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mapping_data.append({
                'WNID': row['WNID'],
                'original_label': int(row['ILSVRC2015_CLSLOC_ID']),
                'name': row['name'],
                'description': row['description'],
                'num_train_images': int(row['num_train_images']),
                'human_label': row['human_label']
            })

    # Filter and include subset labels
    metadata = []
    for wnid in subset_wnids:
        entry = next((item for item in mapping_data if item['WNID'] == wnid), None)
        if entry:
            entry['subset_label'] = wnid_to_subset_label[wnid]
            metadata.append(entry)
        else:
            raise ValueError(f"WNID {wnid} not found in mapping file.")

    # Write the metadata to CSV
    fieldnames = ['subset_label', 'original_label', 'WNID', 'name',
                  'description', 'num_train_images', 'human_label']
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in metadata:
            writer.writerow(data)

    print(f"Metadata CSV file has been generated at '{output_file}'.")

def count_validation_images(val_ground_truth, label_to_wnid, subset_wnids):
    count = 0
    for class_id in val_ground_truth:
        wnid = label_to_wnid.get(class_id)
        if wnid in subset_wnids:
            count += 1
    return count

def process_dataset(split, data_dir, metadata, wnid_to_label, label_to_wnid,
                    val_ground_truth, subset_wnids):
    if split == 'train':
        imageset_file = './aux_files/ImageSets/CLS-LOC/train_cls.txt'
        images_dir = os.path.join(data_dir, 'train')
        total_images = sum(item['num_train_images'] for item in metadata)
    else:
        imageset_file = './aux_files/ImageSets/CLS-LOC/val.txt'
        images_dir = os.path.join(data_dir, 'val')
        # Count the number of validation images in the subset
        total_images = count_validation_images(val_ground_truth, label_to_wnid, subset_wnids)

    # Pre-allocate arrays
    images = np.empty((total_images, 224, 224, 3), dtype=np.uint8)
    labels = np.empty((total_images,), dtype=np.int32)
    ids = np.empty((total_images,), dtype=np.int32)

    idx = 0  # Index for the pre-allocated arrays

    # Read the image list
    with open(imageset_file, 'r') as f:
        image_list = [line.strip().split() for line in f if line.strip()]

    for img_info in tqdm(image_list, desc=f"Processing {split} data"):
        img_rel_path = img_info[0]
        img_id = int(img_info[1])

        if split == 'train':
            wnid, img_name = img_rel_path.split('/')
            if wnid not in subset_wnids:
                continue
            img_file = os.path.join(images_dir, wnid, img_name + '.JPEG')
        else:
            img_name = img_rel_path
            class_id = val_ground_truth[img_id - 1]
            wnid = label_to_wnid.get(class_id)
            if wnid not in subset_wnids:
                continue
            img_file = os.path.join(images_dir, img_name + '.JPEG')

        # Process the image
        try:
            img = Image.open(img_file).convert('RGB')
            img = preprocess(img)
            img = np.array(img, dtype=np.uint8)

            # Assign to arrays
            images[idx] = img
            labels[idx] = wnid_to_label[wnid]
            ids[idx] = img_id
            idx += 1
        except Exception as e:
            raise RuntimeError(f"Error processing image {img_file}: {e}")

    # Trim arrays to actual size
    images = images[:idx]
    labels = labels[:idx]
    ids = ids[:idx]

    return images, labels, ids

def generate_cls_subset(subset_name='imagenet30',
                        val_ground_truth_file='./aux_files/ILSVRC2015_clsloc_validation_ground_truth.txt',
                        data_dir='./Data/CLS-LOC'):
    
    metadata_file = f'./out_files/{subset_name}_metadata.csv'
    # Generate metadata before processing images
    generate_metadata(subset_file=f'./aux_files/{subset_name}.txt',
                      mapping_file='./aux_files/meta_clsloc.csv',
                      output_file=metadata_file)
        # Load metadata
    metadata = []
    with open(metadata_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata.append({
                'subset_label': int(row['subset_label']),
                'original_label': int(row['original_label']),
                'WNID': row['WNID'],
                'num_train_images': int(row['num_train_images']),
            })

    # Create mappings
    wnid_to_label = {item['WNID']: item['subset_label'] for item in metadata}
    label_to_wnid = {item['original_label']: item['WNID'] for item in metadata}
    subset_wnids = set(wnid_to_label.keys())

    # Load validation ground truth
    with open(val_ground_truth_file, 'r') as f:
        val_ground_truth = [int(line.strip()) for line in f if line.strip()]

    # Process datasets
    X_train, y_train, train_id = process_dataset('train', data_dir, metadata,
                                                 wnid_to_label, label_to_wnid,
                                                 val_ground_truth, subset_wnids)
    X_val, y_val, val_id = process_dataset('val', data_dir, metadata,
                                           wnid_to_label, label_to_wnid,
                                           val_ground_truth, subset_wnids)

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

    # Optionally regenerate metadata if needed
    # generate_metadata(subset_file=f'./aux_files/{subset_name}.txt',
    #                   output_file=f'./out_files/{subset_name}_metadata.csv')

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_name', type=str, default='imagenet30', help='Subset name.')
    args = parser.parse_args()
    subset_name = args.subset_name

    # Process datasets
    generate_cls_subset(subset_name=subset_name)
