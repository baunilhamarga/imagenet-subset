import os
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
from tqdm import tqdm
import csv
import math
import zipfile
import numpy.lib.format

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

def generate_metadata(subset_file, mapping_file, output_file, wnid_to_label=None):
    """
    Generates a metadata CSV file for a subset of ImageNet classes.

    Args:
        subset_file (str): Path to the file containing the subset WNIDs.
        mapping_file (str): Path to the mapping file containing WNIDs and other metadata.
        output_file (str): Path to the output CSV file where the metadata will be saved.
        wnid_to_label (dict): Mapping from WNID to global label index.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the subset WNIDs
    with open(subset_file, 'r') as f:
        subset_wnids = [line.strip() for line in f if line.strip()]
    # Use the provided wnid_to_label mapping
    if wnid_to_label is None:
        wnid_to_subset_label = {wnid: idx for idx, wnid in enumerate(subset_wnids)}
    else:
        wnid_to_subset_label = wnid_to_label

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

def count_validation_images(val_ground_truth, label_to_wnid, current_subset_wnids):
    count = 0
    for class_id in val_ground_truth:
        wnid = label_to_wnid.get(class_id)
        if wnid in current_subset_wnids:
            count += 1
    return count

def process_dataset(split, data_dir, current_metadata, wnid_to_label, label_to_wnid,
                    val_ground_truth, current_subset_wnids):
    if split == 'train':
        imageset_file = './aux_files/ImageSets/CLS-LOC/train_cls.txt'
        images_dir = os.path.join(data_dir, 'train')
        total_images = sum(item['num_train_images'] for item in current_metadata)
    else:
        imageset_file = './aux_files/ImageSets/CLS-LOC/val.txt'
        images_dir = os.path.join(data_dir, 'val')
        # Count the number of validation images in the subset
        total_images = count_validation_images(val_ground_truth, label_to_wnid, current_subset_wnids)

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
            if wnid not in current_subset_wnids:
                continue
            img_file = os.path.join(images_dir, wnid, img_name + '.JPEG')
        else:
            img_name = img_rel_path
            class_id = val_ground_truth[img_id - 1]
            wnid = label_to_wnid.get(class_id)
            if wnid not in current_subset_wnids:
                continue
            img_file = os.path.join(images_dir, img_name + '.JPEG')

        # Process the image
        try:
            img = Image.open(img_file).convert('RGB')
            img = preprocess(img)
            img = np.array(img, dtype=np.uint8)

            # Assign to arrays
            images[idx] = img
            labels[idx] = wnid_to_label[wnid]  # Global label index
            ids[idx] = img_id
            idx += 1
        except Exception as e:
            raise RuntimeError(f"Error processing image {img_file}: {e}")

    # Trim arrays to actual size
    images = images[:idx]
    labels = labels[:idx]
    ids = ids[:idx]

    return images, labels, ids

def generate_cls_subset_part(subset_name, part_num, total_parts, wnids, global_metadata,
                             wnid_to_label, label_to_wnid, val_ground_truth, data_dir):
    current_subset_wnids = set(wnids)

    # Filter metadata for current part
    current_metadata = [item for item in global_metadata if item['WNID'] in current_subset_wnids]

    # Process datasets
    X_train, y_train, train_id = process_dataset('train', data_dir, current_metadata,
                                                 wnid_to_label, label_to_wnid,
                                                 val_ground_truth, current_subset_wnids)
    X_val, y_val, val_id = process_dataset('val', data_dir, current_metadata,
                                           wnid_to_label, label_to_wnid,
                                           val_ground_truth, current_subset_wnids)

    # Save to NPZ file
    output_path = f'./out_files/{subset_name}_cls_part{part_num}.npz'
    np.savez(output_path,
             X_train=X_train,
             y_train=y_train,
             train_id=train_id,
             X_val=X_val,
             y_val=y_val,
             val_id=val_id)

    print(f"Part {part_num}/{total_parts} processing complete. Data saved to '{output_path}'.")

def generate_imagenet_subset(subset_name, data_dir, max_ram, concatenate=False):
    # Load the subset WNIDs
    subset_file = f'./aux_files/{subset_name}.txt'
    with open(subset_file, 'r') as f:
        global_wnids = [line.strip() for line in f if line.strip()]

    num_classes = len(global_wnids)
    ram_per_class = 1 / 5  # 1G per 5 classes = 0.2G per class
    total_ram_required = num_classes * ram_per_class  # in G

    N = int(math.ceil(total_ram_required / max_ram))
    if N < 1:
        N = 1

    num_classes_per_part = int(math.ceil(num_classes / N))

    print(f"Total number of classes: {num_classes}")
    print(f"Total estimated RAM required: {total_ram_required}G")
    print(f"Maximum RAM allowed: {max_ram}G")
    print(f"Number of parts: {N}")
    print(f"Number of classes per part: {num_classes_per_part}")

    # Create the global mapping from WNID to label index
    wnid_to_label = {wnid: idx for idx, wnid in enumerate(global_wnids)}

    # Generate combined metadata
    generate_metadata(subset_file=subset_file,
                      mapping_file='./aux_files/meta_clsloc.csv',
                      output_file=f'./out_files/{subset_name}_cls_metadata.csv',
                      wnid_to_label=wnid_to_label)

    # Load the mapping file
    mapping_file = f'./out_files/{subset_name}_cls_metadata.csv'
    global_metadata = []
    with open(mapping_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            global_metadata.append({
                'subset_label': int(row['subset_label']),
                'original_label': int(row['original_label']),
                'WNID': row['WNID'],
                'num_train_images': int(row['num_train_images']),
            })

    label_to_wnid = {item['original_label']: item['WNID'] for item in global_metadata}

    # Load validation ground truth
    val_ground_truth_file = './aux_files/ILSVRC2015_clsloc_validation_ground_truth.txt'
    with open(val_ground_truth_file, 'r') as f:
        val_ground_truth = [int(line.strip()) for line in f if line.strip()]

    # Split WNIDs into N parts
    for part_num in range(N):
        start_idx = part_num * num_classes_per_part
        end_idx = min((part_num + 1) * num_classes_per_part, num_classes)
        part_wnids = global_wnids[start_idx:end_idx]

        print(f"\nProcessing part {part_num + 1}/{N}, classes {start_idx} to {end_idx - 1}")

        # Process datasets for the current part
        generate_cls_subset_part(subset_name=subset_name,
                                 part_num=part_num + 1,
                                 total_parts=N,
                                 wnids=part_wnids,
                                 global_metadata=global_metadata,
                                 wnid_to_label=wnid_to_label,
                                 label_to_wnid=label_to_wnid,
                                 val_ground_truth=val_ground_truth,
                                 data_dir=data_dir)
        
    if concatenate == True:
        concatenate_parts(subset_name, N, output_file=f'./out_files/{subset_name}_cls.npz')
        out_files_dir = './out_files/'
        print("Removing parts...")
        for part_num in range(1, N + 1):
            part_file = os.path.join(out_files_dir, f'{subset_name}_cls_part{part_num}.npz')
            if os.path.exists(part_file):
                os.remove(part_file)
        print("Parts removed.")
        
def concatenate_parts(subset_name, num_parts, output_file):
    # Paths
    out_files_dir = './out_files/'
    temp_files = []

    # Initialize variables
    keys = None
    arrays_info = {}

    # First pass: Determine the total shape and dtype for each key
    for part_num in range(1, num_parts + 1):
        npz_file = os.path.join(out_files_dir, f'{subset_name}_cls_part{part_num}.npz')
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"NPZ file {npz_file} not found.")
        with np.load(npz_file) as data:
            if keys is None:
                keys = data.files  # List of keys in the NPZ files
                # Initialize arrays_info dictionary
                for key in keys:
                    arrays_info[key] = {'shapes': [], 'dtype': data[key].dtype}
            for key in keys:
                arrays_info[key]['shapes'].append(data[key].shape)

    # Compute total shapes for concatenated arrays
    for key, info in arrays_info.items():
        shapes = info['shapes']
        dtype = info['dtype']
        total_size = sum([shape[0] for shape in shapes])  # Sum over the first dimension
        base_shape = list(shapes[0])
        base_shape[0] = total_size
        info['total_shape'] = tuple(base_shape)
        info['dtype'] = dtype

    # Create memory-mapped files for each array
    memmap_arrays = {}
    for key, info in arrays_info.items():
        total_shape = info['total_shape']
        dtype = info['dtype']
        temp_file = os.path.join(out_files_dir, f'temp_{key}.npy')
        temp_files.append(temp_file)
        memmap_arrays[key] = np.lib.format.open_memmap(temp_file, mode='w+', dtype=dtype, shape=total_shape)

    # Second pass: Copy data into memory-mapped arrays with progress bar
    for key in keys:
        idx = 0  # Start index for current key
        total_size = arrays_info[key]['total_shape'][0]
        with tqdm(total=total_size, desc=f"Concatenating {key}", unit='samples') as pbar:
            for part_num in range(1, num_parts + 1):
                npz_file = os.path.join(out_files_dir, f'{subset_name}_cls_part{part_num}.npz')
                with np.load(npz_file) as data:
                    arr = data[key]
                    size = arr.shape[0]
                    memmap_arrays[key][idx:idx + size] = arr
                    idx += size
                    pbar.update(size)

    # Create the final NPZ file by zipping the memory-mapped NPY files with progress bar
    with zipfile.ZipFile(output_file, mode='w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for key in keys:
            temp_file = os.path.join(out_files_dir, f'temp_{key}.npy')
            file_size = os.path.getsize(temp_file)
            zip_info = zipfile.ZipInfo(key + '.npy')
            zip_info.compress_type = zipfile.ZIP_DEFLATED
            zip_info.file_size = file_size  # Set the file size to handle large files
            zip_info.flag_bits |= 0x08  # Set bit 3 of flag_bits to indicate CRC and sizes are not known
            with zipf.open(zip_info, 'w') as dest, open(temp_file, 'rb') as src:
                with tqdm(total=file_size, desc=f"Zipping {key}", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    while True:
                        data = src.read(1024 * 1024)  # Read in 1MB chunks
                        if not data:
                            break
                        dest.write(data)
                        pbar.update(len(data))

    # Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)
    
    print(f"Concatenation complete. Combined NPZ file saved at '{output_file}'.")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_name', type=str, default='imagenet30', help='Subset name.')
    parser.add_argument('--data_dir', type=str, default='./Data/CLS-LOC', help='Directory of the dataset "train" and "val" folders.')
    parser.add_argument('--max_ram', type=float, default=100, help='Maximum RAM in gigabytes to use.')
    parser.add_argument('--concatenate', action='store_true', help='Concatenate parts into a single NPZ file.')
    args = parser.parse_args()

    # Generate ImageNet subset
    generate_imagenet_subset(args.subset_name, args.data_dir, args.max_ram, args.concatenate)
