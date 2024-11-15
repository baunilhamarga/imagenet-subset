from utils import generate_cls_subset as generator
import argparse

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_name', type=str, default='imagenet30', help='Subset name.')
    parser.add_argument('--data_dir', type=str, default='./Data/CLS-LOC', help='Directory of the dataset "train" and "val" folders.')
    parser.add_argument('--max_ram', type=float, default=100, help='Maximum RAM in gigabytes to use.')
    parser.add_argument('--concatenate', action='store_true', help='Concatenate parts into a single NPZ file.')
    args = parser.parse_args()

    # Generate ImageNet subset
    generator.generate_imagenet_subset(args.subset_name, args.data_dir, args.max_ram, args.concatenate)
