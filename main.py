from utils import generate_cls_subset as generator
import argparse

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_name', type=str, default='imagenet30', help='Subset name.')
    parser.add_argument('--data_dir', type=str, default='./Data/CLS-LOC', help='Directory of the dataset "train" and "val" folders.')
    
    args = parser.parse_args()
    subset_name = args.subset_name
    data_dir = args.data_dir
    
    generator.generate_cls_subset(subset_name=subset_name, data_dir=data_dir)
