from utils import generate_cls_subset as generator
import argparse

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_name', type=str, default='imagenet30', help='Subset name.')
    
    args = parser.parse_args()
    subset_name = args.subset_name
    
    generator.generate_cls_subset(subset_name)
