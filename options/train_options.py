import argparse
train_parser = argparse.ArgumentParser(description='UNSB Training')

# Adding arguments to the parser
train_parser.add_argument('--total_iters', type=int, default=0, help='Total number of iterations')
train_parser.add_argument('--optimize_time', type=float, default=0.1, help='Time for optimization')
train_parser.add_argument('--epoch_count', type=int, default=1, help='Starting count of epochs')
train_parser.add_argument('--n_epochs', type=int, default=90, help='Number of epochs')
train_parser.add_argument('--n_epochs_decay', type=int, default=90, help='Number of decaying epochs')
train_parser.add_argument('--print_freq', type=int, default=100, help='Frequency of printing outputs')
train_parser.add_argument('--gpu_ids', type=int, nargs='+', default=[1], help='List of GPU IDs')
train_parser.add_argument('--create_dir', type=str, default='/kaggle/working/generated_images', help='Directory to create for output images')