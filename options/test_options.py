import argparse

# Initialize the parser
test_parser = argparse.ArgumentParser(description="Test Script Configuration")

test_parser.add_argument('--num_threads', type=int, default=0, help='Number of threads, fixed at 0 for this test script')
test_parser.add_argument('--batch_size', type=int, default=1, help='Batch size, fixed at 1 for this test script')
test_parser.add_argument('--serial_batches', action='store_true', default=True, help='Disable data shuffling, fixed as True')
test_parser.add_argument('--no_flip', action='store_true', default=True, help='No flip, fixed as True')
test_parser.add_argument('--aspect_ratio', type=float, default=1.0, help='Aspect ratio for the test')