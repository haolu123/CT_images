import argparse
    
def parse_args():
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--batch_size", default=10, type=int, help="number of batch size")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--distributed", action="store_true", help="use monai distributed training")
    parser.add_argument("--epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--model_save_dir", default="./model_save", type=str, help="model save directory")
    parser.add_argument("--result_dir", default="./result", type=str, help="result save directory")
    parser.add_argument("--val_interval", default=5, type=int, help="validation interval")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed training')
    # parser.add_argument('--rank_id', required=False, default=0, type=int, help='Needed to identify the node and save separate weights.')
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    
    argv = parser.parse_args()
    return argv