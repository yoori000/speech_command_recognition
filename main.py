import argparse
from train import *
from predict import *

## Parser 생성하기
parser = argparse.ArgumentParser(description="speech recognition task")

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=5, type=int, dest="num_epoch")
parser.add_argument("--log_interval", default=20, type=int, dest="log_interval")

parser.add_argument("--ckpt_dir", default="./model", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log/experiment_1", type=str, dest="log_dir", help="tensorboard log directory")
parser.add_argument("--test_model_path", default="./model/3model.pt", type=str, dest="test_model_path")

args = parser.parse_args()


if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        predict(args)
