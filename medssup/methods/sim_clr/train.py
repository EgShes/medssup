import argparse
from pathlib import Path

from medssup.methods.sim_clr.train_loop import train_model


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", default="test", help="Name of experiment")
    # training params
    parser.add_argument("--iterations", type=int, default=100000, help="Number of iterations to run")
    parser.add_argument("--num_workers", type=int, default=8, help="Num workers to use in dataloaders")
    parser.add_argument("--device", default="cuda:0", help="Device to place model parameters on")
    parser.add_argument("--bs", type=int, default=8, help="Batch size")
    parser.add_argument("--eval_interval", type=int, default=500, help="Iters before evaluation and scheduler step")
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=10,
        help="Stops training after this amount of eval intervals with no improvements",
    )
    # optimizer params
    parser.add_argument("--optimizer_type", default="lars", help="Type of model optimizer to use")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Learning rate")
    parser.add_argument("--disable_fp16", dest="fp16", action="store_false", help="Whether to disable fp16 training")

    # data params
    parser.add_argument("--data_path", type=Path, default="data/raw/rsna", help="Path to dataset folder")
    parser.add_argument(
        "--annotations_path",
        type=Path,
        default="data/splits/rsna",
        help="Relative path to subsets folder from dataset root",
    )
    parser.add_argument("--image_size", type=int, default=384, help="Size all images are resized to")
    parser.add_argument(
        "--windows",
        default="bb|br|su",
        help="Windows to apply to a slice. Possible values: 'bb', 'br', 'je', 'st', 'su'. Must be in alphabetic order",
    )

    # architecture params
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--trf_dim", type=int, default=1024)
    parser.add_argument("--trf_depth", type=int, default=6)
    parser.add_argument("--trf_heads", type=int, default=16)
    parser.add_argument("--trf_dropout", type=float, default=0.1)
    parser.add_argument("--trf_emb_dropout", type=float, default=0.1)
    parser.add_argument("--projection_dim", type=int, default=32)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    train_model(args)
