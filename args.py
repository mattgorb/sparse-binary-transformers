import argparse
import sys
import yaml

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # General Config
    parser.add_argument(
        "--data", help="path to dataset base directory", default="/s/luffy/b/nobackup/mgorb/data"
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument("--entity", help="num of entity", type=int, default=None)
    parser.add_argument("--ablation", help="num of entity", type=bool, default=False)
    parser.add_argument("--dmodel", help="num of entity", type=int, default=None)
    parser.add_argument("--dataset", help="dataset", type=str, default="")
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
    parser.add_argument(
        "--window_size", help="WS",   type=int,default=None
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--model_runs", help="number of times to run model", default=3)
    parser.add_argument("--forecast", help="forecast", default=False)
    parser.add_argument("--es_epochs", help="earlystopping epochs", default=None)
    parser.add_argument("--scale_fan", default=False, type=bool)
    parser.add_argument("--lin_prune_rate", default=1, type=float)
    parser.add_argument("--attention_prune_rate", default=1, type=float)
    parser.add_argument("--scheduler", default=False, type=bool)
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--has_src_mask", default=False, help="Weight initialization mode")


    parser.add_argument(
        "-p",
        "--print-freq",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )

    parser.add_argument(
        "--gpu", default=1, type=int, help="gpu id"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save_every", default=-1, type=int, help="Save every ___ epochs"
    )


    parser.add_argument(
        "--conv-type", type=str, default=None, help="What kind of sparsity to use"
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether or not to train only subnet (this freezes weights)",
    )


    parser.add_argument(
        "--weight_init", default=None, help="Weight initialization modifications"
    )
    parser.add_argument(
        "--score_init", default=None, help="Weight initialization modifications"
    )


    parser.add_argument(
        "--weight_seed", default=0, help="Weight initialization modifications"
    )
    parser.add_argument(
        "--score_seed", default=0, help="Weight initialization modifications"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training. "
    )


    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()




run_args()
