import ray, argparse
from exp.config import get_hparams_config, get_exp_dir
from exp import session

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("percentage", type=int)
parser.add_argument("gpus", type=float, default=1.0)


def main():
    args = parser.parse_args()
    dataset, perc, gpus = args.dataset, args.percentage, args.gpus
    exp_dir = get_exp_dir(dataset)
    ray.init(address="auto", ignore_reinit_error=True)
    config = get_hparams_config(dataset, perc)
    session.run(perc, exp_dir, config, gpus_per_trial=gpus)


if __name__ == "__main__":
    main()
