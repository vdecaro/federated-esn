import ray, argparse
from exp.config import get_hparams_config, get_analysis, get_exp_dir
from exp.phase import design, test

parser = argparse.ArgumentParser()
parser.add_argument("method", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("percentage", type=int)
parser.add_argument("phase", type=str, default="mrt")
parser.add_argument("gpus", type=float, default=1.0)


def main():
    args = parser.parse_args()
    method, dataset, perc, gpus = args.method, args.dataset, args.percentage, args.gpus
    exp_dir = get_exp_dir(method, dataset, perc)
    if "m" in args.phase:
        ray.init(address="auto", ignore_reinit_error=True)
        config = get_hparams_config(method, dataset, perc)
        design.run("model_selection", exp_dir, config, gpus_per_trial=gpus)
    if "r" in args.phase:
        ray.init(address="auto", ignore_reinit_error=True)
        if "joint" in method:
            config = get_hparams_config(method, dataset, perc)
        else:
            config = get_analysis("model_selection", exp_dir).get_best_config()
        design.run("retraining", exp_dir, config, gpus_per_trial=gpus)
    if "t" in args.phase:
        analysis = get_analysis("retraining", exp_dir)
        test.run(analysis, exp_dir, device="cuda")


if __name__ == "__main__":
    main()
