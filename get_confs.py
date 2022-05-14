import ray
from ray import tune

def get_config(name, perc, mode):
    tune_exp = tune.Analysis(f"experiments/{name}_{perc}_{mode}/{name}_ms", default_metric='eval_score', default_mode='max')
    config = tune_exp.get_best_config()
    return config

if __name__ == '__main__':
    seq = {'vanilla': [], 'intrinsic_plasticity': []}

    for mode in ['vanilla', 'intrinsic_plasticity']:
        for perc in [25, 50, 75, 100]:
            print(mode, perc)
            seq[mode].append(get_config('WESAD', perc, mode)['SEQ_LENGTH'])
    
    print(seq)