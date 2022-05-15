import os
from tqdm.auto import tqdm


experiment = {}


for num_res_blocks in [0, 1, 2]:
    for discount in [0.0, 0.3, 0.6, 0.9]:
        for data in ['yes', 'no']:
            for use_loss_weights in [False, True]:
                params = {"num_res_blocks": num_res_blocks, "discount": discount, "data": f"data_subset/10k_{data}.csv", "use_loss_weights" : use_loss_weights}
                experiment[f"Res{num_res_blocks}_{int(discount*10)}_{data}_{int(use_loss_weights)}"] = params

for k, v in tqdm(experiment.items()):
    if k +'.log' not in os.listdir('logs'):
        os.system(f"python run.py --experiment_name {k} --epochs 10 --batch_size 32 --num_res_blocks {v['num_res_blocks']} --discount {v['discount']} --data {v['data']} --use_loss_weights {v['use_loss_weights']}")
