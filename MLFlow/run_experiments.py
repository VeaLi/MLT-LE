import os
from tqdm.auto import tqdm


experiment = {}


for num_res_blocks in [0, 1, 2]:
    for discount in [0.0, 0.3, 0.6, 0.9]:
        for data in ['yes', 'no']:
            params = {"num_res_blocks": num_res_blocks, "discount": discount, "data": f"data_subset/10k_{data}.csv"}
            experiment[f"Res{num_res_blocks}_{int(discount*10)}_{data}"] = params

for k, v in tqdm(experiment.items()):
    os.system(f"python run.py --experiment_name {k} --epochs 10 --batch_size 32 --num_res_blocks {v['num_res_blocks']} --discount {v['discount']} --data {v['data']}")
