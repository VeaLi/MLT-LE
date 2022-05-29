import os
from tqdm.auto import tqdm


experiment = {}

if not  os.path.exists('logs/'):
    os.mkdir('logs/')

tasks_auxillary = ['No', 'pH', 'T', 'qed', 'N']

for num_res_blocks in [0]:#,1, 2]:
    for discount in [0.0, 0.9]:
        for data in ['data20k_mltle.zip','data20k_tdc.zip']:
            for auxillary in tasks_auxillary:
                for use_loss_weights in [False]:
                    for positional in [False, True]:
                        for mode in ['protein_3', 'protein_1']:
                            params = {"num_res_blocks": num_res_blocks, "discount": discount, "data": f"data_subset/{data}", "use_loss_weights" : use_loss_weights, "auxillary": auxillary, "positional":positional, 'mode':mode}
                            experiment[f"{data.replace('.zip','')}_Res{num_res_blocks}_{int(discount*10)}_{int(use_loss_weights)}_{auxillary}_{int(positional)}_{mode}"] = params

for k, v in tqdm(experiment.items()):
    if k +'.log' not in os.listdir('logs'):
        os.system(f"python run.py --experiment_name {k} --epochs 20 --batch_size 300 --num_res_blocks {v['num_res_blocks']} --discount {v['discount']} --data {v['data']} --use_loss_weights {v['use_loss_weights']} --auxillary {v['auxillary']} --positional {v['positional']} --mode {v['mode']}")
