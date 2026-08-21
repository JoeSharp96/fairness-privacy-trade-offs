import sys
import json
import os
PATH = "output/final_output"
SKEW = ["0.05", "0.1", "0.2", "0.3"]
SEEDS = [1,2,3,4,5,6,7,8,9,10]
ALPHA = ["0.2","0.5","1.0","10.0","500.0"]
DIRECTORIES = os.listdir(PATH)

with open(f"{PATH}/5.0/10/0.3/results.json", "r") as fp:
    data = json.load(fp)
with open(f"{PATH}/5.0/10/0.3/results.json", "w") as fp:
    with open(f"{PATH}/5.0/10/0.3/results500.json", "r") as nfp:
        data_new = json.load(nfp)
    data['run_metrics']['500.0'] = data_new['run_metrics']['500.0']
    json.dump(data, fp)

for dir in DIRECTORIES:
    for seed in SEEDS:
        for skew in SKEW:
            with open(f"{PATH}/{dir}/{seed}/{skew}/results.json", "r") as fp:
                data = json.load(fp)
            with open(f"{PATH}/{dir}/{seed}/{skew}/results.json", "w") as fp:
                for key in data['final_metrics'].keys():
                    data['final_metrics'][key].clear()
                    print(seed)
                    for alpha in ALPHA:
                        data['final_metrics'][key][alpha] = data['run_metrics'][alpha]['round_metrics']['30'][key]
                json.dump(data,fp)




