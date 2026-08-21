import sys
import json
import os
SEED = range(1,11)
SKEW = [0.05, 0.1, 0.2, 0.3]
SAVE_PATH = sys.argv[1]

old_keys = [
    "train_loss",
    "eval_client_loss",
    "eval_client_acc",
    "eval_server_loss",
    "eval_server_acc",
    "demographic_parity",
    "equalised_odds",
    "equal_opportunity",
    "equalised_accuracy",
    "eval_server_min_acc",
    "eval_server_maj_acc",
    "true_positive_rate",
    "false_positive_rate",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives"
]

new_keys = [
    "train_loss",
    "client_eval_loss",
    "client_eval_acc",
    "server_eval_loss",
    "server_eval_acc",
    "server_demographic_parity",
    "server_equalised_odds",
    "server_equal_opportunity",
    "server_equalised_accuracy",
    "server_min_accuracy",
    "server_maj_accuracy",
    "server_maj_tpr",
    "server_maj_fpr",
    "server_maj_tp",
    "server_maj_fp",
    "server_maj_tn",
    "server_maj_fn"
]
very_new_keys = [
    "server_min_tpr",
    "server_min_fpr",
    "server_min_tp",
    "server_min_fp",
    "server_min_tn",
    "server_min_fn",
]


def fix():
    for seed in SEED:
        for skew in SKEW:
            with open(f"{SAVE_PATH}/{seed}/{skew}/results.json", "r", encoding="utf-8") as fp:
                data = json.load(fp)
            with open(f"{SAVE_PATH}/{seed}/{skew}/results.json", "w", encoding="utf-8") as fp:
                for alpha in ["0.5", "1.0", "10.0", "500.0"]:
                    for round in range(1,31):
                        round = str(round)
                        for new, old in zip(new_keys,old_keys):
                            data['run_metrics'][alpha]['round_metrics'][round][new] = data['run_metrics'][alpha]['round_metrics'][round].pop(old)
                        for key in very_new_keys:
                            data['run_metrics'][alpha]['round_metrics'][round][key] = 0.0
                data['final_metrics'] = {}
                for key in new_keys + very_new_keys:
                    data['final_metrics'][key] = {}
                    for alpha in ["0.5", "1.0", "10.0", "500.0"]:
                        data['final_metrics'][key][alpha] = data['run_metrics'][alpha]['round_metrics']['30'][key]                    
                json.dump(data, fp)

def delete():
    for seed in SEED:
        for skew in SKEW:
            if os.path.exists(f"{SAVE_PATH}/{seed}/{skew}/results_2.json"):
                os.remove(f"{SAVE_PATH}/{seed}/{skew}/results_2.json")

fix()
delete()