import os

def train_zrkgc(
        gpu, exp_name, batch_size=1, eval_batch_size=1, start_valid=0, valid_every=1000, learning_rate=3e-5, gpu_ratio=0.85,
):
    params = {
        "train_batch_size": batch_size,
        "eval_batch_size": eval_batch_size,

        # files
        "src_file": "wizard_data/qkr_train_10.src",
        "tgt_file": "wizard_data/qkr_train_10.tgt",
        "check_file": "wizard_data/qkr_train_10.check",
        "style_file": "wizard_data/qkr_train_10.style",

        "test_seen_rank_file": "test_data/wizard_random/rank_test_wizard_random.src.tk",
        "test_seen_tgt_file": "test_data/wizard_random/test_wizard_random.tgt",

        "test_unseen_rank_file": "test_data/wizard_topic/rank_test_wizard_topic.src.tk",
        "test_unseen_tgt_file": "test_data/wizard_topic/test_wizard_topic.tgt",

        "model_recover_path": "model/ZRKGC_model",
        "valid_every": valid_every,
        "print_every": 10,
        "start_valid": start_valid,
        "learning_rate": learning_rate,
        "num_train_epochs": 10,
        "warmup_proportion_step": 100,
        "gradient_accumulation_steps": 1,
        "mask_prob": 0.3,

        "exp_name": exp_name,
        "log": 'wizard_of_wikipedia/log',

        "gpu_list": gpu,
        "gpu_ratio": gpu_ratio,
    }
    command = "python -u run_seq2seq.py --tokenized_input --always_truncate_tail"
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    for param in sorted_params:
        command += " --{}={}".format(param[0], param[1])
    print(command)
    command += ' | tee wizard_of_wikipedia/{}.txt'.format(exp_name)
    os.system(command)


if __name__ == '__main__':
    train_zrkgc("4", "0525_wizard_10", start_valid=8000, batch_size=4, eval_batch_size=16, learning_rate=7e-6, valid_every=1000)
