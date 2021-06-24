"""Train Code """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import RandomSampler
from tokenization import BertTokenizer, WhitespaceTokenizer
from modeling import BertForPreTrainingLossMask
from optimization import BertAdam

from nn.data_parallel import DataParallelImbalance
import seq2seq_loader as seq2seq_loader
from batch_decode import decode_batch
from metrics import f_one, distinct, bleu_metric
from itertools import cycle



def check_mem(cuda_device):
    devices_info = os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occupy_mem_new(cuda_device_list, ratio=0.6):
    import time

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cuda_device_list)
    for id, cuda_device in enumerate(cuda_device_list):
        while True:
            total, used = check_mem(cuda_device)
            total = int(total)
            used = int(used)
            occupy = int(total * ratio)
            print("Device-{}: {}/{}/{}".format(cuda_device, total, used, occupy))
            if occupy + used <= total * 0.95:
                print('Find device-{}!'.format(cuda_device))
                try:
                    x = torch.cuda.FloatTensor(256, 1024, occupy, device='cuda:{}'.format(id))
                    del x
                except RuntimeError:
                    time.sleep(2)
                    continue
                break
    # input('>>>>') # todo: del


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    # Train File
    parser.add_argument("--src_file", default="qkr_train.src", type=str)
    parser.add_argument("--tgt_file", default="qkr_train.tgt", type=str)
    parser.add_argument("--check_file", default="qkr_train.check", type=str)
    parser.add_argument("--style_file", default="qkr_train.style", type=str)

    # Test File
    parser.add_argument("--test_seen_rank_file", default="test_data/wizard_random/rank_test_wizard_random.src.tk", type=str)
    parser.add_argument("--test_seen_tgt_file", default="test_data/wizard_random/test_wizard_random.tgt", type=str)
    parser.add_argument("--test_unseen_rank_file", default="test_data/wizard_topic/rank_test_wizard_topic.src.tk", type=str)
    parser.add_argument("--test_unseen_tgt_file", default="test_data/wizard_topic/test_wizard_topic.tgt", type=str)

    parser.add_argument("--bert_model", default="unilm_v2_bert_pretrain", type=str)
    parser.add_argument("--log", default="wizard_of_wikipedia/log", type=str)
    parser.add_argument("--exp_name", default="0525_test", type=str)
    parser.add_argument("--model_recover_path", default="unilm_v2_bert_pretrain/unilm1.2-base-uncased.bin", type=str)

    # Other parameters
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--train_batch_size", default=10, type=int)
    parser.add_argument("--eval_batch_size", default=500, type=int)
    parser.add_argument("--valid_every", default=5000, type=int)
    parser.add_argument("--start_valid", default=0, type=int)
    parser.add_argument("--print_every", default=10, type=int)
    parser.add_argument("--learning_rate", default=0.00003, type=float)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--finetune_decay", action='store_true', help="Weight decay to the original weights.")
    parser.add_argument("--num_train_epochs", default=20, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion_step", default=500, type=int, help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout rate for hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout rate for attention probabilities.")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--fp32_embedding', action='store_true', help="Whether to use 32-bit float precision instead of 16-bit for embeddings")
    parser.add_argument('--tokenized_input', action='store_true', help="Whether the input is tokenized.")
    parser.add_argument('--max_len_a', type=int, default=0, help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_b', type=int, default=40, help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--trunc_seg', default='a', help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true', help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.3, type=float, help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument("--mask_prob_eos", default=0, type=float, help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=40, help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of workers for the data loader.")

    parser.add_argument('--max_position_embeddings', type=int, default=256,
                        help="max position embeddings")
    parser.add_argument('--gpu_list', default='', type=str)
    parser.add_argument('--gpu_ratio', default=0.85, type=float)

    args = parser.parse_args()

    assert Path(args.model_recover_path).exists(), "--model_recover_path doesn't exist"


    # Random Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("\nParameters:")
    for attr, value in sorted(vars(args).items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Selecting wihch GPU to use
    occupy_mem_new(args.gpu_list.split(','), ratio=args.gpu_ratio)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    out_dir = os.path.join(args.log, args.exp_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print("Writing to {}\n".format(out_dir))

    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings
    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer

    C_bi_uni_pipeline = [seq2seq_loader.C_Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys(
    )), tokenizer.convert_tokens_to_ids, args.max_seq_length, new_segment_ids=False,
                                                             truncate_config={'max_len_a': args.max_len_a,
                                                                              'max_len_b': args.max_len_b,
                                                                              'trunc_seg': args.trunc_seg,
                                                                              'always_truncate_tail': args.always_truncate_tail},
                                                             mask_source_words=False,
                                                             skipgram_prb=0.0,
                                                             skipgram_size=1,
                                                             mask_whole_word=False, mode="s2s",
                                                             has_oracle=False, num_qkv=0,
                                                             s2s_special_token=False,
                                                             s2s_add_segment=False,
                                                             s2s_share_segment=False,
                                                             pos_shift=False)]

    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Create training dataset begin... | %s " % time_str)
    train_dataset = seq2seq_loader.C_Seq2SeqDataset(
        args.src_file, args.tgt_file, args.check_file, args.style_file, args.train_batch_size, data_tokenizer, args.max_seq_length,
        file_oracle=None, bi_uni_pipeline=C_bi_uni_pipeline
    )
    train_sampler = RandomSampler(train_dataset, replacement=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, sampler=train_sampler, num_workers=args.num_workers,
        collate_fn=seq2seq_loader.batch_list_to_batch_tensors, pin_memory=False
    )
    train_dataloader = cycle(train_dataloader)
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Create training dataset end... | %s " % time_str)

    # note: args.train_batch_size has been changed to (/= args.gradient_accumulation_steps)
    t_total = int(len(train_dataset) / args.train_batch_size * args.num_train_epochs / args.gradient_accumulation_steps)

    # Recover model
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Recover model: {} | {}".format(args.model_recover_path, time_str))
    model_recover = torch.load(args.model_recover_path, map_location='cpu')

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

    model = BertForPreTrainingLossMask.from_pretrained(
        args.bert_model, state_dict=model_recover, num_labels=2, num_rel=0,
        type_vocab_size=2, config_path=None, task_idx=3,
        num_sentlvl_labels=0, max_position_embeddings=args.max_position_embeddings,
        label_smoothing=args.label_smoothing, fp32_embedding=args.fp32_embedding, relax_projection=0,
        new_pos_ids=False, ffn_type=0, hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob, num_qkv=0, seg_emb=False,
        mask_word_id=mask_word_id, search_beam_size=5,
        length_penalty=0, eos_id=eos_word_ids, sos_id=sos_word_id, forbid_duplicate_ngrams=True,
        forbid_ignore_set=None, mode="s2s")

    model.to(device)
    if n_gpu > 1:
        model = DataParallelImbalance(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=-1, t_total=t_total)


    def train_step(global_step):
        lm_loss_total, kl_loss_total, mutual_loss_total = 0.0, 0.0, 0.0
        for _ in range(args.gradient_accumulation_steps):
            batch = next(train_dataloader)
            batch = [t.to(device) if t is not None else None for t in batch]

            input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx, tgt_pos, labels, ks_labels, style_ids, style_labels, check_ids = batch
            oracle_pos, oracle_weights, oracle_labels = None, None, None

            pretrain = None

            model.train()
            loss_tuple = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next, masked_pos=masked_pos,
                               masked_weights=masked_weights, task_idx=task_idx, masked_pos_2=oracle_pos,
                               masked_weights_2=oracle_weights, masked_labels_2=oracle_labels, mask_qkv=mask_qkv,
                               tgt_pos=tgt_pos, labels=labels,
                               ks_labels=ks_labels, train_vae=False, style_ids=style_ids,
                               style_labels=style_labels, check_ids=check_ids, pretrain=pretrain)
            masked_lm_loss, next_sentence_loss, KL_loss, Mutual_loss, Golden_loss, cosine_similarity_loss, predict_kl_loss = loss_tuple
            # if n_gpu > 1:  # mean() to average on multi-gpu.
            masked_lm_loss = masked_lm_loss.mean()
            next_sentence_loss = next_sentence_loss.mean()
            Mutual_loss = Mutual_loss.mean()
            Golden_loss = Golden_loss.mean()
            KL_loss = KL_loss.mean()
            cosine_similarity_loss = cosine_similarity_loss.mean()
            predict_kl_loss = predict_kl_loss.mean()

            loss = masked_lm_loss + next_sentence_loss + KL_loss + predict_kl_loss + Mutual_loss + Golden_loss  # cosine_similarity_loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            lm_loss_total += masked_lm_loss.item() / args.gradient_accumulation_steps
            kl_loss_total += KL_loss.item() / args.gradient_accumulation_steps
            mutual_loss_total += Mutual_loss.item() / args.gradient_accumulation_steps

        optimizer.step()
        optimizer.zero_grad()

        if global_step % args.print_every == 0 and global_step != 0:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("Step: %d \t| lm_loss: %.3f \t| kl_loss: %.3f \t| mutual_loss: %.3f \t| %s" % (
                global_step, lm_loss_total, kl_loss_total, mutual_loss_total, time_str
            ))

    def dev_step(split, global_step):
        if split == 'test_seen':
            src_out_path = args.test_seen_rank_file
            tgt_path = args.test_seen_tgt_file
        elif split == 'test_unseen':
            src_out_path = args.test_unseen_rank_file
            tgt_path = args.test_unseen_tgt_file
        else:
            raise ValueError

        model.eval()


        with open(src_out_path, encoding="utf-8") as file:
            dev_src_lines = file.readlines()
        with open(tgt_path, encoding="utf-8") as file:
            golden_response_lines = file.readlines()

        decode_result = decode_batch(model, dev_src_lines, args.eval_batch_size)
        golden_response_lines = golden_response_lines[:len(decode_result)]
        with open(os.path.join(out_dir, "{}-decoded-iter-{}.txt".format(split, global_step)), "w", encoding="utf-8") as f:
            for _hyp, _ref in zip(decode_result, golden_response_lines):
                f.writelines("{} ||| {}\n".format(_hyp, _ref))

        assert len(decode_result) == len(golden_response_lines)

        b1, b2, b3 = bleu_metric(decode_result, golden_response_lines)
        d1, d2 = distinct(decode_result)
        f1 = f_one(decode_result, golden_response_lines)

        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("*****************************************")
        print("{} results..........".format(split))
        print("hypothesis: ", len(decode_result))
        print("Step: {} \t| {}".format(global_step + 1, time_str))
        print("BLEU-1/2/3: {:.4f}/{:.4f}/{:.4f}".format(b1, b2, b3))
        print("Distinct-1/2: {:.4f}/{:.4f}".format(d1, d2))
        print("F1: {:.4f}".format(f1))
        print("*****************************************")

        return f1

    max_f1 = 0
    for i in range(99999):
        train_step(i + 1)
        if (i + 1) % args.valid_every == 0 and (i + 1) >= args.start_valid:
            f1 = dev_step("test_seen", i + 1)
            if "cmudog" not in args.src_file:
                f1_unseen = dev_step("test_unseen", i + 1)
            if f1 > max_f1:
                max_f1 = f1

                # save_path = "{}-best".format(checkpoint_prefix)
                # os.makedirs(save_path, exist_ok=True)
                # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                # torch.save(model_to_save.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

                print("Saved model checkpoint to {}\n".format(checkpoint_prefix))


if __name__ == "__main__":
    main()
