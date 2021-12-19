# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import argparse
import os
import random
import time
import numpy as np
import paddle
import paddle.nn.functional as F
from tqdm import tqdm
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
import csv
from data import create_dataloader, convert_example, NewsDataset
from model import TextCls
from loss import FocalLoss
from attack import FGM_Paral
from loss_acc import AccLoss
# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--train_set", type=str, default='./news/train.txt', help="The full path of train_set_file")
parser.add_argument("--dev_set", type=str, default='./news/dev.txt', help="The full path of dev_set_file")
parser.add_argument("--save_dir", default='./nezha_checkpoints', type=str,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after tokenization. "
                         "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--max_steps', default=-1, type=int, help="If > 0, set total number of training steps to perform.")
parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--accumulate_steps", default=1, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=5, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--eval_step", default=400, type=int, help="Step interval for evaluation.")
parser.add_argument('--save_step', default=2000, type=int, help="Step interval for saving checkpoint.")
parser.add_argument("--warmup_proportion", default=0, type=float,
                    help="Linear warmup proption over the training process.")
# parser.add_argument("--init_from_ckpt", type=str, default='checkpoints/model_17500/model_state.pdparams', help="The path of checkpoint to be loaded.")
parser.add_argument("--init_from_ckpt", type=str, default='./checkpoints/model_5000/model_state.pdparams ', help="The path of checkpoint to be loaded.")
# parser.add_argument("--init_from_ckpt", type=str, default=' ', help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=2021, help="Random seed for initialization.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument("--rdrop_coef", default=0., type=float, help="The coefficient of"
                                                                  "KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")

args = parser.parse_args()


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, vocab=None):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    total_num = 0
    bad_cases = []
    bad_preds = []
    bad_lbls = []
    label_list = data_loader.dataset.data.label_list
    for batch in tqdm(data_loader):
        input_ids, token_type_ids, labels = batch
        total_num += len(labels)
        logits, _ = model(input_ids=input_ids, token_type_ids=token_type_ids, do_evaluate=True)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
        pred_lbls = paddle.argmax(logits, axis=-1)
        bad_mask = pred_lbls != labels
        cases = input_ids.numpy()[bad_mask.numpy()]
        if vocab is not None:
            cases = [''.join(vocab.to_tokens(case)).replace('[CLS]', '').replace('[PAD]', '').replace('[SEP]', '') for
                     case in cases]
        bad_cases.extend(cases)
        bad_preds.extend(pred_lbls.numpy()[bad_mask.numpy()])
        bad_lbls.extend(labels.numpy()[bad_mask.numpy()])

    bad_data = []
    if vocab is not None:
        for case, pred, lbl in zip(bad_cases, bad_preds, bad_lbls):
            pred = label_list[pred]
            lbl = label_list[lbl]
            bad_data.append((case, pred, lbl))
        bad_data = [('error_text', 'pred', 'label')] + bad_data
        with open('bad_orig.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(bad_data)

    print("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))
    model.train()
    metric.reset()
    return accu


def save_model(model, tokenizer, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
    paddle.save(model.state_dict(), save_param_path)
    tokenizer.save_pretrained(save_dir)
    print(f'Saved to {save_dir} over')


def do_train():
    use_acc_loss = True
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)
    train_ds = NewsDataset(data_path=args.train_set, mode='train')
    dev_ds = NewsDataset(data_path=args.dev_set, mode='dev')

    # pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
    #     'ernie-gram-zh')
    # tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
    #     'ernie-gram-zh')

    # pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
    #     'ernie-gram-zh')
    # tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
    #     'ernie-gram-zh')

    # pretrained_model = ppnlp.transformers.BertModel.from_pretrained(
    #     'bert-base-chinese')
    # tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(
    #     'bert-base-chinese')

    # pretrained_model = ppnlp.transformers.BertModel.from_pretrained(
    #     'macbert-base-chinese')
    # tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(
    #     'macbert-base-chinese')

    # pretrained_model = ppnlp.transformers.AlbertModel.from_pretrained(
    #     'albert-chinese-base')
    # tokenizer = ppnlp.transformers.AlbertTokenizer.from_pretrained(
    #     'albert-chinese-base')

    # pretrained_model = ppnlp.transformers.XLNetModel.from_pretrained(
    #     'chinese-xlnet-base')
    # tokenizer = ppnlp.transformers.XLNetTokenizer.from_pretrained(
    #     'chinese-xlnet-base')

    # pretrained_model = ppnlp.transformers.NeZhaModel.from_pretrained(
    #     'nezha-base-chinese')
    # tokenizer = ppnlp.transformers.NeZhaTokenizer.from_pretrained(
    #     'nezha-base-chinese')

    pretrained_model = ppnlp.transformers.RobertaModel.from_pretrained(
        'roberta-wwm-ext')
    tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained(
        'roberta-wwm-ext')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.train_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.eval_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    model = TextCls(pretrained_model, num_classes=len(train_ds.label_list), rdrop_coef=args.rdrop_coef)
    # fgm = FGM_Paral(model)  # 对抗训练,model先传给fgm再并行

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print(f'load state from {args.init_from_ckpt}')

    model = paddle.DataParallel(model)
    # fgm = FGM_Paral(model)  # 对抗训练,先并行再传给fgm
    # fgm = paddle.DataParallel(fgm)
    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    lr=5e-5
    classifier_lr=4e-5

    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params
    )

    criterion = paddle.nn.loss.CrossEntropyLoss()
    criterion2=AccLoss(reduction='mean')


    metric = paddle.metric.Accuracy()

    global_step = 0
    best_accuracy = 0.0
    # Step1：定义 GradScaler，用于缩放loss比例，避免浮点数溢出
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            # Step2：创建AMP上下文环境，开启自动混合精度训练
            with paddle.amp.auto_cast(enable=False):
                logits1, kl_loss = model(input_ids=input_ids, token_type_ids=token_type_ids)
            correct = metric.compute(logits1, labels)
            metric.update(correct)
            acc = metric.accumulate()

            ce_loss = criterion(logits1, labels)
            acc_loss=0
            if use_acc_loss:
                acc_loss,num_samples=criterion2(logits1,labels)
                acc_loss+=1

            if kl_loss > 0:
                loss = ce_loss + kl_loss * args.rdrop_coef+acc_loss
            else:
                loss = ce_loss+acc_loss

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: [%d/%d], batch: [%d/%d], lr: %f ,loss: %.4f, ce_loss: %.4f., kl_loss: %.4f,acc_loss:%.4f, accu: %.4f, speed: %.2f step/s"
                    % (
                    global_step, epoch, args.epochs, step, len(train_data_loader), lr_scheduler.get_lr(), loss, ce_loss,
                    kl_loss,acc_loss, acc,
                    10 / (time.time() - tic_train)))
                tic_train = time.time()

            # Step3：使用 Step1中定义的 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
            scaled = scaler.scale(loss)
            scaled.backward()
            if (step+1)%args.accumulate_steps==0:
                scaler.minimize(optimizer,scaled)
                optimizer.clear_grad()

            if global_step % args.eval_step == 0 and rank == 0:
                # accuracy = evaluate(model, criterion, metric, dev_data_loader,vocab=tokenizer.vocab)
                accuracy = evaluate(model, criterion, metric, dev_data_loader, vocab=None)
                if accuracy > best_accuracy:
                    save_dir = os.path.join(args.save_dir, "model_best")
                    print(f'global_step:{global_step} | Best eval accu:{accuracy}')
                    save_model(model, tokenizer, save_dir)
                    best_accuracy = accuracy

            if global_step % args.save_step == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                save_model(model, tokenizer, save_dir)

            if global_step == args.max_steps:
                return

    if rank == 0:  
        save_dir = os.path.join(args.save_dir, "model_final")
        save_model(model, tokenizer, save_dir)


if __name__ == "__main__":
    do_train()
