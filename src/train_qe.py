import numpy as np
import pickle
import torch
from torch.utils.data import SequentialSampler, RandomSampler, DataLoader
from transformers import BertForSequenceClassification, AdamW
from mlflow.tracking.fluent import log_metric, log_params
import mlflow
import matplotlib.pyplot as plt
import argparse
import os

# 自作のやつ
from tqdm.auto import tqdm
from tools import notify_to_slack as NT
from utils.bert_selector import BertSelector
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(lo, hi):
    return torch.sigmoid(lo.logits - hi.logits).mean()

def train(model, dataloader, optimizer, loss_fn):
    model.train()
    loss_batch = []
    for i, batch in tqdm(enumerate(dataloader), desc="Training"):
        if i % 20 == 0 and i > 0:
            print(f"batch [20s] : {i}")
            print("loss:", np.array(loss_batch).mean())
        ids_lo, ids_hi = batch[0].to(device), batch[1].to(device)
        preds_lo, preds_hi = model(ids_lo), model(ids_hi)
        loss = loss_fn(preds_lo, preds_hi)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_batch.append(loss.item())
    return np.array(loss_batch).mean()


def test(model, dataloader, loss_fn):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test"):
            ids_lo, ids_hi =  batch[0].to(device), batch[1].to(device)
            preds_lo, preds_hi = model(
                ids_lo), model(ids_hi)
            loss = loss_fn(preds_lo, preds_hi)
            test_loss.append(loss.item())
    return np.array(test_loss).mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--dataset_size", type=int, default=4096)
    parser.add_argument("--lang", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args_dict = {k: v for k, v in vars(args).items()}
    bert_selector = BertSelector(args.lang)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    mlflow.set_experiment(args.task_name)
    with mlflow.start_run():
        log_params(args_dict)

        # ---- load dataset ----
        with open(args.train_dataset, "rb") as f:
            dataset_train = pickle.load(f)
        with open(args.test_dataset, "rb") as f:
            dataset_test = pickle.load(f)

        # ---- prepare dataloader ----
        dataloader_train = DataLoader(
            dataset_train,
            sampler=RandomSampler(dataset_train),
            batch_size=args.batch_size,
        )
        dataloader_test = DataLoader(
            dataset_test,
            sampler=SequentialSampler(dataset_test),
            batch_size=args.batch_size,
        )

        # ---- prepare model & optimizer ----
        model = BertForSequenceClassification.from_pretrained(
            bert_selector.name,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
        )
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        model = torch.nn.DataParallel(model)
        model.to(device)

        # ---- progress bar ----
        progress_bar = tqdm(range(args.num_epochs))

        train_loss, test_loss = [], []
        print("Task name:", args.task_name)
        print("train data length", len(dataset_train))

        for e in range(args.num_epochs):
            train_loss_by_epoch = train(model, dataloader_train, optimizer, loss_fn)
            train_loss.append(train_loss_by_epoch)
            test_loss_by_epoch = test(model, dataloader_test, loss_fn)
            test_loss.append(test_loss_by_epoch)

            model_savepath = os.path.join(
                args.model_dir,
                f"sample_{args.sample}_size_{args.dataset_size}_e_{e+1}_lr_{args.learning_rate}_model"
            )

            model.module.save_pretrained(model_savepath)

            log_metric("train_loss", train_loss_by_epoch, step=e+1)
            log_metric("dev_loss", test_loss_by_epoch, step=e+1)
            progress_bar.update(1)

        # ---- memorize best epoch ----
        best_epoch = np.argmin(test_loss) + 1
        mlflow.log_metric("best-epoch", best_epoch)

        # ---- plot loss ----
        plt.plot(range(1, 1 + args.num_epochs), train_loss, label="train")
        plt.plot(range(1, 1 + args.num_epochs), test_loss, label="valid")
        plt.legend()
        plt.savefig(os.path.join(
            args.model_dir,
            f"sample_{args.sample}_size_{args.dataset_size}_lr_{args.learning_rate}_loss.png"
        ))

        # ---- notify ----
        message = [
            f"Training{args.task_name} was finished",
            f"best epoch: {best_epoch}"
        ]
        NT.notify_to_slack("\n".join(message))


if __name__ == '__main__':
    main()
