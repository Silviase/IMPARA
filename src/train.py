import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking.fluent import log_artifact, log_param, log_metric
import hydra
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import SequentialSampler, RandomSampler, DataLoader
import torch
import pickle
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7,8"


def loss_fn(lo, hi):
    return torch.sigmoid(lo.logits - hi.logits).mean()


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    train_loss = []
    for i, batch in enumerate(dataloader):
        if i % 20 == 0 and i > 0:
            print(f"batch [20s] : {i}")
            print("loss:", np.array(train_loss).mean())
        input_ids_low, input_ids_high = batch[0].to(device), batch[1].to(device)
        output_low, output_high = model(input_ids_low), model(input_ids_high)
        loss = loss_fn(output_low, output_high)
        train_loss.append(loss.item())
        mlflow.log_metric("train_loss_batch", np.array(train_loss).mean())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return np.array(train_loss).mean()


def test(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids_low, input_ids_high = batch[0].to(device), batch[1].to(device)
            output_low, output_high = model(input_ids_low), model(input_ids_high)
            loss = loss_fn(output_low, output_high)
            test_loss.append(loss.item())
            mlflow.log_metric("train_loss_batch", np.array(test_loss).mean())
    return np.array(test_loss).mean()


def load_dataset(dataset_dir):
    with open(os.path.join(dataset_dir, "dataset_train_30.pkl"), "rb") as f:
        train_dataset = pickle.load(f)
    with open(os.path.join(dataset_dir, "dataset_test_30.pkl"), "rb") as f:
        test_dataset = pickle.load(f)
    return train_dataset, test_dataset


@hydra.main(config_path="../config", config_name="training")
def learn(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "training_record/mlruns")
    cpus = os.cpu_count()
    with mlflow.start_run():
        log_param("epoch", cfg.epoch)
        log_param("lr", cfg.learning_rate)
        log_param("batch_size", cfg.batch_size)
        log_param("max_length", cfg.max_length)
        log_param("gpu", 6)
        log_param("cpus", cpus)
        log_param("pretrained_model", cfg.pretrained_model)
        dataset_train, dataset_test = load_dataset(cfg.dataset_dir)
        dataloader_train = DataLoader(
            dataset_train,
            sampler=RandomSampler(dataset_train),
            batch_size=cfg.batch_size,
            num_workers=cpus,
        )
        dataloader_test = DataLoader(
            dataset_test,
            sampler=SequentialSampler(dataset_test),
            batch_size=cfg.batch_size,
            num_workers=cpus,
        )
        model = BertForSequenceClassification.from_pretrained(
            cfg.pretrained_model,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
        )
        model = torch.nn.DataParallel(model)
        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
        model.to(device)
        train_loss = []
        test_loss = []
        print("start training")
        print("train data length", len(dataset_train))
        os.makedirs(cfg.save_dir, exist_ok=True)
        for e in range(cfg.epoch):
            train_ = train(model, dataloader_train, optimizer, loss_fn, device)
            test_ = test(model, dataloader_test, loss_fn, device)
            print(f"epoch: {e}")
            print(f"train loss: {train_}")
            print(f"test loss: {test_}")
            log_metric("train_loss", train_)
            log_metric("test_loss", test_)
            train_loss.append(train_)
            test_loss.append(test_)
            model.module.save_pretrained(
                os.path.join(
                    cfg.save_dir,
                    f"LIM30_simirality_e{e}_lr{cfg.learning_rate}_model"
                )
            )
        plt.plot(range(1, 1 + cfg.epoch), train_loss, label="train")
        plt.plot(range(1, 1 + cfg.epoch), test_loss, label="test")
        plt.legend()
        plt.savefig(
            f"c_{cfg.chain}_e_{cfg.epoch}_lr_{cfg.learning_rate}_loss.png")
        log_artifact(
            f"c_{cfg.chain}_e_{cfg.epoch}_lr_{cfg.learning_rate}_loss.png")
        plt.clf()


if __name__ == '__main__':
    learn()
