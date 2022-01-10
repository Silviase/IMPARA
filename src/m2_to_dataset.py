# 文対について, ある一つの変更を適用しなかった場合の類似度を計測する．
# n個の編集集合から部分集合をとる．このランクは 含まれる編集の類似度差(1-similarity)の総和である．
# 任意に選んだ二つの部分集合はa.によって順序づけられる．このランクによってデータセットが作成できる．

import argparse
import os
import pickle
import random
from torch.utils.data.dataset import TensorDataset
from transformers import BertTokenizer, BertModel
import torch
import re
import numpy as np

LIMIT = 30
MAX_LENGTH = 128
ANNOTATOR = 0
SAMPLE = 4096

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')


def parse_m2_to_db(file):
    db = []
    changes = []
    source = None
    with open(file, "r") as fl:
        for line in fl:
            line = line.strip()
            if line == "":
                assert source is not None
                db.append((source, split_changes_by_annot(changes)))
                changes = []
                source = None
            elif line.startswith("S"):
                source = line[2:]
            elif line.startswith("A"):
                properties = re.split("\|\|\|", line[2:])
                start, end = properties[0].split()
                properties[0] = end
                properties.insert(0, start)
                changes.append(properties)
            else:
                raise "unrecognized line " + line
    return db


def split_changes_by_annot(changes):
    res = {}
    for change in changes:
        annot = change[-1]
        if annot not in res:
            res[annot] = []
        res[annot].append(change)
    return list(res.values())


def apply_changes(sentence, changes):
    changes = sorted(changes, key=lambda x: (int(x[0]), int(x[1])))
    res = []
    last_end = 0
    s = sentence.split()
    for change in changes:
        start = int(change[0])
        assert last_end == 0 or last_end <= start, "changes collide in places:" + \
            str(last_end) + ", " + str(start) + \
            "\nSentence: " + sentence + "\nChanges " + str(changes)
        if start == -1:
            print("noop action, no change applied")
            assert change[2] == "noop"
            return sentence
        res += s[last_end:start] + [change[3]]
        last_end = int(change[1])
    res += s[last_end:]
    return " ".join(res)


def cossim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def sentence_to_ids(sentence):
    encoded = tokenizer.encode_plus(
        sentence,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"]


def calc_similarity(sent1, sent2):
    ids1 = tokenizer(sent1, return_tensors="pt")
    ids2 = tokenizer(sent2, return_tensors="pt")
    embs_s = np.mean(model(**ids1)[0][0].detach().numpy(), axis=0)
    embs_t = np.mean(model(**ids2)[0][0].detach().numpy(), axis=0)
    return min(1, cossim(embs_s, embs_t))


def get_importance(source, target, edit):
    length = len(edit[ANNOTATOR])
    importances = []
    for no_use in range(length):
        use = [True] * length
        use[no_use] = False
        target_use = apply_all_changes(source, edit, use)
        similarity = calc_similarity(target, target_use)
        importance = 1 - similarity
        importances.append(importance)
    return importances


def calc_rank(importances, edit):
    rank = 0
    for i, e in enumerate(edit):
        if e:
            rank += importances[i]
    return rank


def apply_all_changes(source, changes, use):
    edit = []
    for i, e in use:
        if e:
            edit.append(changes[ANNOTATOR][i])
    source = apply_changes(source, edit)
    return source


def get_use(number):
    use = [False for _ in range(number)]
    k = random.randint(0, number)
    use_idxs = random.sample(range(number), k=k)
    for i in use_idxs:
        assert i < number
        use[i] = True
    return use


def change_use(use):
    length = len(use)
    res = [not x if random.random() < length else x for x in use]
    return res


def get_sent_pairs(source, edits, limit):
    length = len(edits[ANNOTATOR])
    limit = min(limit, length * (length + 1) / 2)
    # get target sentence
    target = apply_all_changes(source, edits, [True] * length)

    # calculate importance of each edit
    importances = get_importance(source, target, edits)
    # print("importances", importances)

    # select sentence pair
    sent_pairs = [(source, target)]
    if length > 1:
        for i in range(int(limit)):
            # select edit to use
            use1 = get_use(length)
            use2 = change_use(use1)
            # get sentence
            sentence1 = apply_all_changes(source, edits, use1)
            sentence2 = apply_all_changes(source, edits, use2)
            # calculate rank
            rank1 = calc_rank(importances, use1)
            rank2 = calc_rank(importances, use2)
            # make sentence pair
            if rank1 < rank2:
                sent_pairs.append((sentence1, sentence2))
            elif rank1 > rank2:
                sent_pairs.append((sentence2, sentence1))
    print("sent_pairs", len(sent_pairs))
    return sent_pairs


def make_dataset(sentence_pairs):
    ids_lo, ids_hi = [], []
    for i, (s1, s2) in enumerate(sentence_pairs):
        ids_lo.append(sentence_to_ids(s1))
        ids_hi.append(sentence_to_ids(s2))
    ids_lo, ids_hi = select(ids_lo, ids_hi, SAMPLE)
    ids_lo, ids_hi = torch.cat(ids_lo, dim=0), torch.cat(ids_hi, dim=0)
    dataset = TensorDataset(ids_lo, ids_hi)
    return dataset


def select(lo, hi, sample):
    assert len(lo) == len(hi)
    random.seed(42)
    idxs = random.choices(range(len(lo)), k=sample)
    res_lo, res_hi = [], []
    for i in idxs:
        res_lo.append(lo[i])
        res_hi.append(hi[i])
    return res_lo, res_hi


def main():
    # parse_arguments
    parser = argparse.ArgumentParser(description='Process M2 to dataset for proposal training.')
    parser.add_argument('--m2_train', type=str, required=True, help='M2 file path.')
    parser.add_argument('--m2_test', type=str, required=True, help='M2 file path.')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save dataset.')
    args = parser.parse_args()

    db_train = parse_m2_to_db(args.m2_train)
    sources_train = [x[0] for x in db_train]
    changes_train = [x[1] for x in db_train]
    db_test = parse_m2_to_db(args.m2_test)
    sources_test = [x[0] for x in db_test]
    changes_test = [x[1] for x in db_test]

    # get candidate pairs
    sentence_pairs_train = []
    for source, changes in zip(sources_train, changes_train):
        # changes ; [][][]
        if len(changes) > 0:
            sentence_pairs_train.extend(get_sent_pairs(source, changes, LIMIT))

    sentence_pairs_test = []
    for source, changes in zip(sources_test, changes_test):
        # changes ; [][][]
        if len(changes) > 0:
            sentence_pairs_test.extend(get_sent_pairs(source, changes, LIMIT))

    # make dataset
    dataset_train = make_dataset(sentence_pairs_train)
    dataset_test = make_dataset(sentence_pairs_test)

    # save dataset
    os.makedirs(args.save_path, exist_ok=True)
    print("save path:", args.save_path)
    with open(os.path.join(args.save_path, f"dataset_train_{LIMIT}.pkl"), "wb") as f:
        pickle.dump(dataset_train, f)
    with open(os.path.join(args.save_path, f"dataset_test_{LIMIT}.pkl"), "wb") as f:
        pickle.dump(dataset_test, f)


if __name__ == "__main__":
    main()
