import copy
import numpy as np
import argparse
import re
import random
import pickle
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import os
from typing import List

# in this repository
from utils.bert_selector import BertSelector

class M2Parser:
    def __init__(self, src) -> None:
        self.source_file = src
        self.db = self.parse()

    # load M2 file and return list of [src, [edits]]
    def parse(self) -> list:
        db = []  # source, edits_per_annotator
        edits = []
        source = None
        with open(self.source_file, "r") as fl:
            for line in fl:
                line = line.strip()
                if line == "":
                    if source is not None:
                        for edits_by_id in self.split_edits_by_annot(edits):
                            db.append((source, edits_by_id))
                        edits = []
                        source = None
                elif line.startswith("S"):
                    source = line[2:]
                elif line.startswith("A"):
                    properties = re.split("\|\|\|", line[2:])
                    start, end = properties[0].split()
                    properties[0] = end
                    properties.insert(0, start)
                    edits.append(properties)
                else:
                    raise "unrecognized line " + line
        return db

    def split_edits_by_annot(self, edits):
        res = {}
        for edit in edits:
            annot = edit[-1]
            if annot not in res:
                res[annot] = []
            res[annot].append(edit)
        return list(res.values())


class M2Dataset:
    def __init__(self, args):

        if args.use_cache:
            print("Loading cached information...")
            with open(args.cache, "rb") as fl:
                self = pickle.load(fl)
            print("Loading cached information... Done")

            # config difference
            self.output_file = args.out
            self.dataset_size = args.dataset_size
        else:
            # Configuration
            self.source_file = args.src
            self.output_file = args.out
            self.lang = args.lang
            self.max_length = args.max_length
            self.dataset_size = args.dataset_size
            self.parallel = args.parallel

            # Make directory for output file if not exists
            if not os.path.exists(os.path.dirname(self.output_file)):
                os.makedirs(os.path.dirname(self.output_file))

            # Load BERT items
            bert_selector = BertSelector(self.lang)
            self.tokenizer = bert_selector.load_tokenizer()
            self.model = bert_selector.load_pretrained_model()

            # Parse M2 file and calculate impacts of each edit
            print("Parsing M2 file...")
            self.parser = M2Parser(self.source_file)
            print("Parsing M2 file... Done")
            print("Length: ", len(self.parser.db))

            self.src = [s[0] for s in self.parser.db]
            self.edits = [s[1] for s in self.parser.db]
            print("Get target sentences...")
            self.trg = [self.apply_edits(s, es)
                        for s, es in zip(self.src, self.edits)]
            print("Get target sentences... Done")

            if not self.parallel:
                # Calculate impacts of each edit
                self.impacts = [self.calculate_impacts(s, es, trg) 
                                for s, es, trg in tqdm(zip(self.src, self.edits, self.trg))]

            # Save in cache_dir
            with open(self.output_file + "_info.pkl", "wb") as fl:
                pickle.dump(self, fl)

        # Create dataset
        self.dataset = self.create_dataset()
        self.save()

    def save(self):
        with open(self.output_file + ".pkl", "wb") as fl:
            pickle.dump(self.dataset, fl)

    def calculate_impact_for_edit(self, s1, s2):
        s1_tokens = self.tokenizer(s1, return_tensors="pt")
        s2_tokens = self.tokenizer(s2, return_tensors="pt")
        s1_emb = np.mean(self.model(**s1_tokens)[0][0].detach().numpy(), axis=0)
        s2_emb = np.mean(self.model(**s2_tokens)[0][0].detach().numpy(), axis=0)
        return 1 - min(1, np.dot(s1_emb, s2_emb) / (np.linalg.norm(s1_emb) * np.linalg.norm(s2_emb)))

    def get_sentence_pairs(self):
        lo, hi = [], []
        for src, edits, trg, impacts in tqdm(zip(self.src, self.edits, self.trg, self.impacts), desc="Create sentence pairs"):
            # continue if no edits
            if len(edits) == 0 or src == trg:
                continue
            
            for loop in range(min(10, len(edits) * (len(edits)-1) // 2)):
                e1, e2, i1, i2 = self.select_two_edits(edits, impacts)
                if sum(i1) < sum(i2):
                    s_lo, s_hi = M2Dataset.apply_edits(src, e1), M2Dataset.apply_edits(src, e2)
                else:
                    s_lo, s_hi = M2Dataset.apply_edits(src, e2), M2Dataset.apply_edits(src, e1)

                if s_lo == s_hi:
                    continue
                lo.append(s_lo)
                hi.append(s_hi)
        return lo, hi

    def create_dataset(self):
        print("Creating dataset...")
        if self.parallel:
            lo, hi = [], []
            # filter out sentences with no edits
            for s, t in zip(self.src, self.trg):
                if s == t:
                    continue
                lo.append(s)
                hi.append(t)
        else:
            lo, hi = self.get_sentence_pairs()

        # Modify dataset_size
        if len(lo) >= self.dataset_size:
            before = len(lo)
            selected_ids = random.sample(range(len(lo)), self.dataset_size)
            lo = [lo[i] for i in selected_ids]
            hi = [hi[i] for i in selected_ids]
            assert len(lo) == len(hi) == self.dataset_size
            print("Modified dataset size from {} to {}".format(before, self.dataset_size))
        else:
            print(f"Dataset size({len(lo)}) is smaller than the size to sample ({self.dataset_size})!")

        with open(self.output_file + ".tsv", "w") as fl:
            for l, h in zip(lo, hi):
                fl.write(f"{l}\t{h}\n")

        ids_lo = [self.sentence_to_ids(s) for s in tqdm(lo, desc="toknize;lo")]
        ids_hi = [self.sentence_to_ids(s) for s in tqdm(hi, desc="toknize;hi")]
        ids_lo = torch.cat(ids_lo, dim=0)
        ids_hi = torch.cat(ids_hi, dim=0)
        print("Creating dataset... Done")
        print("Dataset size:", ids_lo.shape[0])
        return TensorDataset(ids_lo, ids_hi)

    @staticmethod
    def apply_edits(sentence, edits):
        edits = sorted(edits, key=lambda x: (int(x[0]), int(x[1])))
        res = []
        last_end = 0
        s = sentence.split()
        for edit in edits:
            start = int(edit[0])
            assert last_end == 0 or last_end <= start, "edits collide in places:" + \
                str(last_end) + ", " + str(start) + \
                "\nSentence: " + sentence + "\nedits " + str(edits)
            if start == -1:
                assert edit[2] == "noop"
                return sentence
            res += s[last_end:start] + [edit[3]]
            last_end = int(edit[1])
        res += s[last_end:]
        return " ".join(res)

    @staticmethod
    def select_two_edits(edits, impacts):
        n = len(edits)
        e1_idxs = random.sample(range(n), k=random.randint(0, n))
        e2_idxs = copy.deepcopy(e1_idxs)
        for i in range(n):
            if random.random() < 1 / n:
                if i in e1_idxs:
                    e2_idxs.remove(i)
                else:
                    e2_idxs.append(i)
        e1 = [edits[i] for i in e1_idxs]
        e2 = [edits[i] for i in e2_idxs]
        i1 = [impacts[i] for i in e1_idxs]
        i2 = [impacts[i] for i in e2_idxs]
        return e1, e2, i1, i2

    def calculate_impacts(self, src, edits, trg) -> List[float]:
        impacts = []
        for idx in range(len(edits)):
            edits_without_e = edits[:idx] + edits[idx+1:]
            trg_without_e = M2Dataset.apply_edits(src, edits_without_e)
            impacts.append(self.calculate_impact_for_edit(trg, trg_without_e))
        return impacts

    def sentence_to_ids(self, sentence):
        encoded = self.tokenizer.encode(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        return encoded


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--cache", type=str)
    parser.add_argument("--dataset_size", type=int, default=4096)
    parser.add_argument("--message", type=str, default="m2 to dataset experiment.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = M2Dataset(args)


if __name__ == "__main__":
    main()
