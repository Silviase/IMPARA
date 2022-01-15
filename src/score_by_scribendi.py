from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import Levenshtein
from fuzzywuzzy import fuzz
import torch
import os
from typing import List
import pandas as pd
import scipy.stats as stats
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"


batch_size = 32
twelve_systems_path = 'path/to/twelve_systems'
twelve_systems = [
    'AMU', 'CAMB', 'CUUI',
    'IITB', 'IPN', 'NTHU',
    'PKU', 'POST', 'RAC',
    'SJTU', 'UFC', 'UMC', 'INPUT',
]

# cited from table 3(b), 3(c)
human_ew = {'AMU': 0.628, 'CAMB': 0.561, 'CUUI': 0.550,
            'IITB': 0.485, 'IPN': 0.300, 'NTHU': 0.437,
            'PKU': 0.506, 'POST': 0.539, 'RAC': 0.566,
            'SJTU': 0.463, 'UFC': 0.513, 'UMC': 0.495,
            'INPUT': 0.456,
            }

# cited from table 3(c)
human_ts = {'AMU': 0.273, 'CAMB': 0.182, 'CUUI': 0.105,
            'IITB': -0.055, 'IPN': -0.358, 'NTHU': -0.142,
            'PKU': -0.001, 'POST': 0.080, 'RAC': 0.114,
            'SJTU': -0.074, 'UFC': -0.041, 'UMC': -0.022,
            'INPUT': -0.062
            }


class GECDataset(torch.utils.data.Dataset):
    def __init__(self, srcs: List[str], preds: List[str]) -> None:
        self.srcs = srcs
        self.preds = preds

    def __len__(self) -> int:
        return len(self.srcs)

    def __getitem__(self, idx: int) -> List[str]:
        return self.srcs[idx], self.preds[idx]


class ScribendiScorer():

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = "gpt2-medium"
        self.model = GPT2LMHeadModel.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_id)

    def perplexity(self, text: str) -> float:
        tokens = self.tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)[0].item()
        return outputs

    def _token_sort_ratio(self, src: str, pred: str) -> float:
        return fuzz.token_sort_ratio(src, pred)

    def _levenshtein_distance_ratio(self, src: str, pred: str) -> float:
        return 1 - (Levenshtein.distance(src, pred) / (len(src) + len(pred)))

    def _score(self, ppl_srcs, ppl_preds, tsrs, ldrs) -> List[float]:
        scores = []
        for ppl_src, ppl_pred, tsr, ldr in zip(ppl_srcs, ppl_preds, tsrs, ldrs):
            if ppl_src < ppl_pred:
                scores.append(-1)
            elif ppl_src == ppl_pred:
                scores.append(0)
            elif max(tsr, ldr) > 0.8:
                scores.append(1)
            else:
                scores.append(-1)
        return scores

    def score_batch(self, sources: List[str], outputs: List[str]) -> List[float]:
        scores = []
        gec_dataset = GECDataset(sources, outputs)
        gec_dataloader = torch.utils.data.DataLoader(gec_dataset, batch_size=batch_size, shuffle=False)
        for i, batch in enumerate(gec_dataloader):
            print(f'{i} / {len(gec_dataloader)}')
            srcs, preds = batch
            with torch.no_grad():
                ppl_srcs = [self.perplexity(src) for src in srcs]
                ppl_preds = [self.perplexity(pred) for pred in preds]
                tsrs = [self._token_sort_ratio(src, pred) for src, pred in zip(srcs, preds)]
                ldrs = [self._levenshtein_distance_ratio(src, pred) for src, pred in zip(srcs, preds)]
                score = self._score(ppl_srcs, ppl_preds, tsrs, ldrs)
                scores.extend(score)
        return scores


def main():
    # load input and output
    outputs = {}
    for system in twelve_systems:
        with open(os.path.join(twelve_systems_path, system), 'r') as f:
            outputs[system] = f.readlines()

    # initialize scorer
    scorer = ScribendiScorer()

    print("========== Corpus Level ===========")
    scores = {}
    # calculate scores
    for system in twelve_systems:
        scores[system] = scorer.score_batch(outputs['INPUT'], outputs[system])
    point = {}

    # calculate corpus point
    for system, score in scores.items():
        print(system, score.count(0), score.count(1), score.count(-1))
        point[system] = sum(score)
    
    # calculate correlation
    df = pd.DataFrame([point, human_ew, human_ts], index=['Scribendi', 'EW', 'TS']).T
    r = stats.pearsonr(list(df['Scribendi']), list(df['TS']))
    rho = stats.spearmanr(list(df['Scribendi']), list(df['TS']))
    print(f"TrueSkill Pearson: {r[0]}, p-value: {r[1]}")
    print(f"TrueSkill Spearman: {rho[0]}, p-value: {rho[1]}")
    
    r = stats.pearsonr(list(df['Scribendi']), list(df['EW']))
    rho = stats.spearmanr(list(df['Scribendi']), list(df['EW']))    
    print(f"Expected Wins Pearson: {r[0]}, p-value: {r[1]}")
    print(f"Expected Wins Spearman: {rho[0]}, p-value: {rho[1]}")

    print("========== Sentence Level ===========")
    import xml.etree.ElementTree as ET
    # load XML file
    xml = ET.parse('/path/to/human_judgments')
    root = xml.getroot()
    print(root.tag)
    wins = []
    for rankingitem in root[0]:
        ranks = []
        srcid = rankingitem.attrib['src-id']
        for item in rankingitem:
            rank = item.attrib['rank']
            systems = item.attrib['system'].split()
            for system in systems:
                ranks.append([rank, system])
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                if ranks[i][0] < ranks[j][0]:
                    wins.append([srcid, ranks[i][1], ranks[j][1], 1])
                elif ranks[i][0] > ranks[j][0]:
                    wins.append([srcid, ranks[i][1], ranks[j][1], 2])
    df = pd.DataFrame(wins, columns=['src-id', 'sys-1', 'sys-2', 'human_win'])
    sent1 = [outputs[sys1][int(srcid)] for srcid, sys1 in zip(df['src-id'], df['sys-1'])]
    sent2 = [outputs[sys2][int(srcid)] for srcid, sys2 in zip(df['src-id'], df['sys-2'])]
    inputs = [outputs['INPUT'][int(srcid)] for srcid in df['src-id']]
    df['sent-1'], df['sent-2'] = sent1, sent2
    df['input'] = inputs
    scores_1 = scorer.score_batch(inputs, sent1)
    scores_2 = scorer.score_batch(inputs, sent2)

    scribendi_win = [1 if s1 > s2 else 0 if s1 == s2 else 2 for s1, s2 in zip(scores_1, scores_2)]
    df['scribendi_win'] = scribendi_win
    df['human-scribendi'] = (df['human_win'] == df['scribendi_win'])
    df.to_csv('path/to/judgments.csv')
    print(df['human-scribendi'].value_counts().keys())
    scribendi_acc = df['human-scribendi'].value_counts()[True] / df['human-scribendi'].value_counts().sum()
    scribendi_kendall = (df['human-scribendi'].value_counts()[True] - df['human-scribendi'].value_counts()[False]) / df['human-scribendi'].value_counts().sum()
    print(f"Scribendi accuracy: {scribendi_acc}")
    print(f"Scribendi kendall: {scribendi_kendall}")


if __name__ == "__main__":
    main()
