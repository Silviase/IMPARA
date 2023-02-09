from torch.utils.data.dataset import Dataset


class SEDataset(Dataset):
    def __init__(self, sources, sentences):
        self.sources = sources
        self.sentences = sentences

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.sentences[idx]
