from transformers import BertTokenizer
from transformers import BertModel, BertForSequenceClassification

class BertSelector:
    def __init__(self, lang):
        self.lang = lang
        self.lang_dict = {
            "en": "bert-base-cased",
            "de": "bert-base-german-cased",
            "cz": "DeepPavlov/bert-base-bg-cs-pl-ru-cased",
            "ja": "cl-tohoku/bert-base-japanese",
            "zh": "bert-base-chinese",
            "multi": "bert-base-multilingual-cased",
        }
        self.name = self.load_name(lang)

    def load_name(self, lang):
        if lang in self.lang_dict:
            return self.lang_dict[lang]
        else:
            raise "unknown language : " + lang

    def load_tokenizer(self):
        return BertTokenizer.from_pretrained(self.name)

    def load_pretrained_model(self):
        return BertModel.from_pretrained(self.name)

    def load_qe_model(self):
        return BertForSequenceClassification.from_pretrained(self.name)
