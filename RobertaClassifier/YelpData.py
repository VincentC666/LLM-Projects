import json
from torch.utils.data import Dataset
from datasets import load_dataset
from common import constants
from transformers import RobertaTokenizer
import torch




class YelpDataset(Dataset):
    def __init__(self, datafile_path,model_path,filename='train'):
        self.dataset = load_dataset('csv',data_files=f'{constants.CLEAN_DATA_PATH}/{filename}.csv',split='train')
        self.tokenizer = RobertaTokenizer.from_pretrained(f'{model_path}')


    def load_data(self,data):
        tokenizer = self.tokenizer
        texts = [d[0] for d in data]
        labels = [d[1] for d in data]
        tokens = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=texts,
            truncation=True,
            max_length = 1024,
            padding='max_length',
            return_tensors ='pt',
            return_length = True
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = torch.LongTensor(labels)

        return tokens,labels


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        return text, label


