import os

import numpy as np

from model import TextModel
from transformers import BertTokenizer, BertConfig
import torch
import torch.utils.data as torchdata
import argparse
from utils import *
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()


def get_args():
    parser = argparse.ArgumentParser(description='Bert representation')
    parser.add_argument('--batch_size', type=int, default=7, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    # @TODO:处理出现频次高于10次的songs,现在用1000首代替
    parser.add_argument('--data_dir', type=str, default='./data/songs_1000.csv')
    parser.add_argument('--write_dir', type=str, default='./data/')
    my_args = parser.parse_args()

    return my_args


def extract_embedding(my_args):
    textNet.eval()
    final_embedding = np.zeros((dataset.__len__(), 768), dtype=float)
    id_seq = np.zeros(dataset.__len__())
    id_num = 0
    for batch_idx, (texts, ids) in enumerate(dataloader):
        tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese/vocab.txt')
        tokens, segments, input_masks = get_tokens(texts=texts, tokenizer=tokenizer)
        features = textNet(tokens, segments, input_masks)  # features在GPU上
        features_numpy = features.data.cpu().numpy()
        final_embedding[id_num: id_num + ids.shape[0]] = features_numpy
        id_seq[id_num: id_num + ids.shape[0]] = ids
        id_num += ids.shape[0]
    return final_embedding, id_seq


def write(embeddings, ids):
    filepath_emb = os.path.join(args.write_dir, 'Bert_embedding')
    filepath_ids = os.path.join(args.write_dir, 'ids')
    np.save(filepath_emb, embeddings)
    np.save(filepath_ids, ids)
    # read(filepath_emb+'.npy')
    # read(filepath_ids+'.npy')


def read(path):
    data = np.load(path)
    print(data)


if __name__ == '__main__':
    args = get_args()
    # @ TODO: 重写dataloader
    dataset = MyDataset(args=args)
    dataloader = torchdata.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    textNet = TextModel()
    textNet.cuda()
    embedding_seqs, id_seqs = extract_embedding(args)
    write(embedding_seqs, id_seqs)
