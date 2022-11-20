import os
from model import TextModel
from transformers import BertTokenizer, BertConfig
import torch
import torch.utils.data as torchdata
import argparse
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()


def get_args():
    parser = argparse.ArgumentParser(description='Bert representation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    # parser.add_argument('--root_dir', type=str, default='/home/zhuyuejia/Code/BERT')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    # @TODO:处理出现频次高于10次的songs,现在用1000首代替
    parser.add_argument('--data_dir', type=str, default='./data/songs_1000.csv')
    my_args = parser.parse_args()

    return my_args


def test(args, epoch):
    textNet.eval()

# def tmp():
#     args = get_args()
#     texts, ids = read_data(args=args)  # string, int

if __name__ == '__main__':
    texts = ["我今天想睡觉", "fufu是一只可爱的猫猫"]
    texts = texts*100000
    # @ TODO: 重写dataloader
    # dataloader = torchdata.DataLoader(texts, batch_size=64, shuffle=False)
    args = get_args()
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese/vocab.txt')
    textNet = TextModel(batch_size=args.batch_size)
    textNet.cuda()

    for epoch in range(args.num_epochs):
        test(args, epoch)

    tokens, segments, input_masks = [], [], []
    for text in texts:
        indexed_tokens = tokenizer.encode(text)
        tmp_tokenized_text = tokenizer.convert_ids_to_tokens(indexed_tokens)
        tokens.append(indexed_tokens)
        segments.append([0]*len(indexed_tokens))
        input_masks.append([1]*len(indexed_tokens))
    max_len = max(set(len(single) for single in tokens))

    for i in range(len(tokens)):
        padding = [0]*(max_len-len(tokens[i]))
        tokens[i] += padding
        segments[i] += padding  # 一个输入只有一个句子
        input_masks[i] += padding

    tokens_tensor = torch.tensor(tokens)
    segments_tensor = torch.tensor(segments)
    input_masks_tensor = torch.tensor(input_masks)

    text_hashCodes = textNet(tokens_tensor, segments_tensor, input_masks_tensor)
    print(text_hashCodes)
    print(text_hashCodes.shape)
