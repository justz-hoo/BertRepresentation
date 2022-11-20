import torch
import torch.utils.data as torchdata
import os
import re
import opencc


# 文本预处理， 删掉一些奇奇怪怪的东西
def process(input_text):
    pattern = re.compile("[^\u4e00-\u9fa5^!]")  # 只保留中文和感叹号
    sentence = re.sub(pattern, '', input_text)  # 把文本中匹配到的字符替换成空字符
    sentence = ''.join(sentence.split())  # 去除空白
    sentence = sentence.replace(',,,', ',')
    sentence = sentence.replace(',,', ',')
    # 繁体转化为简体
    new_sentence = opencc.OpenCC('t2s').convert(sentence)  # 繁体转为简体
    # Bert一次性处理的token的长度不能超过512, 这里没有加入[CLS][SEP]
    if len(new_sentence) > 510:
        new_sentence = new_sentence[:510]
    return new_sentence


def read_data(filepath):
    file = open(filepath, 'r', encoding='utf-8')
    rows = file.readlines()[1:]
    tags_list = []
    ids_list = []
    for row in rows:
        i = 0
        song_id = ''
        for character in row:
            song_id += character
            i += 1
            if character == ',':
                break
        tags = row[i + 1: -2]
        song_id = song_id[:-1]
        new_tags = process(tags)
        tags_list.append(new_tags)
        ids_list.append(eval(song_id))
    return tags_list, ids_list


def get_tokens(texts, tokenizer):
    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
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
    return tokens_tensor.cuda(), segments_tensor.cuda(), input_masks_tensor.cuda()


class MyDataset(torchdata.Dataset):
    def __init__(self, args):
        self.args = args
        self.texts, self.ids = read_data(self.args.data_dir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        song_text = '[CLS]' + self.texts[idx] + '[SEP]'
        song_id = self.ids[idx]

        return song_text, song_id
