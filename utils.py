from tqdm import tqdm
import torch
import torch.utils.data as torchdata
import re
import pandas as pd


class Process:
    def __init__(self, filepath='./kgrec_songs/KGRec-musicSongs.txt', stopwords_path='./stoplist.csv'):
        self.filepath = filepath
        self.data = pd.read_csv(filepath, sep=',', encoding='utf_8', header=0)
        self.stopwords = set(pd.read_csv(stopwords_path, encoding='utf_8', header=None, names=['words'])['words'])

    def process_tag(self, input_tags):  # indie-rock catchy ...
        tags_sentence = input_tags.replace('-', ' ') + ' '
        tags = tags_sentence.split(' ')
        tags = [tag for tag in tags if tag not in self.stopwords]
        output_tags = ''
        for tag in tags:
            output_tags = output_tags + tag + ' '
        output_tags = output_tags.lower()  # 模型为uncased，所以要转小写
        return output_tags

    def process_des(self, input_des):
        # 保留字母(a-z, A-Z) 数字 , . 仅仅一个空格
        output_des = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039\x20\u002e\u002c])", "", input_des)
        output_des = output_des.lower()  # 全部转化为小写字母
        return output_des

    def read_data(self):
        ids_list = []  # 存储歌曲id [id1, id2, ...]
        sentiment_list = []  # 存储歌曲的情感文本 [text1, text2, text3, ...]
        print("---------开始读取数据-----------")
        for i in tqdm(range(len(self.data))):
            song_id = int(self.data['idx'][i])
            tag, des = '', ''
            if not pd.isnull(self.data['tag'][i]):
                tag = self.data['tag'][i]
            if not pd.isnull(self.data['des'][i]):
                des = self.data['des'][i]
            info_sentence = self.process_tag(tag) + self.process_des(des)
            sentiment_list.append(info_sentence)  # 获取item的情感句子
            ids_list.append(song_id)  # 获取item的idx
        print("---------读取数据结束-----------")
        print('---------数据长度 %d %d-----------' % (len(sentiment_list), len(ids_list)))
        return sentiment_list, ids_list


def get_tokens(texts, tokenizer):
    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        if len(tokenized_text) > 512:  # 处理长度不能超过512，采取从前面截断的方法控制长度
            tokenized_text = tokenized_text[:512]
            tokenized_text[-1] = '[SEP]'
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))
    max_len = max(set(len(single) for single in tokens))
    for i in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[i]))
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
        process = Process(self.args.data_dir)
        self.texts, self.ids = process.read_data()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        song_text = '[CLS]' + self.texts[idx] + '[SEP]'
        song_id = self.ids[idx]

        return song_text, song_id


def main():
    process = Process(filepath='./kgrec_songs/KGRec-musicSongs.txt', stopwords_path='./stoplist.csv')
    process.read_data()


if __name__ == '__main__':
    main()
