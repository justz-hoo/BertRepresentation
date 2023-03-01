from model import TextModel
from transformers import BertTokenizer
import argparse
from utils import *
import numpy as np
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()


def get_args():
    parser = argparse.ArgumentParser(description='Bert representation')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    # @TODO:处理出现频次高于10次的songs,现在用1000首代替
    parser.add_argument('--data_dir', type=str, default='./kgrec_songs/KGRec-musicSongs.txt',
                        help='读取歌曲的id和info信息原始数据')
    parser.add_argument('--write_dir', type=str, default='./kgrec_songs/Embedding',
                        help='写歌曲的情感向量信息，存储为向量形式')
    my_args = parser.parse_args()

    return my_args


def extract_embedding():
    textNet.eval()
    final_embedding = np.zeros((dataset.__len__(), 768), dtype=float)
    id_seq = np.zeros(dataset.__len__())
    id_num = 0
    for batch_idx, (texts, ids) in tqdm(enumerate(dataloader)):
        tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')
        tokens, segments, input_masks = get_tokens(texts=texts, tokenizer=tokenizer)
        features = textNet(tokens, segments, input_masks)  # features在GPU上
        features_numpy = features.data.cpu().numpy()
        final_embedding[id_num: id_num + ids.shape[0]] = features_numpy
        id_seq[id_num: id_num + ids.shape[0]] = ids
        id_num += ids.shape[0]
    print(id_num)

    return final_embedding, id_seq


def write(embeddings, ids):
    filepath_emb = os.path.join(args.write_dir, 'Bert_embedding_v2')
    filepath_ids = os.path.join(args.write_dir, 'ids_v2')
    np.save(filepath_emb, embeddings)
    np.save(filepath_ids, ids)
    # read(filepath_emb+'.npy')
    # read(filepath_ids+'.npy')


def read(path):
    data = np.load(path)
    print(len(data))


if __name__ == '__main__':
    args = get_args()
    dataset = MyDataset(args=args)
    dataloader = torchdata.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    textNet = TextModel(config_path='./bert-base-uncased/config.json',
                        model_path='./bert-base-uncased/pytorch_model.bin')
    textNet.cuda()
    embedding_seqs, id_seqs = extract_embedding()
    write(embedding_seqs, id_seqs)

    # read(os.path.join(args.write_dir, 'ids') + '.npy')
