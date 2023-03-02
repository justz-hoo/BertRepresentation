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
    embedding_with_id = np.concatenate((id_seq.reshape(8640, 1), final_embedding), axis=1)
    return final_embedding, id_seq, embedding_with_id


def write(embeddings, ids, embedding_with_id):
    filepath_emb = os.path.join(args.write_dir, 'pure_embedding')
    filepath_ids = os.path.join(args.write_dir, 'idx')
    filepath_emb_id = os.path.join(args.write_dir, 'embedding')
    np.save(filepath_emb, embeddings)
    np.save(filepath_ids, ids)
    np.save(filepath_emb_id, embedding_with_id)


def read(path):
    data = np.load(path)
    return data


if __name__ == '__main__':
    args = get_args()
    dataset = MyDataset(args=args)
    dataloader = torchdata.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    textNet = TextModel(config_path='./bert-base-uncased/config.json',
                        model_path='./bert-base-uncased/pytorch_model.bin')
    textNet.cuda()
    embedding_seqs, id_seqs, embedding_with_id = extract_embedding()
    write(embedding_seqs, id_seqs, embedding_with_id)

    # idx = read(os.path.join(args.write_dir, 'ids_v2') + '.npy')
    # pure_embedding = read(os.path.join(args.write_dir, 'Bert_embedding_v2') + '.npy')
    # embedding_with_idx = read(os.path.join(args.write_dir, 'embedding' + '.npy'))

    print('finished')
