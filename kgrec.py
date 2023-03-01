import numpy as np
import pandas as pd
from collections import defaultdict
import glob


class KGRec:
    def __init__(self, root_path):
        self.root_path = root_path
        self.item_num = 8640
        self.user_num = 5199

    def count_times(self):
        songs_count = defaultdict(int)
        seq_data = self.read_seq_data(self.root_path + '/implicit_lf_dataset.csv', '\t')
        items = list(seq_data['items'])
        # print(min(items))
        # print(max(items))
        for i in items:
            songs_count[i] += 1
        print('歌曲长度为', len(songs_count))
        print('min_count', min(songs_count.values()))
        print('max_count', max(songs_count.values()))
        # 歌曲出现的频次均大于10，所以不用过滤
        return songs_count

    def fileter_custom_tims(self, freq_num=10):
        pass

    def read_seq_data(self, path, sep):
        seq_data = pd.read_csv(path, sep=sep, header=None, usecols=[0, 1], names=['users', 'items'])
        return seq_data

    def read_info(self, option='descriptions'):
        data_dict = defaultdict(str)
        if option == 'tags':
            fnames = glob.glob(self.root_path + '/tags/*.txt')
            fnames = [fname.replace('\\', '/') for fname in fnames]
            for fname in fnames:
                idx = int(fname.split('/')[-1][:-4])
                f = open(fname, encoding='utf-8')
                datas = f.readlines()
                data = ''
                for i in datas:
                    data += i
                data_dict[idx] = data
        elif option == 'descriptions':
            fnames = glob.glob(self.root_path + '/descriptions/*.txt')
            fnames = [fname.replace('\\', '/') for fname in fnames]
            for fname in fnames:
                idx = int(fname.split('/')[-1][:-4])
                f = open(fname, encoding='utf-8')
                datas = f.readlines()
                data = ''
                for i in datas:
                    data += i
                data_dict[idx] = data
        else:
            print('please choose between tags and descriptions')
            return
        return data_dict

    def write_seqs(self):
        seq_data = self.read_seq_data(self.root_path + '/implicit_lf_dataset.csv', '\t')
        users = seq_data['users']
        items = list(seq_data['items'])
        users = pd.factorize(users)[0]
        users_new = [user + 1 for user in users]
        df = pd.DataFrame(zip(users_new, items), columns=['users', 'items'])
        df.to_csv(self.root_path + 'Seq.txt', sep=' ', index=False)

    def write_info(self):
        tags_dict = self.read_info(option='tags')
        des_dict = self.read_info(option='descriptions')
        idx_list = list(range(1, self.item_num + 1))
        des_list = []
        tag_list = []
        for i in range(1, self.item_num + 1):
            if i in tags_dict:
                tag_list.append(tags_dict[i])
            else:
                tag_list.append('')
            if i in des_dict:
                des_list.append(des_dict[i])
            else:
                des_list.append('')
        df = pd.DataFrame(zip(idx_list, des_list, tag_list), columns=['idx', 'des', 'tag'])
        df.to_csv(self.root_path + 'Songs.txt', sep=',', index=False)


if __name__ == '__main__':
    kgrec = KGRec('./kgrec_songs/KGRec-music')
    # 写KGREc数据集的info信息
    kgrec.write_info()
    # 写KGRec数据集的(user, item)交互信息
    # kgrec.write_seqs()
    # 统计每首歌出现的频次
    # kgrec.count_times()