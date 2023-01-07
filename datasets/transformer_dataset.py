import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import data_preprocessing_util as dpu
from utils import exists


class TransformerDataset(Dataset):

    def __init__(self, data_dir, k, mode, suffix, full_src=False, max_seq_len=None):
        self.k = k
        # mode = 'words'
        self.mode = mode
        self.suffix = suffix
        self.full_src = full_src
        data = pd.read_csv(
            f'{data_dir}ref_reads_dataset_{suffix}.csv',
            # usecols=["ref", "read", "cigar"],
            # chunksize=100,
            compression='gzip'
        # ).get_chunk(100)
        )

        if not exists(max_seq_len):
            raise "provide max_seq_len"
        else:
            self.block_size = max_seq_len

        # data = data.sample(frac=0.12, random_state=0).reset_index(drop=True)
        # data = data[np.logical_and(data.read.map(len) < 500, data.start < 1)].reset_index(drop=True)
        data = data[data.read.map(len) < 200].reset_index(drop=True)
        # data = data.drop_duplicates().reset_index(drop=True)
        # data['mutation'] = data['mutation'].map(dpu.remove_insertion_decoration)
        # print(data)

        # vocab = dpu.define_data_vocab(data)
        # vocab.insert(0, '<PAD>')
        self.data = data
        src_vocab = dpu.load_vocabulary_to_array(f"data/processed/src_vocabulary_max_{mode}_{suffix}_{k}.txt")
        self.src_vocab_size = len(src_vocab)
        try:
            trg_vocab = dpu.load_vocabulary_to_array(f"data/processed/trg_vocabulary_max_{mode}_{suffix}_{k}.txt")
        except FileNotFoundError:
            trg_vocab = None
        self.kmertoi = {kmer: i for i, kmer in enumerate(src_vocab)}
        self.itokmer = {i: kmer for i, kmer in enumerate(src_vocab)}

        if exists(trg_vocab):
            self.trg_kmertoi = {kmer: i for i, kmer in enumerate(trg_vocab)}
            self.trg_itokmer = {i: kmer for i, kmer in enumerate(trg_vocab)}
            self.trg_vocab_size = len(trg_vocab)
        else:
            self.trg_kmertoi = None
            self.trg_itokmer = None
            self.trg_vocab_size = None

        del data
        del src_vocab
        del trg_vocab

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # sample_index, timeseries_index = self.index_map[idx]
        # grab a chunk of (block_size + 1) characters from the data
        full_x = self.data.ref[idx]
        full_y = self.data.read[idx]
        cigar = self.data.cigar[idx]

        # full_x = self.data.kmer[idx]
        # full_y = self.data.mutation[idx]

        # full_x_decorated = '<BOS>' + ' ' + dpu.sequence_to(full_x, self.k, self.mode) + ' ' + '<EOS>'
        full_x_decorated = dpu.sequence_to(full_x, self.k, self.mode)
        # full_y = '<BOS>' + ' ' + dpu.remove_insertion_decoration(dpu.read_sequence_to_words(full_y, self.data.ref[
        # idx], cigar, self.k)) + ' ' + '<EOS>'
        # full_y = '<BOS>' + ' ' + dpu.sequence_to(full_y, self.k, self.mode) + ' ' + '<EOS>'
        if self.full_src:
            full_y_decorated = '<BOS>' + ' ' + dpu.sequence_to(full_y, self.k, self.mode) + ' ' + '<EOS>'
        else:
            full_y_decorated = '<BOS>' + ' ' + dpu.remove_insertion_decoration(
                dpu.read_sequence_to_words(full_y, full_x, cigar, self.k)) + ' ' + '<EOS>'

        # print(f'{full_x_decorated}\n{full_y_decorated}')

        # full_x = dpu.split_sequence_with_special_tokens(full_x, self.k, self.mode)
        # full_y = dpu.split_sequence_with_special_tokens(full_y, self.k, self.mode)

        # full_x, full_y = decorate_sequences(full_x, full_y, padding_side='left')
        full_x_decorated = dpu.pad_kmer_sentence(full_x_decorated, self.block_size, 'right')
        full_y_decorated = dpu.pad_kmer_sentence(full_y_decorated, self.block_size+1, 'right')
        # full_y = dpu.pad_kmer_sentence(full_y, self.max_read_length, 'right')

        # encode every character to an integer
        dix_x = [self.kmertoi[s] for s in full_x_decorated.split()]

        if exists(self.trg_kmertoi):
            dix_y = [self.trg_kmertoi[s] for s in full_y_decorated.split()]
        else:
            dix_y = [self.kmertoi[s] for s in full_y_decorated.split()]

        src = torch.tensor(dix_x, dtype=torch.long)
        trg = torch.tensor(dix_y[:-1], dtype=torch.long)
        trg_y = torch.tensor(dix_y[1:], dtype=torch.long)
        # trg = torch.tensor([self.kmertoi['<PAD>']] + dix_y[:-1], dtype=torch.long)
        # trg_y = torch.tensor(dix_y[1:] + [self.kmertoi['<PAD>']], dtype=torch.long)

        # pad_idx = self.kmertoi['<PAD>']
        # trg[trg == pad_idx] = -100
        # trg_y[trg_y == pad_idx] = -100
        # maxlen =
        return src, trg, trg_y

    def get_block_size(self):
        return self.block_size


