import numpy as np
import utils.data_preprocessing_util as dpu
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, data=None)
        # self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, data=dataset.data)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split, data=None, len_trunc_threshold=1000):
        if data is None:
            if split == 0.0:
                return None, None

            idx_full = np.arange(self.n_samples)

            np.random.seed(0)
            np.random.shuffle(idx_full)

            if isinstance(split, int):
                assert split > 0
                assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
                len_valid = split
            else:
                len_valid = int(self.n_samples * split)

            valid_idx = idx_full[0:len_valid]
            train_idx = np.delete(idx_full, np.arange(0, len_valid))

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            # turn off shuffle option which is mutually exclusive with sampler
            self.shuffle = False
            self.n_samples = len(train_idx)
        else:
            # np.random.seed(0)
            data["len"] = data.ref.map(dpu.get_sequence_within_special_tokens).map(len).array
            data = data.sample(frac=1, random_state=0).reset_index(drop=True)

            train_idx = np.array(data.index[data.len < len_trunc_threshold])

            # Downsampling the validation set
            valid_idx_data = data[data.len > len_trunc_threshold].sample(frac=0.2, random_state=0)
            valid_idx = np.array(valid_idx_data.index)

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            # turn off shuffle option which is mutually exclusive with sampler
            self.shuffle = False
            self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def get_vocab_size(self):
        return self.dataset.vocab_size

    def get_src_vocab_size(self):
        return self.dataset.src_vocab_size

    def get_trg_vocab_size(self):
        try:
            return self.dataset.trg_vocab_size
        except AttributeError:
            return self.dataset.src_vocab_size

    def get_block_size(self):
        return self.dataset.get_block_size()

    def get_embedding_weights(self):
        return self.dataset.embedding_weights

    def get_max_read_length(self):
        try:
            return self.dataset.max_read_length
        except AttributeError:
            return None

    def get_src_pad_idx(self):
        try:
            return self.dataset.kmertoi['<PAD>']
        except KeyError:
            return 0

    def get_trg_pad_idx(self):
        try:
            return self.dataset.trg_kmertoi['<PAD>']
        except AttributeError:
            return self.dataset.kmertoi['<PAD>']
        except KeyError:
            return None

    def get_src_eos_idx(self):
        try:
            return self.dataset.kmertoi['<EOS>']
        except KeyError:
            return None

    def get_src_bos_idx(self):
        try:
            return self.dataset.kmertoi['<BOS>']
        except KeyError:
            return None

    def get_trg_eos_idx(self):
        try:
            return self.dataset.trg_kmertoi['<EOS>']
        except AttributeError:
            return self.dataset.kmertoi['<EOS>']
        except KeyError:
            return None

    def get_trg_bos_idx(self):
        try:
            return self.dataset.trg_kmertoi['<BOS>']
        except AttributeError:
            return self.dataset.kmertoi['<BOS>']
        except KeyError:
            return None

    def get_fwd_idx(self):
        return self.dataset.kmertoi['<FWD>']

    def get_rev_idx(self):
        return self.dataset.kmertoi['<REV>']
