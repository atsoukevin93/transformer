from base.base_data_loader import BaseDataLoader
from datasets import TransformerDataset


class TransformerDataLoader(BaseDataLoader):
    """
        Sequencing reads data loading using BaseDataLoader
    """
    def __init__(self, data_dir, k, batch_size, max_seq_len, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.k = k
        # self.dataset = TransformerDataset(data_dir, k, mode, suffix, n_reads)
        self.dataset = TransformerDataset(data_dir, max_seq_len=max_seq_len)
        print(f"datadir: {data_dir}")
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)