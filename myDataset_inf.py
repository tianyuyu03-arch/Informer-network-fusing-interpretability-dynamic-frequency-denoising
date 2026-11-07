from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Chronological sequence windows with non-overlapping prediction targets."""

    def __init__(self, data, split, seq_len, label_len, pred_len,
                 train_size, validation_size, test_size):
        super().__init__()
        if split not in {'train', 'validation', 'test'}:
            raise ValueError("split must be 'train', 'validation', or 'test'.")
        if min(seq_len, pred_len, train_size, test_size) <= 0 or validation_size < 0:
            raise ValueError('Sequence and split sizes must be positive (validation may be zero).')
        if label_len < 0 or label_len > seq_len:
            raise ValueError('label_len must be between zero and seq_len.')

        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size

        if split == 'train':
            start = 0
            end = train_size + seq_len + pred_len - 1
            expected = train_size
        elif split == 'validation':
            start = train_size
            end = start + validation_size + seq_len + pred_len - 1
            expected = validation_size
        else:
            start = train_size + validation_size
            end = start + test_size + seq_len + pred_len - 1
            expected = test_size

        if expected == 0:
            raise ValueError('The selected split cannot be empty.')
        if len(data) < end:
            raise ValueError(f'Dataset needs at least {end} rows for the configured splits; got {len(data)}.')
        self.dataX = data[start:end]
        self.dataY = self.dataX

    def __len__(self):
        return len(self.dataX) - self.seq_len - self.pred_len + 1

    def __getitem__(self, item):
        if item < 0 or item >= len(self):
            raise IndexError(item)
        s_begin = item
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return self.dataX[s_begin:s_end], self.dataY[r_begin:r_end]
