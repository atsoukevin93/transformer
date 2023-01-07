import pandas as pd


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        # self.metrics_data = pd.DataFrame(columns=['epoh', 'counts', 'average'])
        self.data_header = True
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def save_metrics(self, metrics_dict, step, output):
        row = pd.DataFrame(metrics_dict, index=[step])
        row.to_csv(
            output,
            header=self.data_header,
            compression='gzip',
            index=False,
            mode='a')
        self.data_header = False

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)