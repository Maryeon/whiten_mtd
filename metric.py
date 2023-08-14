import torch
import logging
import time, datetime
from collections import defaultdict, deque
import utils
import torch.distributed as dist
from itertools import islice


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=None, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def sync(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=torch.device("cuda:"+str(torch.cuda.current_device())))
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=False)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, logger, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def sync(self):
        for meter in self.meters.values():
            meter.sync()

    def log_every(self, iterable, log_freq, header=None, iterations=None):
        iterations = len(iterable) if iterations is None else iterations
        if self.logger is None:
            for i, obj in enumerate(islice(iterable, 0, iterations)):
                yield i, obj
            return

        header = '' if header is None else header

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(iterations))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                'Iter: [{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'iter time: {time}',
                'data time: {data}',
                'gpu mem: {memory:.0f}MB'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'iter time: {time}',
                'data time: {data}'
            ])
        MB = 1024.0 * 1024.0
        for i, obj in enumerate(islice(iterable, 0, iterations)):
            data_time.update(time.time() - end)
            yield i, obj
            iter_time.update(time.time() - end)
            if i == iterations - 1 or i % log_freq == 0:
                eta_seconds = iter_time.global_avg * (iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.logger.info(log_msg.format(
                        i+1, iterations, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.memory_reserved() / MB))
                else:
                    self.logger.info(log_msg.format(
                        i+1, iterations, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        end_msg = self.delimiter.join([
            header,
            'Total time: {0} ({1:.4f} s / it)'
        ])
        self.logger.info(end_msg.format(total_time_str, total_time / iterations))


class MetricScorer:

    def __init__(self, k=0):
        self.k = k

    def score(self, sorted_labels):
        return 0.0

    def getLength(self, sorted_labels):
        length = self.k
        if length > len(sorted_labels) or length <= 0:
            length = len(sorted_labels)
        return length

    def name(self):
        if self.k > 0:
            return "%s@%d" % (self.__class__.__name__.replace("Scorer",""), self.k)
        return self.__class__.__name__.replace("Scorer","")

    def setLength(self, k):
        self.k = k;


class APScorer(MetricScorer):
 
    def __init__(self, k=0):
        MetricScorer.__init__(self, k)    

    def score(self, sorted_labels):
        length = self.getLength(sorted_labels)
        nr_relevant = len([x for x in sorted_labels[:length] if x > 0])
        if nr_relevant == 0:
            return 0.0
        
        ap = 0.0
        rel = 0
        
        for i in range(length):
            lab = sorted_labels[i]
            if lab > 0:
                rel += 1
                ap += float(rel) / (i+1.0)
        ap /= nr_relevant

        return ap
