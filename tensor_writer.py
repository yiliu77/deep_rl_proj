from torch.utils.tensorboard import SummaryWriter


class TensorWriter(SummaryWriter):
    def __init__(self, path):
        super(TensorWriter, self).__init__(path)
        self.train_info_buffer = []
        self.train_iteration = None

    def add_train_step_info(self, train_info, i):
        self.train_info_buffer.append(train_info)
        self.train_iteration = i

    def write_train_step(self):
        keys = self.train_info_buffer[0].keys()

        for k in keys:
            total = 0
            for i in range(len(self.train_info_buffer)):
                total += self.train_info_buffer[i][k]
            self.add_scalar(k, total / len(self.train_info_buffer), self.train_iteration)
        self.train_info_buffer = []
        self.train_iteration = None
