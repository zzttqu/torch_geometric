import inspect
import random
from abc import ABC, abstractmethod

import torch


class BasicClass(ABC):
    class_id = 0

    def __init__(self, process_id):
        """
        父类不建议使用私有属性，因为会导致子类无法继承
        Args:
            process_id: 工序id
        """
        self._id = self.next_id()
        self._process_id = process_id

    @classmethod
    def next_id(cls):
        """
        生成该类的id
        Returns:
            int: 生成的id
        """
        current_id = cls.class_id
        cls.class_id += 1
        return current_id

    @classmethod
    def reset_class_id(cls):
        """
        重置该类的id
        Returns:

        """
        cls.class_id = 0

    @property
    def id(self):
        return self._id

    @property
    def process(self):
        return self._process_id

    @abstractmethod
    def work(self, *args):
        pass

    @abstractmethod
    def send(self, *args):
        pass

    @abstractmethod
    def receive(self, *args):
        pass

    @abstractmethod
    def reset(self, *args):
        pass


if __name__ == '__main__':
    from torch.distributions import Categorical
    import torch.nn.functional as F

    edge_names = ["cell2center", "cell2storage", "storage2center", "storage2cell", "center2cell"]
    # logger.info(self.storage_id_relation)
    a = Categorical(logits=torch.tensor([[2, 0], [0.2, 0.7]]))
    print(F.softmax(torch.tensor([[0.2, 0.7], [2, 0]]), dim=1))
    e = Categorical(logits=torch.tensor([2, 0]))
    print(F.softmax(torch.tensor([2.0, 0.0]), dim=0))
    c = Categorical(logits=torch.tensor([0.2, 0.7]))
    a0 = a.sample()
    e0 = e.sample()
    c0 = c.sample()
    # 1.22 = > 0.37754
    # 2.01 = > 0.62245

    proba = a.log_prob(a0)
    probe = e.log_prob(e0)
    probc = c.log_prob(c0)
    print(a0, proba)
    print(e0, probe)
    print(c0, probc)
    raise SystemExit
    a1 = torch.tensor([2, 4, 6, 5])
    _a1 = a1.clone()
    b1 = [torch.tensor([[1, 2], [2, 3], [0, 4]]), torch.tensor([[4, 2], [5, 3], [6, 6]])]
    for b in b1:
        condition = b[:, 1] == _a1.view(-1, 1)
        result = torch.where(condition, b[:, 0], torch.tensor(-1))
        nnn_a = torch.nonzero(condition)[:, 0]
        nnn_c = torch.nonzero(condition)[:, 1]

        _a1[nnn_a] = -1
        # a = torch.nonzero(torch.isin(b1, a1)).flatten()
        # for pp, qq in zip(nnn_b, nnn_a):
        #     print(b[pp, 1], a1[qq])


    #
    # id22 = torch.where(torch.eq(b1, a1[a][0]))
    # print(torch.eq(b1, a1[a][0]))
    # cc = a1.clone()
    # cc[0] = 0
    # # b = torch.where(torch.tensor([2, 4]), torch.tensor([1, 2, 3]))
    # print(id22)

    class CC(BasicClass):
        def reset(self, *args):
            pass

        def send(self, *args):
            pass

        def receive(self, *args):
            pass

        def work(self, ab):
            pass

        id = 0

        def __init__(self):
            super().__init__(random.randint(1, 100), 1)
            print(self._id)
            print(f"CCprocess {self.process},{id(self.process)}")


    class DD(BasicClass):
        def send(self, *args):
            pass

        def receive(self, *args):
            pass

        def work(self, *args):
            pass

        id = 0

        def __init__(self):
            super().__init__(2, 3)
            print(f"DDprocess {self.process},{id(self.process)}")


    a = CC()
    a.work(2)
    print(a.__dict__)
    print(a.id)
    b = CC()
    c = CC()
    d = DD()
    d = DD()
    d = DD()
    # print(f"a {id(a.id)},{a.id[0]},{id(a.__id1)},{a.__id1}")
    # print(f"BC {id(BasicClass.id)},{CC.id[0]},{id(BasicClass.__id1)},{BasicClass.__id1}")
    # print(f"CC {id(CC.id)},{CC.id[0]},{id(CC.__id1)},{CC.__id1}")
