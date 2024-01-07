import inspect
import random
from abc import ABC, abstractmethod


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
