from abc import ABC, abstractmethod


class BasicClass(ABC):
    _id = 0

    def __init__(self, process_id, function_id, speed=0):
        self.__id = self.next_id()
        self.__speed = speed
        self.__process_id = process_id
        self.__function_id = function_id

    @classmethod
    def next_id(cls):
        current_id = cls._id
        cls._id += 1
        return current_id

    @property
    def id(self):
        return self.__id

    @property
    def process(self):
        return self.__process_id

    @property
    def function(self):
        return self.__function_id

    @property
    def speed(self):
        return self.__speed

    @abstractmethod
    def work(self, *args):
        pass

    @abstractmethod
    def send(self, *args):
        pass

    @abstractmethod
    def receive(self, *args):
        pass


if __name__ == '__main__':
    class CC(BasicClass):
        def send(self, *args):
            pass

        def receive(self, *args):
            pass

        def work(self, ab):
            print(ab)

        _id = 0

        def __init__(self):
            super().__init__(1, 56)
            print(f"CC {self.id}")
            print(f"CCprocess {self.function}")


    class DD(BasicClass):
        def send(self, *args):
            pass

        def receive(self, *args):
            pass

        def work(self, *args):
            pass

        _id = 0

        def __init__(self):
            super().__init__(2, 3)
            print(f"DDprocess {self.process}")


    a = CC()
    a.work(2)
    b = CC()
    c = CC()
    d = DD()
    d = DD()
    d = DD()
    # print(f"a {id(a._id)},{a._id[0]},{id(a.__id1)},{a.__id1}")
    # print(f"BC {id(BasicClass._id)},{CC._id[0]},{id(BasicClass.__id1)},{BasicClass.__id1}")
    # print(f"CC {id(CC._id)},{CC._id[0]},{id(CC.__id1)},{CC.__id1}")
