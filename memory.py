import random
import copy

class Memory:
    def __init__(self, size=10000):
        self.buffer = [0] * size  # 真正用来存储四元组的buffer
        self.top = -1  # 用来记录最新存进来的四元组的位置(因为size有限，所以这个top在0~size之间循环)
        self.current_size = 0  # 用来记录现在buffer中存放着几个四元组
        self.max_size = size  # 顾名思义，最多能存几个

    def save(self, state, action, reward, state_):
        self.top = (self.top + 1) % self.max_size
        self.buffer[self.top] = (state, action, reward, state_)
        if self.current_size < self.max_size:
            self.current_size += 1

    def sample(self):
        ret_index = random.randint(0, self.current_size - 1)
        return self.buffer[ret_index]
