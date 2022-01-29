class Server:
    # 具有的方法包括：
    #    构造函数
    #    从agent接受Loss值
    #    对所有agent的Loss值做平均，然后更新critic网络
    #    把新的critic网络参数反馈给每个agent
    def __init__(self):
        # 拥有的成员变量包括:
        #    一个critic网络
        #    一个用来存放所有Loss值的buffer
        pass

    def recv_agent(self):
        pass

    def send_agent(self):
        pass

    def update_critic(self):
        pass

