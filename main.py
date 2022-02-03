from agent import Agent
from server import Server
import gym
import torch


# 在执行MADDPG算法之前进行相关变量的初始化


env = gym.make('Pendulum-v0')
M = 1000
MAX_EPISODE_LENGTH = 2000
N = 1
GAMMA = 0.9
TAU = 0.01
<<<<<<< HEAD
=======
RENDER = False
>>>>>>> fffa034c7ea150a385f43e955006ba8d12cdc09b
STATE_DIMENSION = env.observation_space.shape[0]
ACTION_DIMENSION = env.action_space.shape[0]

state_list = [0 for i in range(N)]
agent_list = [Agent(STATE_DIMENSION, ACTION_DIMENSION) for i in range(N)]
server = Server(N, STATE_DIMENSION, ACTION_DIMENSION)





a_bound = env.action_space.high
a_low_bound = env.action_space.low
var = 3
# 真正开始执行MADDPG
cnt = 3
for episode in range(M):
    torch.autograd.Variable
    state = env.reset()
    state_list = [state] * N
<<<<<<< HEAD
    env.render()
    
    for step in range(MAX_EPISODE_LENGTH):
        for i, agent_i in enumerate(agent_list):
            action = agent_i.actor(state_list[i])
            action = np.clip(np.random.normal(action, var), a_low_bound, a_bound)
            state_, reward, done, info = env.step(action)
            agent_i.store(state_list[i], action, reward, state_)
            state_list[i] = state_
        var *= 0.9995 # decay the exploration controller factor
        for i, agent_i in enumerate(agent_list):
            state, action, reward, state_ = agent_i.sample()
            
            action_ = agent_i.target_actor(state_)
            y_ = reward + GAMMA * agent_i.target_critic(state_, action_)
            y = agent_i.critic(state, action)
            
            #server.recv(i, y, y_)
            #server.update_critic()




            y = torch.tensor(y, requires_grad=True, dtype=torch.float)
            y_ = torch.tensor(y_, requires_grad=True, dtype=torch.float)
            loss = F.mse_loss(y_, y)
            server.critic_network.optimizer.zero_grad()
            loss.backward()
            optimizer = torch.optim.Adam(server.critic_network.parameters(), lr=0.001)
            #server.critic_network.optimizer.step()
            optimizer.step()
            #theta = server.send()
            #if step  3:
                #print(theta)

=======
    total_reward = 0
    for step in range(MAX_EPISODE_LENGTH):
        if RENDER:
            env.render()
        for i, agent_i in enumerate(agent_list):  # 这个for循环与环境交互，产生新的(s,a,r,s_)并存入memory
            state = torch.FloatTensor(state_list[i])
            action = agent_i.actor(state)
            action = action.detach()

            state_, reward, done, info = env.step(action)
            agent_i.store(state_list[i], action, reward, state_)
            state_list[i] = state_

            total_reward += reward

        if step == MAX_EPISODE_LENGTH - 1:  # 这个if用于输出提示和控制是否渲染动画
            print('Episode: ', episode, ' Reward: %i' % total_reward)
            if total_reward > -300:
                RENDER = True

        if agent_list[0].memory.current_size < 10000:  # memory没装满时不学习，直接continue
            continue

        for i, agent_i in enumerate(agent_list):  # 开始学习
            state, action, reward, state_ = agent_i.sample()  # 存在buffer里的都是numpy

            state = torch.FloatTensor(state)  # 转换成tensor类型
            action = torch.FloatTensor(action)
            reward = torch.FloatTensor([reward])
            state_ = torch.FloatTensor(state_)

            action_ = agent_i.target_actor(state_)  # 本地更新critic
            y_ = reward + GAMMA * agent_i.target_critic(state_, action_)
            y = agent_i.critic(state, action)
            agent_i.update_critic(y, y_)

            theta = agent_i.get_critic_parameter()  # 把本地的critic网络参数传给server进行平均计算
            server.recv(i, theta)
>>>>>>> fffa034c7ea150a385f43e955006ba8d12cdc09b

            action = agent_i.actor(state)  # 本地更新actor
            loss = agent_i.critic(state, action)
            agent_i.update_actor(loss)

<<<<<<< HEAD
            
            theta = server.send()
            if step > 3:
                print(theta)
            

        theta = server.send()

        for agent_i in agent_list:
            if cnt > 0:
                print(theta)
            cnt -= 1
=======
        server.update_critic()  # for循环结束，说明server已经收到了所有agent的critic参数，开始进行平均
        theta = server.send()  # server返回计算好的平均参数
        # print(theta)

        for agent_i in agent_list:  # 更新target_actor和target_critic
>>>>>>> fffa034c7ea150a385f43e955006ba8d12cdc09b
            agent_i.set_critic_parameter(theta)
            agent_i.update_target_actor(TAU)
            agent_i.update_target_critic(TAU)

env.close()
