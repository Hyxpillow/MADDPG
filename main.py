from agent import Agent
from server import Server
import gym
import torch


# 在执行MADDPG算法之前进行相关变量的初始化
env = gym.make('Pendulum-v1')
M = 1000
MAX_EPISODE_LENGTH = 200
N = 1
GAMMA = 0.9
TAU = 0.01
RENDER = False
STATE_DIMENSION = env.observation_space.shape[0]
ACTION_DIMENSION = env.action_space.shape[0]

state_list = [0 for i in range(N)]
agent_list = [Agent(STATE_DIMENSION, ACTION_DIMENSION) for i in range(N)]
server = Server(N, STATE_DIMENSION, ACTION_DIMENSION)


# 真正开始执行MADDPG
for episode in range(M):
    state = env.reset()
    state_list = [state] * N
    total_reward = 0
    for step in range(MAX_EPISODE_LENGTH):
        if RENDER:
            env.render()
        for i, agent_i in enumerate(agent_list):
            state = torch.FloatTensor(state_list[i])
            action = agent_i.actor(state)
            action = action.detach()

            state_, reward, done, info = env.step(action)
            agent_i.store(state_list[i], action, reward, state_)
            state_list[i] = state_

            total_reward += reward

        if step == MAX_EPISODE_LENGTH - 1:
            print('Episode: ', episode, ' Reward: %i' % total_reward)
            if total_reward > -300:
                RENDER = True

        if agent_list[0].memory.current_size < 10000:
            continue

        for i, agent_i in enumerate(agent_list):
            state, action, reward, state_ = agent_i.sample()

            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            reward = torch.FloatTensor([reward])
            state_ = torch.FloatTensor(state_)

            action_ = agent_i.target_actor(state_)
            y_ = reward + GAMMA * agent_i.target_critic(state_, action_)
            y = agent_i.critic(state, action)
            theta = agent_i.update_critic(y, y_)
            # server.recv(i, y, y_)
            # server.update_critic()

            action = agent_i.actor(state)
            loss = agent_i.critic(state, action)
            agent_i.update_actor(loss)

        # theta = server.send()

        for agent_i in agent_list:
            agent_i.set_critic_parameter(theta)
            agent_i.update_target_actor(TAU)
            agent_i.update_target_critic(TAU)

            # print(dict(agent_i.critic_network.named_parameters()))


env.close()
