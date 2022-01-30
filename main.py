import agent
import server
import gym

# 在执行MADDPG算法之前进行相关变量的初始化
env = gym.make('CartPole-v0')
M = 1000
MAX_EPISODE_LENGTH = 200
N = 10
GAMMA = 0.05
state_list = [0 for i in range(N)]
agent_list = [agent.Agent() for i in range(N)]
center_server = server.Server()

# 真正开始执行MADDPG
for episode in range(M):
	state = env.reset()
	state_list = [state] * N
	for step in range(MAX_EPISODE_LENGTH):
		# env.render()
		for i, agent_i in enumerate(agent_list):
			action = agent_i.actor(state_list[i])
			state_, reward, done, info = env.step(action)
			agent_i.store(state_list[i], action, reward, state_)
			state_list[i] = state_
		for i, agent_i in enumerate(agent_list):
			state, action, reward, state_ = agent_i.sample()
			action_ = agent_i.target_actor(state_)
			y = reward + GAMMA * agent_i.target_critic(state_, action_)

			center_server.recv_agent()
		# if done: # 如果游戏结束，则跳出step循环，进入下一episode；否则，继续step
		# 	break
env.close() #结束游戏，关闭环境
