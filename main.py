import time
import gym
env = gym.make('CartPole-v0') #导入CartPole-v0游戏
for episode in range(1000):  #跑Episodes幕，一幕相当于一次完整游戏
	state = env.reset() #初始化环境，得到第一步状# 态
	for step in range(200): #每一局游戏至多跑STEPS步
		env.render()
		action = 0 #强化学习算法，根据状态生成行动
		state, reward, done, info = env.step(action) #根据action将游戏向前推进
		# 把 (状态 奖励 动作 下一个状态) 存入 memory
		# if step % 50 == 0:
            # learn()
		if done: # 如果游戏结束，则跳出step循环，进入下一episode；否则，继续step
			break
env.close() #结束游戏，关闭环境
