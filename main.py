from data.data_loader import load_stock_data
from environment.trading_env import TradingEnvironment
from models.agent import DQNAgent
import numpy as np

"""
Ana eğitim döngüsünü başlatır.
"""
def train():
    data = load_stock_data()
    env = TradingEnvironment(data)
    state_size = 4
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    episodes = 10
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(batch_size)
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    train()