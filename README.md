# 🚕 Smart Taxi Simulation with Q-Learning and Animation 
This repository contains a Python implementation of a Smart Taxi environment with Q-Learning, visualized using Matplotlib animations. The taxi learns to pick up passengers and drop them off at their destinations while avoiding obstacles.

# 📝 Features
6x6 grid environment with 6 predefined passenger locations: R, G, Y, B, K, L
Obstacles are placed in the grid to make navigation more challenging
6 actions: south, north, east, west, pickup, dropoff
Q-Learning agent for training the taxi
Training visualization: Reward curves and moving taxi animation



!pip install numpy matplotlib -q

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from IPython.display import HTML

# CONFIG

@dataclass
class Config:
    episodes: int = 20000
    max_steps: int = 50
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995

CONFIG = Config()

# ENVIRONMENT

class TaxiEnv:
    def __init__(self):
        self.grid_size = 6
        self.locs = [(0,0), (0,5), (5,0), (5,4), (2,2), (3,1)]
        self.loc_names = ['R','G','Y','B','K','L']
        
        # Engeller
        self.obstacles = [(1,1),(2,3),(3,3),(4,2)]
        
        # State space: taxi_pos(36) * passenger(6) * dest(6) * in_taxi(2) = 2592
        self.state_space = 36*6*6*2
        self.action_space = 6
        
        self.reset()
    
    def reset(self):
        self.taxi_pos = [random.randint(0,5), random.randint(0,5)]
        while tuple(self.taxi_pos) in self.obstacles:
            self.taxi_pos = [random.randint(0,5), random.randint(0,5)]
        
        self.passenger = random.randint(0,5)
        self.dest = random.randint(0,5)
        while self.dest == self.passenger:
            self.dest = random.randint(0,5)
        
        self.in_taxi = False
        return self._get_state()
    
    def _get_state(self):
        taxi = self.taxi_pos[0]*self.grid_size + self.taxi_pos[1]  # 0..35
        in_taxi = 1 if self.in_taxi else 0
        passenger = 0 if self.in_taxi else self.passenger       # 0..5
        dest = self.dest                                      # 0..5
        # State encoding
        state = taxi*6*6*2 + passenger*6*2 + dest*2 + in_taxi
        return state
    
    def step(self, action):
        reward = -1
        done = False
        
        moves = [(1,0),(-1,0),(0,1),(0,-1)]  # south, north, east, west
        
        if action < 4:
            dr, dc = moves[action]
            nr, nc = self.taxi_pos[0]+dr, self.taxi_pos[1]+dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and (nr,nc) not in self.obstacles:
                self.taxi_pos = [nr,nc]
            else:
                reward = -5
        
        elif action == 4:  # pickup
            if not self.in_taxi and self.taxi_pos==list(self.locs[self.passenger]):
                self.in_taxi = True
                reward = 20
            else:
                reward = -10
        
        elif action == 5:  # dropoff
            if self.in_taxi and self.taxi_pos==list(self.locs[self.dest]):
                reward = 100
                done = True
            else:
                reward = -10
        
        return self._get_state(), reward, done

# AGENT

class QAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.state_space, env.action_space))
    
    def train(self):
        rewards = []
        for ep in range(CONFIG.episodes):
            state = self.env.reset()
            total_reward = 0
            for _ in range(CONFIG.max_steps):
                if random.random() < CONFIG.epsilon:
                    action = random.randint(0,5)
                else:
                    action = np.argmax(self.q_table[state])
                
                next_state, reward, done = self.env.step(action)
                
                # Q-learning update
                self.q_table[state, action] += CONFIG.alpha * (
                    reward + CONFIG.gamma*np.max(self.q_table[next_state]) - self.q_table[state,action]
                )
                
                state = next_state
                total_reward += reward
                if done:
                    break
            
            CONFIG.epsilon = max(CONFIG.epsilon_min, CONFIG.epsilon*CONFIG.epsilon_decay)
            rewards.append(total_reward)
            
            if ep % 5000 == 0:
                print(f"Episode {ep} | Avg Reward: {np.mean(rewards[-100:]):.2f}")
        return rewards
    
    def test(self):
        state = self.env.reset()
        path = []
        for _ in range(CONFIG.max_steps):
            path.append(tuple(self.env.taxi_pos))
            action = np.argmax(self.q_table[state])
            state, _, done = self.env.step(action)
            if done:
                break
        return path

# GRAFİK & ANIMASYON

def plot_rewards(rewards):
    plt.figure(figsize=(10,4))
    plt.plot(rewards, alpha=0.4)
    avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
    plt.plot(avg)
    plt.title("Training Performance")
    plt.grid()
    plt.show()

def animate(env, path):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-0.5, env.grid_size-0.5)
    ax.set_ylim(-0.5, env.grid_size-0.5)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Engeller
    for r,c in env.obstacles:
        ax.add_patch(plt.Rectangle((c-0.5, env.grid_size-1-r-0.5),1,1,color='gray'))
    
    # Duraklar
    for i,(r,c) in enumerate(env.locs):
        ax.text(c, env.grid_size-1-r, env.loc_names[i], ha='center', va='center', fontsize=14)
    
    taxi, = ax.plot([], [], 'o', markersize=15, color='yellow')
    line, = ax.plot([], [])
    
    def update(i):
        x = [p[1] for p in path[:i+1]]
        y = [env.grid_size-1-p[0] for p in path[:i+1]]
        line.set_data(x,y)
        taxi.set_data([path[i][1]], [env.grid_size-1-path[i][0]])
        return taxi, line
    
    anim = FuncAnimation(fig, update, frames=len(path), interval=400)
    plt.close()
    return HTML(anim.to_jshtml())

# RUN

env = TaxiEnv()
agent = QAgent(env)

print("🚀 Eğitim başlıyor...")
rewards = agent.train()
plot_rewards(rewards)

print("🎬 Test...")
path = agent.test()
animate(env, path)
