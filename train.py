from pacman import Pacman
import gym
import numpy as np
import random
from collections import deque

def process_frame(frame):
    mspacman_color = np.array([210, 164, 74]).mean()
    img = frame[1:176:2, ::2]    
    img = img.mean(axis=2)       
    img[img==mspacman_color] = 0 
    img = (img - 128) / 128 - 1  
    
    return np.expand_dims(img.reshape(88, 80, 1), axis=0)

def blend_images(images, blend):
    avg_image = np.expand_dims(np.zeros((88, 80, 1), np.float64), axis=0)

    for image in images:
        avg_image += image
        
    if len(images) < blend:
        return avg_image / len(images)
    else:
        return avg_image / blend

env = gym.make('MsPacman-v0')

state_size = (88, 80, 1)
action_size = env.action_space.n

player = Pacman(state_size, action_size)

episodes = 500
batch_size = 8
skip_start = 90  
total_time = 0   
all_rewards = 0  
blend = 4      
done = False

for e in range(episodes):
    total_reward = 0
    game_score = 0
    state = process_frame(env.reset())
    images = deque(maxlen=blend)
    images.append(state)
    
    for t in range(skip_start):
        env.step(0)
    
    for time in range(20000):
        env.render()
        total_time += 1
        
        if total_time % player.update_rate == 0:
            player.update_target_model()
        
        state = blend_images(images, blend)
        
        action = player.act(state)
        next_state, reward, done, _ = env.step(action)
        
        next_state = process_frame(next_state)
        images.append(next_state)
        next_state = blend_images(images, blend)
        
        player.remember(state, action, reward, next_state, done)
        
        state = next_state
        game_score += reward
        reward -= 1  
        total_reward += reward
        
        if done:
            all_rewards += game_score
            
            print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, time: {}, total time: {}"
                  .format(e+1, episodes, game_score, total_reward, all_rewards/(e+1), time, total_time))
            
            break
            
        if len(player.memory) > batch_size:
            player.replay(batch_size)

player.save('models/5k-memory_1k-games')