import torch 
from env import GridWorldEnv
from agent import DQN
import time 


env = GridWorldEnv()
model = DQN(n_actions = 4)
observation, info = env.reset()

for step in range(20):
    #environment observation to pytorch tensor
    state = torch.tensor(
        observation,
        dtype = torch.float32
    ).unsqueeze(0)

    #cnn predicts the q values
    q_values = model(state)

    #choose action with highest q value
    action = torch.argmax(q_values, dim = 1).item()

    print(
        f"Step: {step} |"
        f"Q-values: {q_values.detach().numpy()} |"
        f"Action: {action}"
    )

    #cnns chosen action to environment
    observation, reward, terminated, truncated, info = env.step(action)

    #render the new position
    env.render()
    time.sleep(0.5)
    
    if terminated or truncated:
        print("Epsiode finished!")
        observation, info = env.reset()

env.close()