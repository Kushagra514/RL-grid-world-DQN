# RL Grid World DQN

An educational reinforcement learning project that builds a small GridWorld and a Deep Q-Network (DQN) agent from first principles. The goal is to make the interaction loop visible: observe the world, choose an action, receive a reward and next observation, then learn from that experience.

The environment and learning algorithm are implemented directly with Gymnasium and PyTorch. High-level reinforcement learning libraries are intentionally avoided so the important pieces remain understandable.

## Project status

The repository currently contains an early GridWorld prototype. The environment has a discrete four-action interface, separate agent, goal, and hazard observation channels, randomized reset positions, rewards, and a Pygame display. The DQN network is a convolutional model; training infrastructure is being developed incrementally.

## Planned learning path

1. Make the environment API reproducible and well tested.
2. Separate environment, rendering, and interactive play.
3. Implement replay memory and a PyTorch Q-network.
4. Add the DQN update and training loop.
5. Evaluate learned policies and document experiments.

## License

This project is provided for learning and experimentation.
