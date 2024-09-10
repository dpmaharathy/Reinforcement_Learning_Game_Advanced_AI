import pygame
import common.game_constants as game_constants
import common.game_state as game_state

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class KeyboardController:
    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        keys = pygame.key.get_pressed()
        action = game_state.GameActions.No_action
        if keys[pygame.K_LEFT]:
            action = game_state.GameActions.Left
        if keys[pygame.K_RIGHT]:
            action = game_state.GameActions.Right
        if keys[pygame.K_UP]:
            action = game_state.GameActions.Up
        if keys[pygame.K_DOWN]:
            action = game_state.GameActions.Down
    
        return action


# class AIController:
# ### ------- You can make changes to this file from below this line --------------
#     def __init__(self) -> None:
#         # Add more lines to the constructor if you need
#         pass


#     def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
#         # This function should select the best action at a given state
        
#         # A wrong example (just so you can compile and check)
#         return game_state.GameActions.Right
    
#     def TrainModel(self):
#         # Complete this function
#         epochs = 10 # You might want to change the number of epochs
#         for _ in range(epochs):
#             state = game_state.GameState()
#             # Explore the state by updating it

#             action = self.GetAction(state) # Select best action
#             obs = state.Update(action) # obtain the observation made due to your action

#             # You must complete this function by
#             # training your model on the explored state,
#             # using a suitable RL algorithm, and
#             # by appropriately rewarding your model on 
#             # the state that it lands

#             pass
#         pass
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(18, 128)  # Assuming 18 features in the state

        # self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class AIController:
    def __init__(self) -> None:
        self.state_dim = 2 + (2 * game_constants.ENEMY_COUNT)  # Player (x,y) and Goal (x,y), and all enemies (x,y)
        self.action_dim = len(game_state.GameActions)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Epsilon for exploration
        self.epsilon_min = 0.01  # Minimum epsilon
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=2000)  # Replay memory

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the policy network and the target network
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy weights to target net
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def GetAction(self, state: game_state.GameState) -> game_state.GameActions:
        # Convert the game state into a feature vector
        state_vector = self._extract_state_vector(state)

        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.choice(list(game_state.GameActions))  # Exploration
        else:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action_idx = torch.argmax(q_values).item()  # Exploitation

            # Return the corresponding action based on the predicted q-values
            return list(game_state.GameActions)[action_idx]

    def _extract_state_vector(self, state: game_state.GameState):
        # Extract a vector representing the current game state
        state_vector = [
            state.PlayerEntity.entity.x / game_constants.GAME_WIDTH,  # Normalize player x
            state.PlayerEntity.entity.y / game_constants.GAME_HEIGHT,  # Normalize player y
            state.GoalLocation.x / game_constants.GAME_WIDTH,  # Normalize goal x
            state.GoalLocation.y / game_constants.GAME_HEIGHT  # Normalize goal y
        ]
        # Add all enemies' positions (normalized)
        for enemy in state.EnemyCollection:
            state_vector.append(enemy.entity.x / game_constants.GAME_WIDTH)
            state_vector.append(enemy.entity.y / game_constants.GAME_HEIGHT)

        return np.array(state_vector)

    def TrainModel(self):
        num_episodes = 500
        max_steps = 1000

        for episode in range(num_episodes):
            state = game_state.GameState()  # Reset game state at the beginning of each episode
            total_reward = 0

            for step in range(max_steps):
                action = self.GetAction(state)
                prev_state_vector = self._extract_state_vector(state)

                # Perform the action and get the next state and reward
                observation = state.Update(action)
                next_state_vector = self._extract_state_vector(state)

                # Reward function: +1 for reaching the goal, -1 for getting attacked, small negative reward for each step
                reward = -0.01  # Time penalty to encourage shorter paths
                if observation == game_state.GameObservation.Reached_Goal:
                    reward = 1.0
                elif observation == game_state.GameObservation.Enemy_Attacked:
                    reward = -1.0

                total_reward += reward
                done = observation in [game_state.GameObservation.Reached_Goal, game_state.GameObservation.Enemy_Attacked]

                # Store the transition in memory
                self.memory.append((prev_state_vector, action.value, reward, next_state_vector, done))

                # Train the model on a batch of experiences
                if len(self.memory) > self.batch_size:
                    self._replay()

                # Update state
                if done:
                    break

            # Update epsilon to reduce exploration over time
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

            # Update the target network periodically
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def _replay(self):
        # Sample a batch from the replay memory
        minibatch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q values for current states
        q_values = self.policy_net(states).gather(1, actions)

        # Compute Q values for next states using the target network
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Update policy network
        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



### ------- You can make changes to this file from above this line --------------

    # This is a custom Evaluation function. You should not change this function
    # You can add other methods, or other functions to perform evaluation for
    # yourself. However, this evalution function will be used to evaluate your model
    def EvaluateModel(self):
        attacked = 0
        reached_goal = 0
        state = game_state.GameState()
        for _ in range(100000):
            action = self.GetAction(state)
            obs = state.Update(action)
            if(obs==game_state.GameObservation.Enemy_Attacked):
                attacked += 1
            elif(obs==game_state.GameObservation.Reached_Goal):
                reached_goal += 1
        return (attacked, reached_goal)