import random


NUM_EPISODES = 400
MAX_EPISODE_LENGTH = 500

DEFAULT_DISCOUNT = 0.9
EPSILON = 0.5
LEARNINGRATE = 0.1


class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): # You can add more arguments if you want
        self.name = "agent1"
        self.num_actions = num_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.allActions = [0,1,2,3]
        self.justQ = []
        for state in range(num_states):
            self.justQ.append([])
            for action in range(num_actions):
                self.justQ[state].append(0)

    def process_experience(self, state, action, next_state, reward, done): # You can add more arguments if you want
        """
        Update the Q-value based on the state, action, next state and reward.
        """

        if not done:
            greedyReward = max(self.justQ[next_state])
            self.justQ[state][action] = (1 - self.learning_rate)*self.justQ[state][action] + self.learning_rate*(reward + self.discount*greedyReward)
        else:
            self.justQ[state][action] = (1 - self.learning_rate) * self.justQ[state][action] + self.learning_rate * (reward)


    def select_action(self, state, episode, totalEpisodes): # You can add more arguments if you want
        """
        Returns an action, selected based on the current state
        """
        max_reward = max(self.justQ[state])
        action_indices = []

        for i, action in enumerate(self.justQ[state]):
            if action == max_reward:
                action_indices.append(i)

        greedy_action = action_indices[random.randint(0, len(action_indices) - 1)]

        if random.random() < (EPSILON - (episode*0.45)/totalEpisodes):
            return random.randint(0,3)
        else:
            return greedy_action


    def report(self):

        for i in range(len(self.justQ)):
            print(i)
            print(self.justQ[i])