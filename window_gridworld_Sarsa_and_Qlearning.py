import numpy as np
import matplotlib.pyplot as plt



#------- Parameter ---------
# Windy grid world
GRID_WIDTH = 10
GRID_HEIGHT = 7

START = (0, 3)
GOAL = (7, 3)
WINDS = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)


EPSILON = 0.1
ALPHA = 0.5
GAMMA = 0.5
EPISODE = 1000

# ---------------------------


# grid 경계 안에 있는 가능한 action을 return
def get_possible_actions(s):
    possible_actions = []
    width_range = GRID_WIDTH - 1
    height_range = GRID_HEIGHT - 1

    if s[0] != 0:
        possible_actions.append("left")
    if s[0] != width_range:
        possible_actions.append("right")
    if s[1] != 0:
        possible_actions.append("up")
    if s[1] != height_range:
        possible_actions.append("down")

    return possible_actions


def initialize_Q(width, height):
    states = []
    for w in range(width):
        for h in range(height):
            states.append((w, h))

    Q = {}
    for state in states:
        Q[state] = {}
        possible_actions = get_possible_actions(state)
        for action in possible_actions:
            Q[state][action] = 0
    return Q

def is_GOAL(state):
    if state == GOAL:
        return True
    else:
        return False


def take_action_with_wind(action, state):
    next_state = list(state)
    if action == "left":
        next_state[0] = state[0] - 1
    elif action == "right":
        next_state[0] = state[0] + 1
    elif action == "up":
        next_state[1] = state[1] - 1
    elif action == "down":
        next_state[1] = state[1] + 1

    # wind 적용했을 때
    next_state[1] -= WINDS[next_state[0]]

    if next_state[1] < 0:
        next_state[1] = 0

    # reward 결정
    if is_GOAL(next_state):
        reward = 1
    else:
        reward = -1

    return tuple(next_state), reward


# epsilon greedy
def set_action_with_epsilon_greedy(Q, epsilon, state):

    possible_actions = get_possible_actions(state)

    random_probablity = np.random.rand()

    max_reward = -99999
    selected_action = "up"

    if random_probablity < epsilon: # Random 으로 선택
        selected_action = possible_actions[np.random.choice(len(possible_actions))]

    else:                             # max Q 선택
        for action, reward in Q[state].items():
            if reward > max_reward:
                max_reward = reward
                selected_action = action

    return selected_action

def my_salsa(start_state, epsilon, alpha, gamma, episodes):

    result_reward = []

    for episode in range(episodes): # Per Episode

        # Reset windy grid world
        world = np.full((GRID_WIDTH, GRID_HEIGHT), -1)  # 나머지 grid의 reward는 -1
        world[7, 3] = 1  # Goal의 reward 는 1
        world_current_state = start_state

        # initialize
        current_S = world_current_state[:]
        rewards = []

        while True: # Per Step

            # 현재 state
            current_S = world_current_state[:]

            # take action
            current_A = set_action_with_epsilon_greedy(Q, epsilon, current_S)
            world_current_state, reward = take_action_with_wind(current_A, current_S)

            # epsilon greedy로 다음 state로 부터 다음 action을 선택
            next_S = world_current_state[:] # 움직이고 난 뒤의 상태
            next_A = set_action_with_epsilon_greedy(Q, epsilon, next_S)

            # update Q
            Q[current_S][current_A] = Q[current_S][current_A] + alpha * (reward + gamma * Q[next_S][next_A] - Q[current_S][current_A])

            # reward에 추가
            rewards.append(reward)

            if is_GOAL(next_S):
                result_reward.append(sum(rewards))
                break

    return result_reward


def set_action_with_argmax(Q, epsilon, state):
    selected_action = "up"
    max_reward = -9999
    for action, reward in Q[state].items():
        if reward > max_reward:
            max_reward = reward
            selected_action = action

    return selected_action

def my_q_learning(start_state, epsilon, alpha, gamma, episodes):

    result_reward = []

    for episode in range(episodes): # Per Episode

        # Reset windy grid world
        world = np.full((GRID_WIDTH, GRID_HEIGHT), -1)  # 나머지 grid의 reward는 -1
        world[7, 3] = 1  # Goal의 reward 는 1
        world_current_state = start_state

        # initialize
        current_S = world_current_state[:]
        rewards = []

        while True: # Per Step

            # 현재 state
            current_S = world_current_state[:]

            # take action
            current_A = set_action_with_epsilon_greedy(Q, epsilon, current_S)
            world_current_state, reward = take_action_with_wind(current_A, current_S)

            # epsilon greedy로 다음 state로 부터 다음 action을 선택
            next_S = world_current_state[:] # 움직이고 난 뒤의 상태
            t = Q[next_S]
            next_A = set_action_with_argmax(Q, epsilon, next_S)

            # update Q
            Q[current_S][current_A] = Q[current_S][current_A] + alpha * (reward + gamma * Q[next_S][next_A] - Q[current_S][current_A])

            # reward에 추가
            rewards.append(reward)

            if is_GOAL(next_S):
                result_reward.append(sum(rewards))
                break

    return result_reward


if __name__ == "__main__":
    Q = initialize_Q(GRID_WIDTH, GRID_HEIGHT)

    salsa_result_reward = my_salsa(START, EPSILON, ALPHA, GAMMA, EPISODE)

    Q = initialize_Q(GRID_WIDTH, GRID_HEIGHT)

    q_learning_result_reward = my_q_learning(START, EPSILON, ALPHA, GAMMA, EPISODE)

    plt.plot(salsa_result_reward)
    plt.plot(q_learning_result_reward)
    plt.legend(['salsa', 'q'])


    plt.show()
