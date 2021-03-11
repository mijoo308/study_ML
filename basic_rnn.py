import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features)) # 난수 생성

state_t = np.zeros((output_features,))


# 랜덤 가중치
W1 = np.random.random((output_features, input_features))
W2 = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []

for now_t in inputs:
    output_t = np.tanh(np.dot(W1, now_t) + np.dot(W2, state_t) + b)

    successive_outputs.append(output_t)
    state_t = output_t

final_output_sequence = np.stack(successive_outputs, axis=0)