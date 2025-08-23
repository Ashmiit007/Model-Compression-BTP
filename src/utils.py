
import math

def sigmoid(x):
    if x > -70:
        return 1 / (1 + math.exp(-x))
    return 0

def update_w(delta_w, weights, lr=0.01):
    for l, layer in enumerate(weights):
        for i in range(len(layer)):
            weights[l][i] -= lr * delta_w[l][i]
    return weights

def calculate_advantage(layer_reward, total_layer_reward, iter_cnt):
    return [layer_reward[i] - (total_layer_reward[i] / iter_cnt) for i in range(len(layer_reward))]
