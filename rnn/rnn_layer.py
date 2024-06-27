import numpy as np

from common.activation import Activation
from common.layer import Layer


class RNNLayer(Layer):
    __input_size: int
    __hidden_size: int
    __output_size: int

    __input_weights: np.ndarray
    __hidden_weights: np.ndarray
    __hidden_bias: np.ndarray
    __hidden_state: np.ndarray
    __output_weights: np.ndarray
    __output_bias: np.ndarray

    __sequence: np.ndarray
    __outputs: np.ndarray
    __sequence_length: int

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            activation: Activation,
    ):
        k: float = 1 / np.sqrt(hidden_size)

        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__output_size = output_size

        self.__activation = activation
        self.__input_weights = np.random.uniform(size=(hidden_size, input_size), low=-k, high=k)
        self.__hidden_weights = np.random.uniform(size=(hidden_size, hidden_size), low=-k, high=k)
        self.__hidden_bias = np.random.uniform(size=hidden_size, low=-k, high=k)
        self.__output_weights = np.random.uniform(size=(output_size, hidden_size), low=-k, high=k)
        self.__output_bias = np.random.uniform(size=output_size, low=-k, high=k)

    def forward(self, sequence: np.ndarray) -> np.ndarray:
        self.__sequence_length: int = sequence.shape[0]
        self.__hidden_state = np.zeros((self.__sequence_length, self.__hidden_size))
        self.__outputs = np.zeros((self.__sequence_length, self.__output_size))
        self.__sequence = sequence

        h_prev: np.ndarray = np.zeros(self.__hidden_size)  # h_{t-1}

        for i, row in enumerate(sequence):
            h_cur = self.__activation(
                np.dot(self.__input_weights, row) + np.dot(self.__hidden_weights, h_prev) + self.__hidden_bias
            )  # h_t
            self.__hidden_state[i] = h_cur

            current_output: np.ndarray = np.dot(self.__output_weights, h_cur) + self.__output_bias
            self.__outputs[i] = current_output

            h_prev = h_cur

        return self.__outputs

    def backward(self, gradient: np.ndarray, learning_rate: float) -> None:
        output_weights_grad: np.ndarray = np.zeros_like(self.__output_weights)
        output_bias_grad: np.ndarray = np.zeros_like(self.__output_bias)

        hidden_weights_grad: np.ndarray = np.zeros_like(self.__hidden_weights)
        hidden_bias_grad: np.ndarray = np.zeros_like(self.__hidden_bias)

        input_weights_grad: np.ndarray = np.zeros_like(self.__input_weights)
        prev_hidden_grad: np.ndarray = np.zeros((self.__hidden_size, 1))

        prev_hidden_value: np.ndarray = np.zeros((self.__output_size, 1))

        for i in range(self.__sequence_length - 1, -1, -1):
            cur_output_grad: np.ndarray = gradient[i]
            cur_hidden_value: np.ndarray = self.__hidden_state[i]

            output_weights_grad += np.dot(cur_output_grad.reshape(-1, 1), cur_hidden_value.reshape(1, -1))
            output_bias_grad += cur_output_grad

            hidden_grad: np.ndarray = np.dot(cur_output_grad, self.__output_weights)
            hidden_grad += np.dot(prev_hidden_grad.reshape(1, -1), self.__hidden_weights.T).reshape(hidden_grad.shape)

            activation_grad: np.ndarray = self.__activation.gradient(cur_hidden_value)
            hidden_grad = np.multiply(hidden_grad, activation_grad)
            prev_hidden_grad = hidden_grad.copy()

            hidden_weights_grad += np.dot(prev_hidden_value.reshape(-1, 1), hidden_grad.reshape(1, -1))

            if i > 0:
                hidden_bias_grad += hidden_grad

            input_weights_grad += np.dot(self.__sequence[i].reshape(-1, 1), hidden_grad.reshape(1, -1)).T
            prev_hidden_value = cur_hidden_value.copy()

        adjusted_learning_rate: float = learning_rate / self.__sequence_length
        self.__input_weights -= adjusted_learning_rate * input_weights_grad
        self.__hidden_bias -= adjusted_learning_rate * hidden_bias_grad
        self.__hidden_weights -= adjusted_learning_rate * hidden_weights_grad
        self.__output_weights -= adjusted_learning_rate * output_weights_grad
        self.__output_bias -= adjusted_learning_rate * output_bias_grad

        # This layer is always first, so we don't have to return anything here
