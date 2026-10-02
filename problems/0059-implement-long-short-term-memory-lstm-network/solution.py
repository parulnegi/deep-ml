import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		hidden_states = []
		for t in range(len(x)):
			x_t = x[t].reshape(self.input_size ,1)
			combined = np.vstack((initial_hidden_state, x_t))
			zft = (self.Wf @ combined) + self.bf
			ft = 1 / (1 + np.exp(-zft))

			zit = (self.Wi @ combined) + self.bi
			it = 1 / (1 + np.exp(-zit))

			ct_delta = np.tanh((self.Wc @ combined) + self.bc)

			initial_cell_state = (ft * initial_cell_state) + (it * ct_delta)
			zot = (self.Wo @ combined) + self.bo
			ot = 1/ (1+ np.exp(-zot))
			initial_hidden_state = ot * np.tanh(initial_cell_state)

			hidden_states.append(initial_hidden_state.copy())


		return hidden_states, initial_hidden_state, initial_cell_state

