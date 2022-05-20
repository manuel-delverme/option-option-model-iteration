import copy

import numpy as np

class DeterministicModel:
    def __init__(self, transition_matrix, value_model, discount):
        self.discount = discount
        self.void_state = transition_matrix.shape[0]

        if len(transition_matrix.shape) == 2:
            void_transitions = transition_matrix.max(axis=1) == 0
            transition_matrix = transition_matrix.argmax(axis=1)
            transition_matrix[void_transitions] = self.void_state

        self.transition_model = transition_matrix
        self._value_model = value_model

    @property
    def value_model(self):
        return self._value_model * self.discount

    @value_model.setter
    def value_model(self, value):
        self._value_model = value

    def copy(self):
        return DeterministicModel(copy.deepcopy(self.transition_model), copy.deepcopy(self.value_model), self.discount)

    def compose(self, other):
        if isinstance(other, DeterministicModel):
            void_transition = self.transition_model == self.void_state

            P1 = np.full_like(self.transition_model, fill_value=self.void_state)
            R1 = np.zeros_like(self.transition_model)

            s1 = self.transition_model[~void_transition]
            P1[~void_transition] = other.transition_model[s1]
            R1[~void_transition] = other.value_model[s1]
            return DeterministicModel(P1, R1, self.discount * other.discount)

        elif isinstance(other, np.ndarray):
            if len(other.shape) == 1:
                if isinstance(self.transition_model, np.ndarray):
                    return other[self.transition_model] * self.discount + self.value_model
                else:
                    return self.transition_model(other) * self.discount + self.value_model
        else:
            raise NotImplementedError

    def project(self, state_idx, value):
        s1 = self.transition_model[state_idx]
        return s1, value

    def __getitem__(self, state_idx):
        if state_idx == self.void_state:
            return self.void_state, 0
        return self.transition_model[state_idx], self.value_model[state_idx]

    def __setitem__(self, state_idx, value_):
        new_state, new_value = value_
        self.transition_model[state_idx] = new_state
        self.value_model[state_idx] = new_value
