from collections import deque


class TemporaryBufferInput:
    def __init__(self):
        self._states = deque([])

    def __len__(self):
        return len(self._states)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._states:
            raise StopIteration
        return self._prepare_batch()

    def _prepare_batch(self):
        state = self._states.popleft()
        return state

    def update_data(self, state):
        self._states.append(state)
