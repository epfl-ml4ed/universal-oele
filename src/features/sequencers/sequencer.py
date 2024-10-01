import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Sequencer:
    """Generate a matrix where the interaction is represented according to the sequencer's type.
    This one in particular groups all function applicable to all sequencers
    """
    def __init__(self, settings: dict):
        self._settings = dict(settings)
        self._click_length = 0.05

    def get_n_states(self):
        return self._n_states
    
    def get_n_actions(self):
        return self._n_actions

    # def _input_time(self):
    #     """given a one-hot-encoded array, if
    #     """

    def _get_all_breaks(self, begin: list, end: list):
        b = begin + [0]
        e = [0] + end 
        
        breaks = list(np.array(b) - np.array(e))
        breaks = breaks[:-1]
        
        breaks = [b for b in breaks if b > 0]
        return breaks

    def get_threshold(self, begins:list, ends:list, threshold:float):
        begin = [b for b in begins]
        end = [e for e in ends]
        breaks = self._get_all_breaks(begin, end)
        if len(breaks) == 0:
            return 0
        breaks.sort()
        threshold = int(np.floor(threshold * len(breaks)))
        
        threshold = breaks[threshold]
        return threshold

    def _impute_breaks(self, sequence):
        """inputs breaks in a sequence of actions
        Usually, two methodology:
        - take the x% higher pauses between two actions, these are breaks, the rest is discarded
        - every period of inaction longer than Xs is a break, the rest i discarded

        Args:
            sequence (_type_): one student sequence
        """
        raise NotImplementedError

    def _scale_features(self, states, actions, actions_scale):
        assert len(states) == len(actions)
        sequences = []
        for student in range(len(states)):
            scaler = MinMaxScaler()
            scaler.fit(actions[student])
            transformed_actions = scaler.transform(actions[student])
            sequences.append(
                [[*states[student][ts], *transformed_actions[ts]] for ts in range(len(states[student]))]
            )
        return sequences

    def _load_sequence(self, sim):
        """Creates the sequences when give a simulation object

        Args:
            sim (SimulationObject): Simulation object (type depends on the simulation)
        """
        print('Sequencer class, no specific sequencer selected')
        raise NotImplementedError

    def load_sequences(self):
        raise NotImplementedError
