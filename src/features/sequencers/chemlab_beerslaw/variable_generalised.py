import pickle
from re import T
import yaml
import numpy as np
import pandas as pd

from features.sequencers.sequencer import Sequencer
from features.sequencers.sequencer_chemlab import ChemLabSequencer

from sklearn.preprocessing import MinMaxScaler

class GeneralisedChemLabSequencer(ChemLabSequencer):
    """
    The goal is for this encoding to be generalised across sims
    This class creates sequences such that the first cells implement the states:
    - green green
    - green red
    - not green not red
    - not observed

    - explore
    - analysis
    - record
    Actions:
    - break
    - other
    - colour
    - concentration
    - width

    We make it such that breaks of _break-time_ seconds only are represented and that actions separated by less than _merge-time_ are merged


    Args:
        ColoradoSequencer (_type_): _description_
    """

    def __init__(self, settings):
        super().__init__(dict(settings))
        self._name = 'generalised_chemlab'
        self._notation = 'cb_gen'
        self._dataset = 'chemlab_beerslaw'

        self._states = [
            'nonobserved', #no stored energy ticked on
            'notoptimal', #not played with voltage yet
            'easymode', #closed circuit and stored energy
            'difficultmode', #open circuit and stored energy
            'explore',
            'analysis',
            'record',
            'break', # 0
            'colour', #1
            'concentration', #2
            'width', #3
            'other', #4
        ]

        self._old_states = [
            'greengreen', #0
            'greenred', #1
            'notgreennotred', #2
            'noobserved', #3
            'problem',
            'inquiry',
            'other', # 0
            'concentration', #1
            'width', #2
            'solution', #3
            'wavelength', #4
            'tools', #5
            'concentrationlab', #6
            'pdf', #7
            'break' #8
        ]

        self._state_converter = {
            0: 3,
            1: 2,
            2: 1,
            3: 0,
        }

        self._action_converter = {
            0: 4,
            1: 2,
            2: 3,
            3: 1,
            4: 1,
            5: 4,
            6: 999, #should be filtered
            7: 0,
            8: 0
        }

        self._n_states = 7
        self._n_actions = 5
        self._break_threshold = self._settings['chemlab']['sequence_parameters']['break_time']
        self._merge_threshold = self._settings['chemlab']['sequence_parameters']['merge_time']
        self._concentration_lab_index = 10
        self._old_break_index = 12
        self._new_break_index = 0
        self._n_old_states = 4

    def _create_break_action_vector(self):
        vec = [0 for _ in range((self._n_actions))]
        vec[self._new_break_index] = 1
        return vec

    def _return_new_action(self, action):
        current_action = action[self._n_old_states:]
        current_action_idx = np.argmax(current_action)
        new_action = [0 for _ in range(self._n_actions)]
        # print('idx', current_action_idx, self._action_converter[current_action_idx])
        new_action[self._action_converter[current_action_idx]] = 1
        return new_action

    def _return_new_states(self, states):
        current_states = states[0:self._n_old_states]
        current_states_idx = np.argmax(current_states)
        new_states = [0 for _ in range(self._n_old_states)]
        new_states[self._state_converter[current_states_idx]] = 1
        return new_states + [1, 0, 0]

    def _return_new_vector(self, vector):
        n_actions = self._return_new_action(vector)
        n_states = self._return_new_states(vector)
        # print(current_action, new_action)
        return n_states, n_actions

    def _filter_sequences_for_old_breaks(self, sequences, begins, ends):
        ns, nb, ne = [], [], []
        for i in range(len(sequences)):
            if sequences[i][self._old_break_index] == 0 and sequences[i][self._concentration_lab_index] == 0:
                    ns.append(sequences[i])
                    nb.append(begins[i])
                    ne.append(ends[i])

        return ns, nb, ne
    


    def _create_sequence(self, **kwargs):
        # print()
        assert len(kwargs['sequence']) == len(kwargs['begin']) and len(kwargs['begin']) == len(kwargs['end'])

        sequences = [s for s in kwargs['sequence']]
        begins = [b for b in kwargs['begin']]
        ends = [e for e in kwargs['end']]
        sequences, begins, ends = self._filter_sequences_for_old_breaks(sequences, begins, ends)

        new_states, new_actions, new_begins, new_ends = [], [], [], []
        break_time = 0
        n_breaks = np.floor(begins[0] / self._break_threshold)
        for _ in range(int(n_breaks)):
            initial_state = [1, 0, 0, 0, 1, 0, 0]
            initial_action = self._create_break_action_vector()
            new_states.append(initial_state)
            new_actions.append(initial_action)
            new_begins.append(break_time)
            break_time += self._break_threshold
            new_ends.append(break_time)


        sv_0, av_0 = self._return_new_vector(sequences[0])
        new_states.append(sv_0)
        new_actions.append(av_0)
        new_begins.append(begins[0])
        new_ends.append(ends[0])

        for i_a in range(1, len(sequences)):
            # print(i_a, len(new_actions)) 
            # Handling breaks
            break_time = (begins[i_a] - ends[i_a-1])
            n_breaks = np.floor(break_time/self._break_threshold)
            for i in range(int(n_breaks)):
                new_states.append([s for s in new_states[-1]])
                new_actions.append(self._create_break_action_vector())
                new_begins.append(ends[i_a-1] + (self._break_threshold * i))
                new_ends.append(ends[i_a-1] + (self._break_threshold * (i+1)))

            # Handling Actions
            new_state_vector, new_action_vector = self._return_new_vector(sequences[i_a])
            if np.argmax(new_action_vector) == np.argmax(new_actions[-1]) and new_state_vector  == new_states[-1]:
                if begins[i_a] - new_ends[-1] < self._merge_threshold:
                    new_ends[-1] = ends[i_a]
                    continue
                new_states.append(new_state_vector)
                new_actions.append(new_action_vector)
                new_begins.append(begins[i_a])
                new_ends.append(ends[i_a])
                # print('    action: {}'.format(new_action_vector))
            else:
                new_states.append(new_state_vector)
                new_actions.append(new_action_vector)
                new_begins.append(begins[i_a])
                new_ends.append(ends[i_a])

        new_actions = [(np.array(a)) * (new_ends[i_a] - new_begins[i_a]) for i_a, a in enumerate(new_actions)]
        new_actions = [[aa for aa in a] for a in new_actions]

        assert len(new_states) == len(new_actions)
        return new_states, new_actions


                

if __name__ == '__main__': 
    settings = {}
    sequencer = GeneralisedChemLabSequencer(settings)
    s, a, l, d = sequencer.load_all_sequences()

    print('Process Over')
            


            






