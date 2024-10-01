import numpy as np
from features.sequencers.light_beer.light_beer_sequencer import LightBeerSequencer
from features.sequencers.sequencer_lightbeerbeerslaw import LightBeerBeerslawSequencer
from features.parsers.parser import Parser

class UniversalLightBeerslawSequencer(LightBeerSequencer, LightBeerBeerslawSequencer):

    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'beerslaw variable sequencer'
        self._notation = 'blvar'

        self._states = [
            'nonobserved', #no stored energy ticked on - 0
            'notoptimal', #not relevant
            'easymode', #closed circuit and stored energy - 2 
            'difficultmode', #open circuit and stored energy - 3
            'explore', # 4
            'analysis', # 5
            'record', # 6
            'break', # 7
            'colour', # 8
            'concentration', # 9
            'width', # 10
            'other', # 11
        ]

        self._dataset = 'lightbeer_beerslaw'
        self._n_states = 7

    def _get_state(self, wavelength_values, wavelength_timestamps, timestamp):
        wl = self._get_value_timestep(
            wavelength_timestamps, wavelength_values, timestamp
        )
        if wl >= 397 and wl <= 582:
            state = 0
        elif wl < 397 or (wl > 582 and wl < 645):
            state = 3
        else:
            state = 2
        return wavelength_timestamps, wavelength_values, state

    def _categorise_analysis_action(self, action:str, begin:float, end:float, xaxis:dict, yaxis:dict) -> str:
        """categorise the graph and table actions into groups such that depending on the axis, the graph actions are categorised
        as analysis + the right variable, or as analysis + table if an action is made on the table, or as analysis_else for any other
        graph action not categorised.
        Returns the original action if it does not correpond to the above case

        Args:
            action (str): the name of the action
            begin (float): beginning timestamp of that action
            end (float): end timestamp of that action
            xaxis (dict): values on the xaxis across time
            yaxis (dict): values on the yaxis across time

        Returns:
            str: the new name of the action
        """
        if action == 'graph':
            x_value, _ = self._find_axis_value(xaxis['values'], xaxis['timestamps'], begin)
            y_value, _ = self._find_axis_value(yaxis['values'], yaxis['timestamps'], begin)

            if (x_value == 'absorbance' and y_value == 'width') or (x_value == 'width' and y_value == 'absorbance'):
                return 'analysis_width'

            elif (x_value == 'absorbance' and y_value == 'wavelength') or (x_value == 'wavelength' and y_value == 'absorbance'):
                return 'analysis_wavelength'

            elif (x_value == 'absorbance' and y_value == 'concentration') or (x_value == 'concentration' and y_value == 'absorbance'):
                return 'analysis_concentration'

            else:
                return 'analysis_other'

        elif action == 'table':
            return 'analysis_table'
        
        else:
            return action

    def _create_vector_sequence(self, sim:Parser):
        actions, begins, ends = self._create_action_sequences(sim)
        if len(actions) == 0:
            return [], [], []
        sequences = sim.get_sequences()

        vector_actions = [self._qualify_action(a) for a in actions]

        wavelength_timsetamps = [ts for ts in sequences['wavelength']['timestamps']]
        wavelength_values = [v for v in sequences['wavelength']['values']]

        states = []
        for i in range(len(begins)):
            wavelength_timsetamps, wavelength_values, s = self._get_state(
                wavelength_values, wavelength_timsetamps, begins[i]
            )
            states.append(s)

        vector_features = [
            self._qualify_state(states[ii], vector_actions[ii]) for ii in range(len(states))
        ]

        states = [vf[:self._n_states] for vf in vector_features]
        actions = [vf[self._n_states:] for vf in vector_features]
        actions = [(ends[i] - begins[i]) * np.array(actions[i]) for i in range(len(actions))]
        actions = [[aaa for aaa in aa] for aa in actions]
        return states, actions

