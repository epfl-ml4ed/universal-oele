import numpy as np
from features.sequencers.light_beer.light_beer_sequencer import LightBeerSequencer
from features.sequencers.sequencer_lightbeercapacitor import LightBeerCapacitorSequencer
from features.sequencers.light_beer.capacitorvariable_sequencer import UniversalLightCapacitorSequencer
from features.parsers.parser import Parser

class SpecificLightCapacitorSequencer(LightBeerSequencer, LightBeerCapacitorSequencer):

    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'capacitor variable sequencer'
        self._notation = 'capvar'
        self._dataset = 'lightbeer_capacitor'

        self._states = [
            'break',
            'explore_other', 'explore_area', 'explore_voltage', 'explore_separation',
            'explore_area_voltage', 'explore_area_separation', 'explore_area_other', 'explore_separation_voltage',
            'explore_other_voltage', 'explore_other_separation',
            'explore_area_separation_voltage', 'explore_area_other_voltage', 'explore_area_other_separation',
            'explore_other_separation_voltage', 'explore_area_other_separation_voltage',
            'analysis_area', 'analysis_voltage', 'analysis_separation',
            'analysis_other', 'analysis_area_other', 'analysis_other_separation', 'analysis_table',
            'record_area', 'record_voltage', 'record_separation',
            'record_area_voltage', 'record_area_separation', 'record_separation_voltage',
            'record_area_separation_voltage', 'record'
        ]

        self._universal_states = [
            'nonobserved', #no stored energy ticked on - 0
            'notoptimal', #not played with voltage yet - 1
            'easymode', #closed circuit and stored energy - 2
            'difficultmode', #open circuit and stored energy - 3
            'explore' # 4,
            'analysis' # 5,
            'record' # 6,
            'break', # 7
            'voltage', # 8
            'separation', # 9
            'area', # 10
            'other' # 11
        ]

        self._state_map = {
            self._states[i]: i for i in range(len(self._states))
        }

        self._voltage_notplayed = False
        self._n_states = len(self._states)
        self._universal = UniversalLightCapacitorSequencer(settings)


    def _states_to_vec(self, vector):
        action = ''
        if vector[4] > 0:
            action += 'explore'
        elif vector[5] > 0:
            action += 'analysis'
        elif vector[6] > 0:
            action += 'record'
        
        if vector[10] > 0:
            action += '_area'
        if vector[11] > 0:
            action += '_other'
        if vector[9] > 0:
            action += '_separation'
        if vector[8] > 0:
            action += '_voltage'

        if action == 'explore' or action == 'analysis':
            action = 'explore_other'
        
        if vector[7] > 0:
            action = 'break'

        newv = [0 for _ in range(self._n_states)]
        newv[self._state_map[action]] = np.sum((vector[7:]))
        return newv

    def _create_vector_sequence(self, sim: Parser):
        s, a = self._universal._create_vector_sequence(sim)
        new_sequence = [
            self._states_to_vec([*s[ts], *a[ts]]) for ts in range(len(s))
        ]
        new_states = [[] for ts in range(len(s))]
        return new_states, new_sequence



        




        