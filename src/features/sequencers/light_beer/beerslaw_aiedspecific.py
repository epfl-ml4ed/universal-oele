import numpy as np
from features.sequencers.light_beer.light_beer_sequencer import LightBeerSequencer
from features.sequencers.sequencer_lightbeerbeerslaw import LightBeerBeerslawSequencer
from features.sequencers.light_beer.beerslawvariable_sequencer import UniversalLightBeerslawSequencer
from features.parsers.parser import Parser

class SpecificAiedLightBeerslawSequencer(LightBeerSequencer, LightBeerBeerslawSequencer):

    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'beerslaw variable sequencer'
        self._notation = 'beervar'
        self._dataset = 'lightbeer_capacitor'

        self._states = ['break',
            'explore_other', 'explore_concentration','explore_width','explore_wavelength',
            'explore_concentration_width', 'explore_concentration_wavelength', 'explore_concentration_other',
            'explore_wavelength_width', 'explore_other_width', 'explore_other_wavelength',
            'explore_concentration_wavelength_width', 'explore_concentration_other_width', 'explore_concentration_other_wavelength',
            'explore_other_wavelength_width', 'explore_concentration_other_wavelength_width',
            'analysis_concentration', 'analysis_width', 'analysis_wavelength', 'analysis_other',
            'analysis_concentration_other', 'analysis_other_wavelength', 'analysis_table',
            'record_concentration', 'record_width', 'record_wavelength',
            'record_concentration_width', 'record_concentration_wavelength', 'record_wavelength_width',
            'record_concentration_wavelength_width',
            'record'
        ]

        self._universal_states = [
            'nonobserved', #no stored energy ticked on - 0
            'notoptimal', #not played with voltage yet - 1
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

        self._state_map = {
            self._states[i]: i for i in range(len(self._states))
        }

        self._voltage_notplayed = False
        self._n_states = len(self._states)
        self._universal = UniversalLightBeerslawSequencer(settings)


    def _states_to_vec(self, vector):
        action = ''
        if vector[4] > 0:
            action += 'explore'
        elif vector[5] > 0:
            action += 'analysis'
        elif vector[6] > 0:
            action += 'record'
        
        if vector[9] > 0:
            action += '_concentration'
        if vector[11] > 0:
            action += '_other'
        if vector[8] > 0:
            action += '_wavelength'
        if vector[10] > 0:
            action += '_width'

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

