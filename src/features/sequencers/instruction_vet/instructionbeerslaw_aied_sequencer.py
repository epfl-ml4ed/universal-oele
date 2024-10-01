import numpy as np
from features.sequencers.instruction_vet.instruction_sequencer import InstructionSequencer
from features.sequencers.light_beer.light_beer_sequencer import LightBeerSequencer
from features.sequencers.sequencer_instructionbeerslaw import InstructionBeerslawSequencer
from features.sequencers.sequencer_lightbeerbeerslaw import LightBeerBeerslawSequencer
from features.sequencers.light_beer.beerslawvariable_sequencer import UniversalLightBeerslawSequencer
from features.sequencers.instruction_vet.instruction_beerslaw_sequencer import UniversalInstructionBeerslawSequencer
from features.parsers.parser import Parser

class InstructionAiedLightBeerslawSequencer(InstructionSequencer,InstructionBeerslawSequencer):

    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'beerslaw aied variable sequencer'
        self._notation = 'beervar'
        self._dataset = 'instructionvet_beer'

        self._states = [
            'break',
            'explore_other', 'explore_concentration','explore_width','explore_wavelength',
            'explore_concentration_width', 'explore_concentration_wavelength', 'explore_concentration_other',
            'explore_wavelength_width', 'explore_other_width', 'explore_other_wavelength',
            'explore_concentration_wavelength_width', 'explore_concentration_other_width', 'explore_concentration_other_wavelength',
            'explore_other_wavelength_width', 'explore_concentration_other_wavelength_width',
            'analysis_concentration', 'analysis_width', 'analysis_wavelength', 'analysis_other', 'analysis_other_width',
            'analysis_concentration_other', 'analysis_other_wavelength', 'analysis_concentration_wavelength',  'analysis_wavelength_width',
            'analysis_concentration_width', 'analysis_concentration_other_wavelength_width', 'analysis_concentration_other_wavelength', 'analysis_table',
            'record_concentration', 'record_width', 'record_wavelength', 'record_other_wavelength',
            'record_concentration_width', 'record_concentration_wavelength', 'record_wavelength_width', 'record_other_width',
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
        self._universal = UniversalInstructionBeerslawSequencer(settings)

    def _mergeactions(self, actions, begins, ends):
        """Merge actions which are separated by less than a specific threshold

        Args:
            actions (_type_): actions
            begins (_type_): beginning ts
            ends (_type_): ending ts

        Returns:
            _type_: _description_
        """
        new_actions = [actions[0]]
        new_begins = [begins[0]]
        new_ends = [ends[0]]
        for i in range(1, len(actions)):
            if (('analysis' in actions[i] and 'analysis' in new_actions[-1]) or \
                    ('record' in actions[i] and 'record' in new_actions[-1]) or\
                        (actions[i] in ['voltage', 'circuit', 'area', 'separation'] and new_actions[-1] in ['voltage', 'circuit', 'area', 'separation'])) and\
                         actions[i] != 'break':
                if begins[i] - new_ends[-1] < self._merge_threshold:
                    new_ends[-1] = ends[i]
                    if actions[i] not in new_actions[-1]:
                        new_actions[-1] = '{}_{}'.format(new_actions[-1], actions[i].replace('record_', '').replace('analysis_', ''))
                else:
                    new_actions.append(actions[i])
                    new_begins.append(begins[i])
                    new_ends.append(ends[i])
            else:
                new_actions.append(actions[i])
                new_begins.append(begins[i])
                new_ends.append(ends[i])

        assert len(new_actions) == len(new_begins) and len(new_begins) == len(new_ends)
        return new_actions, new_begins, new_ends


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

    def _create_vector_sequence(self, s, a, merged_s, merged_a):
        
        new_sequence = [
            self._states_to_vec([*merged_s[ts], *merged_a[ts]]) for ts in range(len(merged_s))
        ]
        new_states = [[] for _ in range(len(merged_s))]
        return new_states, new_sequence

