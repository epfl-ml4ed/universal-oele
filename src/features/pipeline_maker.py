from re import M
from features.sequencers.colorado_capacitor.variable_generalised import StatePhaseActionColoradoSequencer
from features.sequencers.colorado_capacitor.edm2021 import EDM2021ColoradoSequencer
from features.sequencers.colorado_capacitor.edmfeatures_aiedformat import EDM2021AIEDColoradoSequencer


from features.sequencers.chemlab_beerslaw.edm2022 import EDM2022ChemLabSequencer
from features.sequencers.chemlab_beerslaw.edmfeatures_aiedformat import EDM2022AIEDChemLabSequencer
from features.sequencers.chemlab_beerslaw.variable_generalised import GeneralisedChemLabSequencer
from features.sequencers.instruction_vet.instruction_beerslaw_sequencer import UniversalInstructionBeerslawSequencer
from features.sequencers.instruction_vet.instructionbeerslaw_aied_sequencer import InstructionAiedLightBeerslawSequencer
from features.sequencers.light_beer.beerslawvariable_sequencer import UniversalLightBeerslawSequencer
from features.sequencers.light_beer.beerslaw_aiedspecific import SpecificAiedLightBeerslawSequencer
from features.sequencers.light_beer.capacitor_aiedspecific import SpecificLightCapacitorSequencer
from features.sequencers.light_beer.capacitorvariable_sequencer import UniversalLightCapacitorSequencer

class PipelineMaker:

    def __init__(self, settings):
        self._settings = dict(settings)

    def _select_sequencer(self, sequencer_name):
        ####### Previous Encodings
        if sequencer_name == 'spec_capcol':
            maxlen = 942
            self._settings['data']['type'] = 'simulation'
            self._settings['features']['dimension'] = 7
            return EDM2021ColoradoSequencer(self._settings), maxlen
            
        if sequencer_name == 'spec_chem':
            maxlen = 830
            self._settings['data']['type'] = 'simulation'
            self._settings['features']['dimension'] = 9
            return EDM2022ChemLabSequencer(self._settings), maxlen
        
        if sequencer_name == 'spec_lightcap':
            maxlen = 161
            self._settings['data']['type'] = 'simulation'
            self._settings['features']['dimension'] = 31
            self._settings['data']['merge'] = True
            return SpecificLightCapacitorSequencer(self._settings), maxlen

        if sequencer_name == 'spec_lightbeer':
            maxlen = 620
            self._settings['data']['type'] = 'simulation'
            self._settings['features']['dimension'] = 31
            self._settings['data']['merge'] = True
            return SpecificAiedLightBeerslawSequencer(self._settings), maxlen

        if sequencer_name == 'spec_vetbeer':
            maxlen = 290
            self._settings['data']['type'] = 'simulation'
            self._settings['features']['dimension'] = 39
            self._settings['data']['merge'] = True
            return InstructionAiedLightBeerslawSequencer(self._settings), maxlen
            

        ####### Universal encodings
        if sequencer_name == 'universal_capcol':
            maxlen = 942
            self._settings['data']['type'] = 'simulation'
            self._settings['features']['dimension'] = 12
            return StatePhaseActionColoradoSequencer(self._settings), maxlen

        if sequencer_name == 'universal_chem':
            maxlen = 163
            self._settings['data']['type'] = 'simulation'
            self._settings['features']['dimension'] = 12
            return GeneralisedChemLabSequencer(self._settings), maxlen

        if sequencer_name == 'universal_lightcap':
            maxlen = 272
            self._settings['data']['type'] = 'simulation'
            self._settings['features']['dimension'] = 12
            self._settings['data']['merge'] = True
            return UniversalLightCapacitorSequencer(self._settings), maxlen

        if sequencer_name == 'universal_lightbeer':
            maxlen = 713
            self._settings['data']['type'] = 'simulation'
            self._settings['features']['dimension'] = 12
            self._settings['data']['merge'] = True
            return UniversalLightBeerslawSequencer(self._settings), maxlen

        if sequencer_name == 'universal_vetbeer':
            maxlen = 334
            self._settings['data']['type'] = 'simulation'
            self._settings['features']['dimension'] = 12
            return UniversalInstructionBeerslawSequencer(self._settings), maxlen

        

    def _get_permutation_column(self):
        sequencer_names = self._settings['data']['secundary'].split('.')
       
        if len(sequencer_names) == 1 and not self._settings['coldstart'] and self._settings['ml']['splitters']['stratifier_col'] == 'x':
            if sequencer_names[0] == 'universal_capcol':
                self._settings['ml']['splitters']['stratifier_col'] = 'permutation'
            elif sequencer_names[0] == 'universal_chem':
                self._settings['ml']['splitters']['stratifier_col'] = 'vector_binary'
            elif sequencer_names[0] == 'universal_lightbeer':
                self._settings['ml']['splitters']['stratifier_col'] = 'strat_bl'
            elif sequencer_names[0] == 'universal_lightcap':
                self._settings['ml']['splitters']['stratifier_col'] = 'strat_cap'
            elif sequencer_names[0] == 'universal_vetbeer':
                self._settings['ml']['splitters']['stratifier_col'] = 'label'
            else:
                self._settings['ml']['splitters']['stratifier_col'] = 'y'



    def _load_sequences(self, key):
        state_actions, labels, demographics, indices = [], [], [], []
        sequencer_names = self._settings['data'][key].split('.')
        maxlen_max = []
        for sn in sequencer_names:
            sequencer, maxlen = self._select_sequencer(sn)
            sa, l, d, i = sequencer.load_all_sequences()
            print('{}: {}'.format(sn, len(sa)))
            state_actions = [*state_actions, *sa]
            labels = [*labels, *l]
            demographics = [*demographics, *d]
            indices = [*indices, *i]
            maxlen_max.append(maxlen)
        maxlen = max(maxlen_max)
        
        return state_actions, labels, demographics, indices, maxlen

    def load_sequences(self):
        self._get_permutation_column()
        primary_sa, primary_l, primary_d, primary_i, prim_maxlen = self._load_sequences('primary')
        secundary_sa, secundary_l, secundary_d, secundary_i, sec_maxlen = self._load_sequences('secundary')
        self._settings['ml']['models']['maxlen'] = max([prim_maxlen, sec_maxlen])
        print('SEQUENCES LOADED', self._settings['ml']['splitters'])
        return primary_sa, primary_l, primary_d, primary_i, secundary_sa, secundary_l, secundary_d, secundary_i, self._settings
        


