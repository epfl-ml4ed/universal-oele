
import pickle
import copy
import yaml
from sklearn.preprocessing import MinMaxScaler
from features.sequencers.sequencer import Sequencer

class LightBeerCapacitorSequencer(Sequencer):
    def __init__(self, settings):
        self._settings = dict(settings)
        self._name = 'lightbeercapseq'
        self._notation = 'lbcseq'

        self._load_settings()
        self._get_label_map()

        self._demographics = [
            'lid', 'simulations', 'age', 'gender', 'year', 'major', 
            'english_writing', 'english_reading', 'activity_order', 'n_prior_labs', 'overall_pocc', 
            'pre_concentration', 'pre_wavelength', 'pre_width', 'pre_area', 'pre_separation', 'pre_battery_voltage', 
            'main_concentration', 'main_wavelength', 'main_width', 'main_area', 'main_separation', 
            'main_battery_voltage', 'beerslaw_first', 'beerslaw_second', 'capacitor_first', 'capacitor_second','strat_cap', 'strat_bl'
        ]

    def _load_settings(self):
        path = './configs/datasets/lightbeer_capacitor.yaml'
        with open(path) as fp:
            setts = yaml.load(fp, Loader=yaml.FullLoader)
            self._settings['lightbeer_capacitor'] = setts

        main_path = './configs/datasets/global_config.yaml'
        with open(main_path) as fp:
            main_setts = yaml.load(fp, Loader=yaml.FullLoader)
        self._settings.update(main_setts)

    def _get_label_map(self):
        root_path = '{}/experiment_keys/lightbeer_'.format(self._settings['paths']['data'])
        if self._settings['lightbeer_capacitor']['data']['label_map'] == '2highlow':
            path = '{}2highlow.yaml'.format(root_path)

        with open(path) as fp:
            self._label_map = yaml.load(fp, Loader=yaml.FullLoader)


    def load_all_sequences(self):
        """Functions returning information about the ChemLab experiment conducted in vocational schools with the
        Beer's Law Lab Phet Simulation.

        Return:
            states (list): processed interaction, only the state features
            actions (list): processed interaction list, only the action features
            demographics (list<dict>): list of demographical dictionaries 


        """
        path = '{}{}canada_dict.pkl'.format(self._settings['paths']['data'], self._settings['lightbeer_capacitor']['paths']['root'])
        with open(path, 'rb') as fp:
            sim_dictionary = pickle.load(fp)

        states = []
        actions = []
        actions_scale = []
        labels = []
        demographics = []
        indices = []


        for idx in sim_dictionary['index']['index_lid']:
            if 'capacitor' not in sim_dictionary[idx]:
                continue
            
            with open(sim_dictionary[idx]['capacitor'].replace('../data', self._settings['paths']['data']), 'rb') as fp:
                sim_capacitor = pickle.load(fp)
            s, a = self._create_vector_sequence(sim_capacitor)
            if len(a) == 0:
                continue
            indices.append(idx)
            l = self._get_label(sim_dictionary[idx])
            d = self._get_demographics(**sim_dictionary[idx])
            d['dataset'] = 'lightcap'
            d['dataset_label'] = '{}-{}'.format(int(l), d['strat_cap'])

            states.append(s)
            actions.append(a)
            actions_scale = [*actions_scale, *a]
            labels.append(l)
            demographics.append(copy.deepcopy(d))
        
        assert len(labels) == len(demographics) and len(demographics) == len(indices)
        sequences = self._scale_features(states, actions, actions_scale)
        return sequences, labels, demographics, indices

    def _get_label(self, simulation:dict):
        """Returns the processed label based on the ranking selected
        """
        if self._settings['lightbeer_capacitor']['data']['label_map'] == '2highlow':
            score = simulation['main_area'] + simulation['main_separation'] + simulation['main_battery_voltage']
        return self._label_map['map'][score]

    def _get_demographics(self, **kwargs):
        """Returns the permutation for the permutation stratification
        """
        return {d: kwargs[d] for d in self._demographics}
        
    def _create_sequence(self, **kwargs):
        """Function to create the sequences under the appropriate format

        Args:
            actions (_type_): chronological list of actions in the interaction
            begins (_type_): beginning timestamp of each of the action
            ends (_type_): end timestamp of each of the action
            simulation_values (_type_): non interaction values taken throughout the simulation (state, tec.)
            xaxis (_type_): _description_
            yaxis (_type_): _description_
            trials (_type_): _description_
        """

        raise NotImplementedError



        



