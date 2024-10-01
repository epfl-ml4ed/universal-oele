import sys
import yaml
import argparse
from visualisers.transfer_plotter import TransferPlotter
from visualisers.model_plotter import ModelPlotter
from visualisers.nested_plotter import NestedXValPlotter

from features.pipeline_maker import PipelineMaker


def create_checkpoint_reproductions(settings):
    """Given an experiment ran as:
        python script_classification.py --full --fulltime --sequencer name_sequencer
        and with TF models recording tensorflow checkpoint, that function recreates the same files as the normal results
        with the validation models instead.
    """
    print('python script_classification.py --checkpoint --sequencer <insert sequencer>')
    
def plot_parameters_distribution(settings):
    config = dict(settings)
    plotter = NestedXValPlotter(config)
    plotter.plot_parameters()
    
def plot_parameters_distribution(settings):
    config = dict(settings)
    plotter = NestedXValPlotter(config)
    plotter.plot_separate_parameters()

def plot_experiment(settings):
    config = dict(settings)
    plotter = NestedXValPlotter(config)
    plotter.plot_experiment()

def plot_transfer(settings):
    transferplotter = TransferPlotter(settings)
    transferplotter.plot()





def test(settings):
    var = input('Hello world, write something \n')
    print(var)
    print(sys.argv)


    
# def train_validation(settings):
#     config = dict(settings)
#     plotter = TrainValidationPlotter(config)
#     if settings['trainvalidation']:
#         for metric in settings['train_validation']['metrics']:
#             plotter.plot(metric)
#     if settings['vap      lidation_scores']:
#         plotter.print_validation_scores()
    

    
def main(settings):
    if settings['seedplot']:
        plot_experiment(settings)
    if settings['test']:
        test(settings)
    # if settings['interpretation']:
    #     plot_interpretation(settings)
    if settings['transfer']:
        plot_transfer(settings)

        
    
if __name__ == '__main__': 
    with open('./configs/plotter_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Plot the results')
    # Taskspip in
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    parser.add_argument('--full', dest='full_sequences', default=False, action='store_true')
    parser.add_argument('--seedplot', dest='seedplot', default=False, action='store_true')
    parser.add_argument('--transfer', dest='transfer', default=False, action='store_true')
    

    # Actions
    parser.add_argument('--print', dest='print', default=False, action='store_true')
    parser.add_argument('--show', dest='show', default=False, action='store_true')
    parser.add_argument('--save', dest='save', default=False, action='store_true')
    parser.add_argument('--saveimg', dest='saveimg', default=False, action='store_true')
    parser.add_argument('--savepng', dest='savepng', default=False, action='store_true')
    parser.add_argument('--partial', dest='partial', default=False, action='store_true')
    parser.add_argument('--nocache', dest='nocache', default=False, action='store_true')
    
    settings.update(vars(parser.parse_args()))
    main(settings)