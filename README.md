
# Raw Data
The raw data at the beginning of the pipeline, for each dataset, is a dictionary which follows the following structure:
- sequences:
    - idx0:
        - path: path where the sequence is stored (if original data is logged per student)
        - sequence: sequence of actions for the student [event_0, event_1, ..., event_n]
        - begin: sequence of the beginning of each action [begin_0, begin_1, ..., begin_n]
        - end: sequence of the end of each action [end_0, end_1, ..., end_n]
        => event_0 laster for (end_0 - begin_0) seconds
        - permutation: permutation given for the post test (can be named as you wish)
        - last_timestamp: last timestamp recorded for the simulation (only necessary for simulations)
        - length: length of the sequence (not required)
        - learner_id: unique identifier for the student
    - idx1:
        - ...
    - ...
    - idxn: 
        - ...
- index:
    - learner_id0: idx0
    - learner_id1: idx1 
    - ...
    - learner_idn: idxn

# Processing the data
The pipeline uses ```sequencers``` which are objects taking in sequences and processing the data according to the paradigm we want to use. No matter the dataset used, you need to implement your own sequencer in ```src/features/sequencers/yourown_sequence.py```. Your sequencer will need to be a subclass of ```src/features/sequencers/sequencer.py``` which will need to implement the function ```load_sequences(self)```. 

## Load Sequences
The ```load_sequences``` function must return the following arguments: ```sequences```, ```labels```, ```demographics```, ```indices```
where ```sequences[n]```, ```labels[n]```, ```demograhpics[n]```, ```indices[n]``` are the sequence, label, demographic and index of student $n$.
Here is a detail of the different return argument:
- ```sequences[n]```: dictionary where the keys are the details of what is needed in the model. For instance, int the case of sequences, we always have a state, and an action. In moocs it could be the topic and the answer, etc. In the phet case we have: {'state': [0 0 1], 'action': [2 0 0 0]}. Depending on your model, you can just make it a list (see model part).
- ```labels[n]```: target to be classified
- ```demographics[n]```: dictionary with the demographics attribute. For example: {'gender': 'non binary', language: 'swahili'}.
- ```indices[n]```: only used when you have conditions on your data, which you didn't have before. Say you usually have 1, 2, 3, 4, but because you only have those longer than 2, and 3 is one long, then the indices is 1, 2, 4. In general, if this does not make sense to you, it should just be $n$.

## Insert your sequencer into the pipeline
You are going to now edit the file ```src/features/pipeline_maker.py```. 
1. Choose a name for your dataset (I chose colorado_capacitor for the data collected at the Colorado University, which used the capacitor lab. I also chose chemlab beerslaw as we called the experiment done in 2021 the chemlab, while it uses the beer's law simulation). 
2. If you have different sequencers for one dataset, insert:
```
def _select_sequencer(self):
    # other code
    if ...
    if ...

    # your code
    if self._settings['data']['dataset'] == 'your_dataset_name':
        self._settings['ml']['models']['maxlen'] = xxx
        if self._settings['data']['sequencer'] == 'your_first_engineers':
            self._sequencer = YourFirstEngineeredFeatures(self._settings)
```
Else, suppress the second indented if statement, for sequencer, and put your sequencer directly.
3. Replace xxx (in maxlen) with the maximum length of a sequence. 


## Using it in the pipeline
1. When you want to use your dataset, open the ```src/configs/classification_config.yaml``` (let's call it ```configs```). 
2. Change the ```configs['data']['dataset']``` parameter to your dataset
3. Change ```configs['data']['sequencer']``` to the name you want.

# Models
## Implementation
All models are a subclass of the Model class in ```src/ml/models/model.py```. As an example, check out ```src/ml/models/lstm_torch.py```
To implement your model, implement the following part:
 - ```_set_seed(self)```: set a specific seed for your model.
 - ```__init__(self, settings)```: put the parameters proper to your model
 - ```_format(x, y)```: format the x and y (in the shape of sequences and labels from the data section) as the model accepts it.
 - ```_format_features```: exactly the same as in ```_format(x, y)``` without the $y$ as it is used to format the features when using probabilities. 

 - ```_init_model(self, x:np.array)```: initalised your model (model, optimiser, loss, etc.)
 - ```fit(self, x_train, y_train, x_val, y_val)```: This is where your model trains. 
 - ```predict```: what to do when predicting (there are already-made tensorflow and pytorch functions)
 - ```predict_proba```: same as above but returns the probabilities
 - ```save(self, extension='')``: saves the model. Change what parameters you want to implement in your model path. 

## Insert your model into the pipeline
1. Choose your model's name
2. Create a file in ```src/configs/gridsearch/``` and call it ```gs_<yourmodelname>.yaml```. In it, make a gridsearch. For each parameter you want to test something, make a list of values you want to try out. The gridsearch will try all combination possible. You can look at an example on ```src/configs/gridsearch/gs_dmkt.yaml```.
2. Open ```src/ml/xval_maker.py```. Before the line ```if self._settings['ml']['pipeline']['gridsearch'] != nogs``` add the following:
```
if self._pipeline_settings['model'] == 'yourmodelname':
    self._model = YourModel()
    gs_path = './configs/gridsearch/gs_<yourmodelname>.yaml'
```

## Use your model
1. When you want to use your model, open the ```src/configs/classification_config.yaml``` (let's call it ```configs```). 
2. Change ```config['ml']['pipeline']['model']``` to your model name.
3. Create an entry in ```config['ml']['models']['yourmodelname']```. Inside, put the parameters you'd like your model to have. (Check the ```config['models']['lstm']``` for an example). Not necessary when doing a gridsearch. You can just have ```config['ml']['models']['yourmodelname']['dummy_paramter']: x```

# Run the code
## Modes
1. cross-validators:
- non nested: used when no gridsearch is necessary, with k-fold cross validation
- nested: uses a gridsearch with k-fold, uses the training set to do a gridsearch (with cross validation too)
## Set the parameters
1. Set your terminal in ```src/```
2. Open the ```src/configs/classification_config.yaml``` (let's call it ```configs```) and change:
    - root_name: the root folder where you want your experiments to be saved. they will be automatically saved into ```../experiments/root_name```. Do not worry about giving the same nae, timestamps are used in the folder creation. 
    - nclasses: number of classes in the classification problem
    - max_seed: the maximum you want your seed to be
    - model_seeds_n: number of seeds you want to try out
    - data/dataset: name of your dataset
    - data/sequencer: name of your sequencer if necessary
    - ml/pipeline/xvalidator: name of the cross validation (nonnested_xval or nested_xval). 
    - ml/pipeline/model: your model name
    - ml/models/yourmodelname: {your parameters}
    If in doubt about the name of the argument, check ```src/features/pipeline_maker.py``` or ```src/ml/xval_maker.py````
3. Run:
```$python script_classification.py --seeds```

# Plot the experiments
1. Set your terminal in ```src/```
2. Open the ```src/configs/plotter_config.yaml``` (let's call it ```configs```) and change:
- experiment/name by the root name of your experiment. If different ones, the more specific/unambiguous one. 
3. run:
```$python script_plotter.py --seedplot``` with the following flags if:
- --show: you want to see your image
- --save: you want to save the html 
- --savepng: you want to save the png 
- --saveimg: you want to save the svg 







