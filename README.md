
# E2E NLG
Code for the INLG 2018 conference paper "Syntactic Manipulation for Generating more Diverse and Interesting Texts"

# Setup & Requirements
We assume an [Anaconda](https://www.anaconda.com/download/) installation. To install the Conda environment run:
```
$ conda env create -f environment.yml
```
This installs tensorflow-gpu==1.10.0, make sure that you have the CUDA environment installed accordingly. Refer to [this](https://www.tensorflow.org/install/install_sources#tested_source_configurations)  for the compatibility.

Next create some output directories, inside the main directory run the following commands:
```
$ mkdir logging
$ mkdir outputs
$ mkdir models
```
# Running the Code
## Preprocessing
First run the preprocessing scrips:
```
$ sh preprocess_data.sh
```
This will handle the whole preprocessing pipeline:
* Parse the Meaning Representations and transform them into vectors.
* Parse the Utterances and transform them into vectors.
* Store the output into "data/e2e_nlg/preprocessed_data/version1"
## Experiments
To run the experiments:
```
$ sh run_experiments.sh
```
This will run all the training procedures and create models for the various synthactic manipulations:
* The models will be stored into the "models/sc_{name_of_experiment}" folder. 
*  Every 10 epochs the outputs of the models are generated and stored into "logging/sc_lstm/{experiment_name}/training_log_{time-stamp}/" folder.

## Generate Outputs
To generate the final outputs run:
```
$ sh run_output_generation.sh
```
This will produce the outputs for each of the model and store it into "outputs/sc_{name_of_experiment}". Here the sampling procedure is used to select the final utterance. Thus, the generation might take a while to complete, especially for the "full" model. 

## Evaluation
To run the rulebased evaluation scripts:
```
$ sh run_evaluation.sh
```
This creates a file called "rule_based_eval.txt" inside the "outputs/sc_{name_of_experiment}" folder. There the incorrect utterances are displayed and the final error rate score.

# References
Please cite the following paper when using this code or pretrained models for your application.

  Jan Deriu and Mark Cieliebak, [*Syntactic Manipulation for Generating more Diverse and Interesting Texts*]() INLG 2018 - [International Conference on Natural Language Generation](https://inlg2018.uvt.nl/)

```
@inproceedings{deriu2018nlg,
  title = {{Syntactic Manipulation for Generating more Diverse and Interesting Texts}},
  author = {Deriu, Jan and Cieliebak, Mark},
  booktitle = {INLG 2018 - International Conference on Natural Language Generation},
  address = {Tilburg, The Netherlands}
  year = {2018},
}
```