The code is fairly simple to run:

1. First install all packages (correct versions!) listed in requirements.txt, alongside Python 3.8. No CUDA or GPU enabling is needed.

2. To train a new model, open 'main_training.py', set the parameters to your liking and run the code

3. To continue training an already existing model, open 'main_training.py' and make sure that 'RUN_NAME' is identical to the one you want to continue. The code with recognize the previously trained model and continue training it.

4. To view a model perform, open 'main_evaluation.py', enter variables 'MODEL_PATH' and 'LEVEL'. 

For each model, copies throughout the training process are saved in 'data/saved_models/...'. The best performing model is saved under 'data/tmp/...'.

5. To visualize all paths of a model throughout training, use 'training_visualizer.py'. Unfortunately, Github did not allow me push a large number of zip files, so I could not push the history copies of my models. If you somehow still want to see this, send an email to A.S.nokhai@student.tudelft.nl and I will send you a WeTransfer link with the models.

6. To plot data (logged in tensorboard), you can use tensorboard, or 'data_plotter.py'
