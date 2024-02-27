# Set up the invironment

1. Install TrackMania Nations Forever (either from Steam or online).
1. Instal TMInterface 1.4.3, [Download TMInterface 1.4.3](https://donadigo.com/files/TMInterface/TMInterface_1.4.3_Setup.exe), latest very will not work as the tminterface Python library only supports version up to 1.4.3.
1. Create conda environment with the dependencies as listed in requirements.txt.

# Run trained model on train and test track
1. Launch TrackMania game using TMInterface.
1. Drag the train or benchmark Gbx file into the game window and launch the track.
1. Open ```observe.py``` and set map_name varible to the map name.
1. Execute this script in project home directory ``` python scripts/observe.py```.
1. Model will run and vehicle will move in game.

# Create and train new model on any track
1. Launch TrackMania game using TMInterface.
1. Create or open a map and launch the track.
1. Execute ```manual_run.py```, once the code is executed, traverse along the center line of the track and finish the track. Centerline map file will be created and stored in maps folder.
1. If like, can change the reward policy in ```tm_gym_interface.py``` ```step``` function.
1. To initialize training, execute ```init_train.py``` with the map name set to the target map, change hyperparameters if like, and set number of timesteps for this initial training loop. Once training is finished, the model will be stored in runs folder
1. To continue training, execute ```cont_train.py``` with the map name set to the target map, change hyperparameters if like under ```custom_objects```. Change timesteps and iter loops to set how long to train for.
1. Can obersere tensorboard to monitor the training process using ```tensorboard --logdir runs/$run_name$```

