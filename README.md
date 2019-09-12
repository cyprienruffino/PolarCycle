# PolarCycle
Extended CycleGAN for RGB-to-polarimetric image transfer

Supports up to 2 GPUs

## Usage 
python train_polarcycle.py config_file rgb_path polar_path

Outputs a TensorBoard log file and the models checkpoints in the 'runs' directory 

## Configuration
See example_config.py

## Dependencies
- numpy
- tensorflow >= 1.10
- python-opencv
- progressbar2

## TODO
Decreasing learning rate (currently Adam)