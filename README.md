# PolarCycle
Extended CycleGAN for RGB-to-polarimetric image transfer. This anonymous repository is associated to the ACCV submission "Generating Polarimetric-encoded Images using Constrained Cycle-Consistent Generative Adversarial Networks" and is here mainly for reproducibility purposes.

The code for the polarimetric RetinaNet if from [rachelblin's github](https://github.com/RachelBlin/keras-retinanet), which is itself adapted from  [fizyr's github](https://github.com/fizyr/keras-retinanet).

Anonymous repository: [https://anonymous.4open.science/r/4a83820e-9c65-417c-af3a-ab2979d6e2e8/]

## Features
- Supports up to 2 GPUs
- Outputs a TensorBoard log file and the model checkpoints in the 'runs' directory
- The model checkpoints are at the Keras hdf5 saved model format. To load a model, use keras.models.load_model(path_to_model)

## Configuration
See polarcycle_config.py and vanilla_cyclegan_config.py

## Usage 
### Training the models from scratch
```shell
python -m deeplauncher --config_path config_file --datasets-paths rgb_path polar_path
```
Takes around two days to train on 2 NVidia GTX1080Ti for the 2485 images datasets. 

### Resuming training
```shell
python -m deeplauncher --config_path config_file --epoch epoch --resume_path --datasets-paths rgb_path polar_path
```
### Generating samples
```shell
python src/generate_samples.py checkpoint_path files_path output_path
```
### Evaluating FID
Needs [keras_retinanet](https://github.com/fizyr/keras-retinanet) to restore the model
```shell
python scripts/evaluate_fid.py fid_model real_imgs fake_imgs
```

## Provided models
- cyclegan_genRGBtoPolar_399.hdf5: Vanilla CycleGAN RGB to polarimetric generator
- polarcycle_genRGBtoPolar_399.hdf5: Our extended CycleGAN RGB to polarimetric generator

## Dependencies
- python3
- numpy
- tensorflow >= 1.10
- python-opencv
- progressbar2
- [keras_retinanet](https://github.com/fizyr/keras-retinanet) (For computing the FID only)
