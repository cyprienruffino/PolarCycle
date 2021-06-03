cd dev/pro/PolarCycle
source env/bin/activate
rm -rf runs/projector_diff_polarcycle_config
python src/train_polarcycle.py projector_diff_polarcycle_config.py ../input/rgb2pol/rgb1500_resized ../input/rgb2pol/polar1500_merged