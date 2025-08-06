###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/media/data/Down/d2d/nuplan/dataset/nuplan-v1.1/mini" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/media/data/Down/d2d/nuplan/dataset/maps/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/media/data/Down/d2d/nuplan_preplan" # preprocess training data
###################################

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000 \

