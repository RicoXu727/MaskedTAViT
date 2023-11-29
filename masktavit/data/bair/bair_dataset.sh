
mkdir -p tempData/bair
wget -P tempData/bair/ http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
tar -xvf tempData/bair/bair_robot_pushing_dataset_v0.tar -C tempData/bair/

python3 masktavit/data/bair/bair_extract_images.py --data_dir tempData/bair
python3 masktavit/data/bair/bair_image_to_hdf5.py --data_dir tempData/bair --output_dir datasets

