# Protein-Activity-Prediction


## Dependencies
```bash
conda install pytorch=1.12.1 cudatoolkit=11.1 -c milagraph -c pytorch-lts -c pyg -c conda-forge
conda install biopython
conda install easydict pyyaml -c conda-forge
git clone git@github.com:DeepGraphLearning/torchdrug.git
cd torchdrug
python setup.py develop
```

## Training
```bash
# Please put all pdbs and data.csv in the dataset directory, which should be set in the yaml file.
python script/train.py -c config/V5/gearnet_edge.yaml
```

## Inference
```bash
# Put all pdb files in the <path_to_dataset>
python script/inference.py -c config/V5/gearnet_edge.yaml --ckpt <path_to_ckpt> --dataset <paht_to_dataset>
```