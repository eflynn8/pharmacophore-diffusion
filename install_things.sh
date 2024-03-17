set -e
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba install -c conda-forge pytorch-lightning -y
mamba install pytorch-cluster pytorch-scatter -c pyg -y
mamba install -c dglteam/label/cu121 dgl -y
mamba install -c conda-forge rdkit -y
mamba install -c conda-forge openbabel -y
mamba install -c conda-forge pandas -y
mamba install -c conda-forge biopython -y
# i also had an error when importing dgl and had to install the following manually
mamba install -c conda-forge pydantic -y
mamba install pytorch::torchdata -y
mamba install -c conda-forge wandb -y