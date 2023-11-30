from collections import defaultdict
import os
import glob
import h5py
import torch
import warnings

import numpy as np

from tqdm import tqdm

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from Bio.PDB import PDBParser


def bio_load_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    protein = parser.get_structure(0, pdb_file)
    residues = [residue for residue in protein.get_residues()]
    residue_type = [data.Protein.residue2id[residue.get_resname()] for residue in residues]
    residue_number = [residue.full_id[3][1] for residue in residues]
    id2residue = {residue.full_id: i for i, residue in enumerate(residues)}
    residue_feature = functional.one_hot(torch.as_tensor(residue_type), len(data.Protein.residue2id)+1)

    atoms = [atom for atom in protein.get_atoms()]
    atoms = [atom for atom in atoms if atom.get_name() in data.Protein.atom_name2id]
    occupancy = [atom.get_occupancy() for atom in atoms]
    b_factor = [atom.get_bfactor() for atom in atoms]
    atom_type = [data.feature.atom_vocab.get(atom.get_name()[0], 0) for atom in atoms]
    atom_name = [data.Protein.atom_name2id.get(atom.get_name(), 37) for atom in atoms]
    node_position = np.stack([atom.get_coord() for atom in atoms], axis=0)
    node_position = torch.as_tensor(node_position)
    atom2residue = [id2residue[atom.get_parent().full_id] for atom in atoms]

    edge_list = [[0, 0, 0]]
    bond_type = [0]

    return data.Protein(edge_list, atom_type=atom_type, bond_type=bond_type, residue_type=residue_type,
                num_node=len(atoms), num_residue=len(residues), atom_name=atom_name, 
                atom2residue=atom2residue, occupancy=occupancy, b_factor=b_factor,
                residue_number=residue_number, node_position=node_position, residue_feature=residue_feature
            ), "".join([data.Protein.id2residue_symbol[res] for res in residue_type])

@R.register("datasets.V5")
class V5(data.ProteinDataset):

    processed_file = "v5.pkl.gz"

    def __init__(self, path, split_ratio=(0.6, 0.2, 0.2), discrete_label=False, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.split_ratio = split_ratio
        self.discrete_label = discrete_label

        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = sorted(glob.glob(os.path.join(path, 'pdb', "*.pdb")))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        label_file = os.path.join(path, "20230927tada.csv")
        label_list = self.get_label_list(label_file)
        activitity = [label_list[os.path.basename(pdb_file)[:-4]] for pdb_file in self.pdb_files]
        self.targets = {"activity": activitity}
        if self.discrete_label:
            num_labels = defaultdict(int)
            for i in activitity:
                num_labels[i] += 1
            print(num_labels)

    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        num_sample = len(pdb_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            protein, sequence = bio_load_pdb(pdb_file)
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(sequence)

    def get_label_list(self, label_file):
        with open(label_file, "r") as fin:
            lines = [line.strip() for line in fin.readlines()][2:]
        label_list = {}
        for line in lines:
            name, sequence, activity = line.split(",")
            activity = float(activity)
            if self.discrete_label:
                label_list[name] = (activity > 5)
            else:
                label_list[name] = activity
        return label_list

    def split(self, split_ratio=None):
        split_ratio = split_ratio or self.split_ratio
        num_samples = [int(len(self) * ratio) for ratio in split_ratio]
        num_samples[-1] = len(self) - sum(num_samples[:-1])
        splits = torch.utils.data.random_split(self, num_samples)
        return splits
    
    def get_item(self, index):
        if self.lazy:
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein, "activity": self.targets["activity"][index]}
        if self.transform:
            item = self.transform(item)
        return item
    

@R.register("datasets.V5Batch")
class V5Batch(data.ProteinDataset):

    processed_file = "v5batch.pkl.gz"

    def __init__(self, path, split_ratio=(0.6, 0.2, 0.2), verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.split_ratio = split_ratio

        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = sorted(glob.glob(os.path.join(path, "*.pdb")))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        label_file = os.path.join(path, "5V_batch.csv")
        batch_dict = self.get_label_list(label_file)
        pdb2id = {
            os.path.basename(pdb_file)[:-4]: i 
                for i, pdb_file in enumerate(self.pdb_files)
        }
        self.pairs = []
        for _, data_list in batch_dict.items():
            v = [x[1] for x in data_list]
            _min, _max = min(v), max(v)
            norm_act = []
            for x in data_list:
                norm_act.append((x[1] - _min) / (_max - _min))
            for i in range(len(norm_act)):
                for j in range(len(norm_act)):
                    if norm_act[i] - norm_act[j] > 0.5:
                        id1 = pdb2id[data_list[i][0]]
                        id2 = pdb2id[data_list[j][0]]
                        self.pairs.append((id1, id2))

    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        num_sample = len(pdb_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            protein, sequence = bio_load_pdb(pdb_file)
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(sequence)

    def get_label_list(self, label_file):
        with open(label_file, "r") as fin:
            lines = [line.strip() for line in fin.readlines()][3:]
        batch_dict = defaultdict(list)
        for line in lines:
            name, sequence, activity1, activity2, batch = line.strip().split(",")
            activity2 = float(activity2)
            batch_dict[batch].append((name, activity2))
        return batch_dict

    def split(self, split_ratio=None):
        split_ratio = split_ratio or self.split_ratio
        num_samples = [int(len(self) * ratio) for ratio in split_ratio]
        num_samples[-1] = len(self) - sum(num_samples[:-1])
        splits = torch.utils.data.random_split(self, num_samples)
        return splits
    
    def get_item(self, index):
        id1, id2 = self.pairs[index]
        if self.lazy:
            protein1 = data.Protein.from_pdb(self.pdb_files[id1], self.kwargs)
            protein2 = data.Protein.from_pdb(self.pdb_files[id2], self.kwargs)
        else:
            protein1 = self.data[id1].clone()
            protein2 = self.data[id2].clone()
        if hasattr(protein1, "residue_feature"):
            with protein1.residue():
                protein1.residue_feature = protein1.residue_feature.to_dense()
        if hasattr(protein2, "residue_feature"):
            with protein2.residue():
                protein2.residue_feature = protein2.residue_feature.to_dense()
        item = {"graph1": protein1, "graph2": protein2}
        if self.transform:
            item = self.transform(item)
        return item
    
    def __len__(self):
        return len(self.pairs)
    
    @property
    def tasks(self):
        """List of tasks."""
        return ["activity pair"]