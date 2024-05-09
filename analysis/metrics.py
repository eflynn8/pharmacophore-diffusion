import torch
from constants import ph_idx_to_type
from analysis.pharm_builder import SampledPharmacophore
from typing import List

class SampleAnalyzer:

    def analyze(self, sample: List[SampledPharmacophore]):
        
        valid_numerator = 0
        valid_denominator = 0
        for ph in sample:

            prot_ph_feat = ph.g.nodes['prot_ph'].data['h_0']
            prot_ph_pos = ph.g.nodes['prot_ph'].data['x_0']

            # convert protein pharmacophore feature indices to types
            prot_ph_types = [ph_idx_to_type[int(idx)] for idx in prot_ph_feat.argmax(dim=1)]

            kwargs = {
                'pharm_pos': ph.ph_coords,
                'pharm_types': ph.ph_types,
                'prot_ph_pos': prot_ph_pos,
                'prot_ph_types': prot_ph_types
            }
            n_valid_ph_nodes = compute_complementarity(**kwargs)
            n_ph_nodes = ph.n_ph_centers
            valid_numerator += n_valid_ph_nodes
            valid_denominator += n_ph_nodes

        metrics = {
            'validity': valid_numerator / valid_denominator
        }
        return metrics

def compute_complementarity(pharm_types, pharm_pos, prot_ph_types, prot_ph_pos, return_count=False):
    #returns fraction of pharmacophore features that are close to a complementary feature in the binding pocket

    matching_types = {
        'Aromatic': ['Aromatic', 'PositiveIon'],
        'HydrogenDonor': ['HydrogenAcceptor'],
        'HydrogenAcceptor': ['HydrogenDonor'],
        'PositiveIon': ['NegativeIon', 'Aromatic'],
        'NegativeIon': ['PositiveIon'],
        'Hydrophobic': ['Hydrophobic']
    }
    matching_distance = {"Aromatic" : 7, 
        "Hydrophobic": 5,
        "HydrogenAcceptor": 4,
        "HydrogenDonor": 4,
        "NegativeIon": 5,
        "PositiveIon": 5 }

    # distances between pharmacophore and receptor features
    distances = torch.cdist(pharm_pos, prot_ph_pos)
    ph_matching_distances = torch.tensor([matching_distance[ph_type] for ph_type in pharm_types]).reshape(-1, 1)

    # mask of complementary receptor features for each pharmacophore feature
    matching = torch.tensor([[rec_type in matching_types[ph_type] for rec_type in prot_ph_types]
                                for ph_type in pharm_types])

    mask = (distances <= ph_matching_distances) & matching
    complement_count = mask.any(dim=1).sum()

    if return_count:
        return complement_count
    else:
        fraction = complement_count / len(pharm_feat)
        return fraction