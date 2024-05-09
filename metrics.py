import torch
from constants import ph_idx_to_type

def compute_complementarity(pharm_feat, pharm_pos, prot_ph_feat, prot_ph_pos):
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
    
    pharm_types = [ph_idx_to_type[idx] for idx in pharm_feat.argmax(dim=1)]
    prot_ph_types = [ph_idx_to_type[idx] for idx in prot_ph_feat.argmax(dim=1)]

    # distances between pharmacophore and receptor features
    distances = torch.cdist(pharm_pos, prot_ph_pos)
    ph_matching_distances = torch.tensor([matching_distance[ph_type] for ph_type in pharm_types]).reshape(-1, 1)

    # mask of complementary receptor features for each pharmacophore feature
    matching = torch.tensor([[rec_type in matching_types[ph_type] for rec_type in prot_ph_types]
                                for ph_type in pharm_types])

    mask = (distances <= ph_matching_distances) & matching
    complement_count = mask.any(dim=1).sum()
    
    fraction = complement_count / len(pharm_feat)
    return fraction