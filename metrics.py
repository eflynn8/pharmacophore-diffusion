import torch
from constants import ph_idx_to_type


def compute_complementarity(pharm_feat, pharm_pos, prot_ph_feat, prot_ph_pos, cutoff=8.0):
    #returns fraction of pharmacophore features that are close to a complementary feature in the binding pocket
    
    compatible_types = {
        'Aromatic': ['Aromatic', 'Hydrophobic', 'HydrogenAcceptor'],
        'HydrogenDonor': ['HydrogenAcceptor'],
        'HydrogenAcceptor': ['HydrogenDonor', 'Aromatic'],
        'PositiveIon': ['NegativeIon'],
        'NegativeIon': ['PositiveIon'],
        'Hydrophobic': ['Aromatic', 'Hydrophobic']
    }
    # distances between pharmacophore and receptor features
    distances = torch.cdist(pharm_pos, prot_ph_pos)
    # mask where distance is within the cutoff
    distance_mask = distances <= cutoff
    
    # get feature types from one-hot encoding
    pharm_types = [ph_idx_to_type[idx] for idx in pharm_feat.argmax(dim=1)]
    prot_ph_types = [ph_idx_to_type[idx] for idx in prot_ph_feat.argmax(dim=1)]
    
    # mask of compatible receptor features for each pharmacophore feat
    compatible = torch.tensor([[rec_type in compatible_types[ph_type] for rec_type in prot_ph_types]
                                for ph_type in pharm_types])
    # combine distance and compatibility masks
    mask = distance_mask & compatible
    # count the number of ph features w/ rec complements within cutoff
    compatible_count = mask.any(dim=1).sum()
    
    fraction = compatible_count / len(pharm_feat)
    return fraction