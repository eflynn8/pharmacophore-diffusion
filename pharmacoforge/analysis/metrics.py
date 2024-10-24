import numpy as np
import torch
from pharmacoforge.constants import ph_idx_to_type
from pharmacoforge.analysis.pharm_builder import SampledPharmacophore
from typing import List

class SampleAnalyzer:

    def analyze(self, sample: List[SampledPharmacophore]):
        
        valid_numerator = 0
        valid_denominator = 0
        for ph in sample:
            n_valid_ph_nodes = ph.compute_n_valid_centers()
            n_ph_nodes = ph.n_ph_centers
            valid_numerator += n_valid_ph_nodes
            valid_denominator += n_ph_nodes

        metrics = {
            'validity': (valid_numerator / valid_denominator).item()
        }
        return metrics

    def pharm_feat_freq(self, sample: List[SampledPharmacophore]):
        total = 0
        type_counts = torch.from_numpy(np.zeros(6))

        for ph in sample:
            pharm_feat = ph.g.nodes['pharm'].data['h_0']

            if pharm_feat.shape[1] > 1: # if it is one-hot encoded (this is the case for diffusion)
                pharm_types = pharm_feat.argmax(dim=1)
            else:
                pharm_types = pharm_feat.flatten() # otherwise, pharm_feat has shape (n_nodes,1)

            total += len(pharm_types)

            for val in pharm_types:
                type_counts[val] += 1
        
        # return torch.div(type_counts, total)
        return type_counts


