from pymol import cmd
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--pocket_dir', type=str, default=None, help='pocket directory that contains the generated pharmacophores')
p.add_argument('--load_reference', action='store_true', help='load reference protein and ligand')
args = p.parse_args()

#load reference receptor and ligand
parent_dir = Path(args.pocket_dir)
if args.load_reference:
    reference_dir = parent_dir / 'reference_files'
    #the PDB is the reference protein and the SDF is the reference ligand
    #load the PDB and SDF into pymol and give them the names 'reference_protein' and 'reference_ligand'
    for file in sorted(reference_dir.iterdir()):
        if file.suffix == '.pdb':
            cmd.load(str(file), 'reference_protein')
        elif file.suffix == '.sdf':
            cmd.load(str(file), 'reference_ligand')

#load the pocket file
cmd.load(str(parent_dir / 'pocket.pdb'), 'pocket')

#load the pharmacophores
for file in sorted(parent_dir.iterdir()):
    if file.suffix == '.xyz':
        cmd.load(str(file), file.stem)
        pharmacophore_file=file.stem
        break

#avoid bond inference on loading pharmacophores
cmd.show_as('spheres',pharmacophore_file)
    
#create seperate trajectories with the following of certain atom types present in the xyz files
cmd.select('PositiveIon', 'elem N and '+ pharmacophore_file )
cmd.create('PositiveIon', 'PositiveIon')
cmd.select('Hydrophobic', 'elem C and ' + pharmacophore_file )
cmd.create('Hydrophobic', 'Hydrophobic')
cmd.select('NegativeIon', 'elem O and ' + pharmacophore_file)
cmd.create('NegativeIon', 'NegativeIon')
cmd.select('Aromatic', 'elem P and ' + pharmacophore_file)
cmd.create('Aromatic', 'Aromatic')
cmd.select('HydrogenAcceptor', 'elem F and ' + pharmacophore_file)
cmd.create('HydrogenAcceptor', 'HydrogenAcceptor')
cmd.select('HydrogenDonor', 'elem S and ' + pharmacophore_file)
cmd.create('HydrogenDonor', 'HydrogenDonor')

#avoid ultra large spheres
select_string=' or '.join([pharmacophore_file,'PositiveIon','Hydrophobic','NegativeIon','Aromatic','HydrogenAcceptor','HydrogenDonor'])
cmd.set('sphere_scale',0.2,select_string)