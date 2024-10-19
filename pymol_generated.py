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

# find all xyz files in the directory - these are generated pharmacophores
ph_files = [file for file in parent_dir.glob('*.xyz')]
ph_pymol_names = [file.stem for file in ph_files] # names of pharmacophores once loaded into pymol
all_ph_sel_str = ' or '.join(ph_pymol_names)


#load the pharmacophores
for ph_file, ph_pymol_name in zip(ph_files, ph_pymol_names):
    cmd.load(str(ph_file))
    cmd.unbond(ph_pymol_name, ph_pymol_name) # remove all bonds between atoms in the pharmacophore
    cmd.show_as('spheres', ph_pymol_name) # show the pharmacophore as spheres

# set the sphere scale for all pharmacophores
cmd.set('sphere_scale', 0.4, all_ph_sel_str)

    
# create selections for each pharmacophore type
cmd.select('PositiveIon', f'elem N and ({all_ph_sel_str})' )
cmd.select('Hydrophobic', f'elem C and ({all_ph_sel_str})' )
cmd.select('NegativeIon', f'elem O and ({all_ph_sel_str})')
cmd.select('Aromatic', f'elem P and ({all_ph_sel_str})')
cmd.select('HydrogenAcceptor', f'elem F and ({all_ph_sel_str})')
cmd.select('HydrogenDonor', f'elem S and ({all_ph_sel_str})')
cmd.select('Mask', f'elem Se and ({all_ph_sel_str})')


pymol_color_map = {
    'Aromatic': 'purple',
    'Hydrophobic': 'green',
    'HydrogenAcceptor': 'orange',
    'HydrogenDonor': 'white',
    'PositiveIon': 'blue',
    'NegativeIon': 'red',
    'Mask': 'grey'
}

# color the pharmacophores
for ph_type, color in pymol_color_map.items():
    cmd.color(color, ph_type)
