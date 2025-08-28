from supercells import Supercell
from ase.io import write
from IPython.display import Image, display

if __name__=='__main__':
    """
    Usage Example
    """
    Pt_fcc110 = Supercell()
    Pt_fcc110.set_sub(title_sub='Pt', lat_sub='fcc110')
    Pt_fcc110_supercells = Pt_fcc110.search_supercell(radius=20, eps_max=6, beta_fix=False, eq_abs=False, eq_eps=4, csv=True)

    atoms = Pt_fcc110.build_supercell(gs_supercell=Pt_fcc110_supercells[2], n_sub_layers=3, directory_res='./data')
    write('data/Pt_fcc110_1.xyz', atoms)
    write('data/Pt_fcc110_1.png', atoms)

    display(Image(filename='data/Pt_fcc110_1.png'))

    supercell_test = Pt_fcc110_supercells[2]
    eps_matrix = Pt_fcc110.compute_mismatch_in_basis(supercell_test)
    print('Mismatch matrix: \n', eps_matrix)
    eps_cc, eps_orth_cc = Pt_fcc110.compute_deform_of_cc(gs_supercell=supercell_test)
    print(f'Deformation: {eps_cc:.4f} {eps_orth_cc:.4f}')