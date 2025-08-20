from pathlib import Path
from supercells import Supercell

if __name__=='__main__':
    Gr_a_revPBE = 2.472387181
    distance = 3.35
    Co_a_revPBE = 2.428759230
    Co_c_revPBE = 3.920363262

    GrMe = Supercell()
    GrMe.distance = distance
    GrMe.set_gr_a(Gr_a_revPBE)

    GrMe.set_me("Co", "hcp0001", Co_a_revPBE)
    cells = GrMe.search_supercell(10, 6, id=0, beta_fix=False, beta_min=0., eq_abs=False, eq_eps=4, csv=True, textmode=True)