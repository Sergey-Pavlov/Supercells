import os
import time
from pathlib import Path
import math as m
import numpy as np
import pandas as pd

from ase.build import fcc100
from ase.build import fcc110
from ase.build import fcc111
from ase.build import bcc100
from ase.build import bcc110
from ase.build import bcc111
from ase.build import hcp0001
from ase.build import diamond100
from ase.build import diamond111


def group_el(array, elements_eq):
    """ elements_eq - func: f(el_1, el_2) -> bool  """
    ar_group = []
    used_indices = set()

    for i in range(len(array)):
        if i in used_indices:
            continue

        group = [array[i]]
        used_indices.add(i)
        for j in range(i + 1, len(array)):
            if j in used_indices:
                continue

            if elements_eq(array[i], array[j]):
                group.append(array[j])
                used_indices.add(j)
        ar_group.append(group)

    return ar_group


def format_write(numbers):
    str = ""
    Num = len(numbers)
    for i in range(Num):
        if isinstance(numbers[i], int):
            formatted_num = f"{numbers[i]}"
            N = 4
        else:
            if numbers[i] > 1000:
                formatted_num = "{:.1f}".format(numbers[i])
            else:
                formatted_num = "{:.2f}".format(numbers[i])
            N = 7

        num_length = len(formatted_num)
        num_spaces = max(0, N - num_length)
        if i == Num - 1:
            str += formatted_num
        else:
            str += formatted_num + "," + " " * num_spaces

    return str


def angle_v(v1, v2):
    """ Return angle between vectors in degree """
    angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if angle >= 1:
        return 0
    elif angle <= -1:
        return 180
    else:
        return np.degrees(np.arccos(angle))


def acute_angle_v(v1, v2):
    """ Return acute angle between vectors in degree """
    angle = angle_v(v1, v2)
    if angle > 90:
        angle = 180 - angle
    return angle


def single_angle_v(v1, v2):
    """
        if return +1 - First vector need rotate counterclockwise
        if return -1 - First vector need rotate clockwise
    """
    v11 = [v1[0], v1[1], 0]
    v12 = [v2[0], v2[1], 0]
    n = np.cross(v11, v12)
    if n[2] > 0:
        return 1
    else:
        return -1


def rotate_matrix(angle):
    cos = np.cos(np.radians(angle))
    sin = np.sin(np.radians(angle))
    return np.array([[cos, -sin], [sin, cos]])


def linear_map(system, M):
    for atom in system:
        vector = np.array([atom[0], atom[1]])
        vector = np.dot(M, vector)
        atom[0] = vector[0]
        atom[1] = vector[1]
    return system


def eq(a, b, eps):
    return abs(a - b) <= eps


def divite(x, y):
    """ остаток от деления максимального числа на минимальное """
    if x > y:
        max = x
        min = y
    else:
        max = y
        min = x
    return abs((max / min) % 1)


class Supercell:
    def __init__(self, title_me: str = "", lat_me: str = "", z_dist: float = 25., distance: float = 3):

        self.z_dist = z_dist
        self.distance = distance

        self.gr_a_exp = 2.4595

        self.gr_a = self.gr_a_exp
        self.compute_gr_a()

        self.me_a_3d_exp = 1.0
        self.me_c_exp = 1.0
        if title_me != "":
            self.title_me = title_me
            match title_me:
                case "Ag":
                    self.me_a_3d_exp = 4.0860
                case "Au":
                    self.me_a_3d_exp = 4.0786
                case 'Pt':
                    self.me_a_3d_exp = 3.9236
                case 'Al':
                    self.me_a_3d_exp = 4.0493
                case 'Ni':
                    self.me_a_3d_exp = 3.5241
                case 'Ag':
                    self.me_a_3d_exp = 4.086
                case 'Cu':
                    self.me_a_3d_exp = 3.615
                case 'Ir':
                    self.me_a_3d_exp = 3.890
                case 'Pd':
                    self.me_a_3d_exp = 3.840
                case 'Fe':
                    self.me_a_3d_exp = 2.866
                case 'Co':
                    self.me_a_3d_exp = 2.505
                    self.me_c_exp = 4.089

        self.me_a_3d = self.me_a_3d_exp
        self.me_c = self.me_c_exp
        self.lat_me = lat_me
        self.compute_me_a()

        self.mg_supercells = []

    def compute_me_a(self):
        if self.lat_me == "":
            self.me_a1 = 1.0
            self.me_a2 = self.me_a1
            self.m1 = np.array([self.me_a1, 0.0])
            self.m2 = np.array([0.0, self.me_a2])

        if self.lat_me == "fcc100":
            self.me_a1 = self.me_a_3d * 2 ** 0.5 / 2
            self.me_a2 = self.me_a1
            self.m1 = np.array([self.me_a1, 0.0])
            self.m2 = np.array([0.0, self.me_a2])

        if self.lat_me == "fcc110":
            self.me_a1 = self.me_a_3d
            self.me_a2 = self.me_a_3d * 2 ** 0.5 / 2
            self.m1 = np.array([self.me_a1, 0.0])
            self.m2 = np.array([0.0, self.me_a2])

        if self.lat_me == "fcc111":
            self.me_a1 = self.me_a_3d * 2 ** 0.5 / 2
            self.me_a2 = self.me_a1
            self.m1 = np.array([self.me_a1, 0.0])
            self.m2 = np.array([self.me_a2 * np.cos(np.radians(60.0)), self.me_a2 * np.sin(np.radians(60.0))])

        if self.lat_me == "bcc100":
            self.me_a1 = self.me_a_3d
            self.me_a2 = self.me_a1
            self.m1 = np.array([self.me_a1, 0.0])
            self.m2 = np.array([0.0, self.me_a2])

        if self.lat_me == "bcc110":
            angle_tmp = np.arccos(1 / 3 ** 0.5)
            self.me_a1 = self.me_a_3d
            self.me_a2 = self.me_a_3d * 3 ** 0.5 / 2
            self.m1 = np.array([self.me_a1, 0.0])
            self.m2 = np.array([self.me_a2 * np.cos(angle_tmp), self.me_a2 * np.sin(angle_tmp)])

        if self.lat_me == "bcc111":
            self.me_a1 = self.me_a_3d * 2 ** 0.5
            self.me_a2 = self.me_a1
            self.m1 = np.array([self.me_a1, 0.0])
            self.m2 = np.array([self.me_a2 * np.cos(np.radians(60.0)), self.me_a2 * np.sin(np.radians(60.0))])

        if self.lat_me == "hcp0001":
            self.me_a1 = self.me_a_3d
            self.me_a2 = self.me_a1
            self.m1 = np.array([self.me_a1, 0.0])
            self.m2 = np.array([self.me_a2 * np.cos(np.radians(60.0)), self.me_a2 * np.sin(np.radians(60.0))])

        if self.lat_me == "diamond100":
            self.me_a1 = self.me_a_3d * 2 ** 0.5 / 2
            self.me_a2 = self.me_a1
            self.m1 = np.array([self.me_a1, 0.0])
            self.m2 = np.array([0.0, self.me_a2])

        if self.lat_me == "diamond111":
            self.me_a1 = self.me_a_3d * 2 ** 0.5 / 2
            self.me_a2 = self.me_a1
            self.m1 = np.array([self.me_a1, 0.0])
            self.m2 = np.array([self.me_a2 * np.cos(np.radians(60.0)), self.me_a2 * np.sin(np.radians(60.0))])

    def compute_m_max(self, radius):
        if self.lat_me == "fcc100":
            return int(m.ceil(radius / self.me_a1))

        if self.lat_me == "fcc110":
            return int(m.ceil(radius / self.me_a2))

        if self.lat_me == "fcc111":
            return int(m.ceil(radius / self.me_a1 / np.sin(np.radians(60.0))))

        if self.lat_me == "bcc100":
            return int(m.ceil(radius / self.me_a1))

        if self.lat_me == "bcc110":
            return int(m.ceil(radius / self.me_a2 / np.sin(np.arccos(1 / 3 ** 0.5))))

        if self.lat_me == "bcc111":
            return int(m.ceil(radius / self.me_a1 / np.sin(np.radians(60.0))))

        if self.lat_me == "hcp0001":
            return int(m.ceil(radius / self.me_a1 / np.sin(np.radians(60.0))))

        if self.lat_me == "diamond100":
            return int(m.ceil(radius / self.me_a1))

        if self.lat_me == "diamond111":
            return int(m.ceil(radius / self.me_a1 / np.sin(np.radians(60.0))))

    def compute_gr_a(self):
        self.g1 = np.array([self.gr_a, 0.0])
        self.g2 = np.array([self.gr_a * np.cos(np.radians(60.0)), self.gr_a * np.sin(np.radians(60.0))])
        self.cc = np.linalg.norm(np.array([self.gr_a / 2, self.gr_a * 3 ** 0.5 / 6]))

    def set_gr_a(self, a):
        """  """
        self.gr_a = a
        self.compute_gr_a()

    def set_me(self, title_me: str = "", lat_me: str = "", a: float = -1.0, c: float = -1.0):
        if title_me != "":
            self.title_me = title_me
        if lat_me != "":
            self.lat_me = lat_me
        if not a < 0:
            self.me_a_3d = a
        if not c < 0:
            self.me_c = c

        self.compute_me_a()

    def build_me(self, size1, size2, n_me_layers):
        if self.lat_me == "fcc100":
            Me = fcc100(self.title_me, size=(size1, size2, n_me_layers), a=self.me_a_3d, vacuum=0.0)
        if self.lat_me == "fcc110":
            Me = fcc110(self.title_me, size=(size1, size2, n_me_layers), a=self.me_a_3d, vacuum=0.0)
        if self.lat_me == "fcc111":
            Me = fcc111(self.title_me, size=(size1, size2, n_me_layers), a=self.me_a_3d, vacuum=0.0)

        if self.lat_me == "bcc100":
            Me = bcc100(self.title_me, size=(size1, size2, n_me_layers), a=self.me_a_3d, vacuum=0.0)
        if self.lat_me == "bcc110":
            Me = bcc110(self.title_me, size=(size1, size2, n_me_layers), a=self.me_a_3d, vacuum=0.0)
        if self.lat_me == "bcc111":
            Me = bcc111(self.title_me, size=(size1, size2, n_me_layers), a=self.me_a_3d, vacuum=0.0)

        if self.lat_me == "hcp0001":
            Me = hcp0001(self.title_me, size=(size1, size2, n_me_layers), a=self.me_a_3d, c=self.me_c, vacuum=0.0)
        if self.lat_me == "diamond100":
            Me = diamond100(self.title_me, size=(size1, size2, n_me_layers), a=self.me_a_3d, vacuum=0.0)
        if self.lat_me == "diamond111":
            Me = diamond111(self.title_me, size=(size1, size2, n_me_layers), a=self.me_a_3d, vacuum=0.0)

        return np.array(Me.get_positions())

    def eq_alpha(self, cell1, cell2):
        if eq(cell1["alpha"], cell2["alpha"], 1.e-5):
            return True
        if round(abs(cell1["alpha"] - cell2["alpha"]), 5) % 60 <= 1.e-5:
            return True
        if round(abs(cell1["alpha"] + cell2["alpha"]), 5) % 60 <= 1.e-5:
            return True
        return False

    def eq_eps(self, cell1, cell2):
        if eq(cell1["eps1"], cell2["eps1"], 1.e-5) and eq(cell1["eps2"], cell2["eps2"], 1.e-5):
            return True
        if eq(cell1["eps1"], cell2["eps2"], 1.e-5) and eq(cell1["eps2"], cell2["eps1"], 1.e-5):
            return True
        return False

    def proportional(self, cell1, cell2):
        divite1 = divite(cell1["me"]["V1_abs"], cell2["me"]["V1_abs"])
        divite2 = divite(cell1["me"]["V2_abs"], cell2["me"]["V2_abs"])
        divite3 = divite(cell1["me"]["V1_abs"], cell2["me"]["V2_abs"])
        divite4 = divite(cell1["me"]["V2_abs"], cell2["me"]["V1_abs"])

        return (divite1 <= 1.e-5 and divite2 <= 1.e-5) or (divite3 <= 1.e-5 and divite4 <= 1.e-5)

    def on_bounary(self, atom, L1, L2):
        tmp_v = atom - L2
        if abs(acute_angle_v(tmp_v, L1)) < 1.e-5:
            return True
        tmp_v = atom - L1
        if abs(acute_angle_v(tmp_v, L2)) < 1.e-5:
            return True

    def atom_not_inside(self, atom, l1, l2):
        eps = 1.e-5
        matrix = np.array([l1, l2]).T
        try:
            uv = np.linalg.solve(matrix, atom)
            u, v = uv
        except np.linalg.LinAlgError:
            return True
        if (-eps <= u <= 1 + eps) and (-eps <= v <= 1 + eps):
            return False
        else:
            return True

    def good_cell(self, abs1, abs2, beta, beta_fix, eq_abs, beta_min, beta_max):

        if eq_abs:
            if not eq(abs1, abs2, 1.e-5):
                return False

        if type(beta_fix) is bool:
            if beta < beta_min or beta > beta_max:
                return False
        else:
            good_beta = False
            for beta_fix_one in beta_fix:
                if eq(beta, beta_fix_one, 1.e-5):
                    good_beta = True
            if not good_beta:
                return False
        return True

    def eq_cell(self, cell1, cell2):

        if eq(cell1["beta"], cell2["beta"], 1.e-10):
            if eq(cell1["V1_abs"], cell2["V1_abs"], 1.e-10) and eq(cell1["V2_abs"], cell2["V2_abs"], 1.e-10):
                return True
            if eq(cell1["V1_abs"], cell2["V2_abs"], 1.e-10) and eq(cell1["V2_abs"], cell2["V1_abs"], 1.e-10):
                return True
        return False

    def eq_mg_supercell(self, cell1, cell2):
        if not eq(cell1["me"]["beta"], cell2["me"]["beta"], 1.e-5):
            return False
        if not self.eq_alpha(cell1, cell2):
            return False
        if not self.eq_eps(cell1, cell2):
            return False
        if not self.proportional(cell1, cell2):
            return False
        return True

    def compute_alpha(self, cell1, cell2, basis1, basis2, textmode=False):

        if cell1["V1_abs"] > cell1["V2_abs"]:
            V11 = cell1["V1"]
            V12 = cell1["V2"]
        else:
            V11 = cell1["V2"]
            V12 = cell1["V1"]
        if cell2["V1_abs"] > cell2["V2_abs"]:
            V21 = cell2["V1"]
            V22 = cell2["V2"]
        else:
            V21 = cell2["V2"]
            V22 = cell2["V1"]

        if not eq(cell1["beta"], cell2["beta"], 1.e-5):
            V11 = -V11

        med_cell1 = V11 + V12
        med_cell1 /= np.linalg.norm(med_cell1)
        med_cell2 = V21 + V22
        med_cell2 /= np.linalg.norm(med_cell2)

        med_basis1 = np.array(basis1[0]) + np.array(basis1[1])
        med_basis2 = np.array(basis2[0]) + np.array(basis2[1])
        med_basis1 /= np.linalg.norm(med_basis1)
        med_basis2 /= np.linalg.norm(med_basis2)

        V21_abs = np.linalg.norm(V21)
        V22_abs = np.linalg.norm(V22)
        if single_angle_v(V11, V12) != single_angle_v(V21, V22) and not eq(V21_abs, V22_abs, 1.e-5):
            i = np.array([-med_cell1[1], med_cell1[0]])
            H = np.eye(2) - 2 * np.outer(i, i)
            med_basis1 = np.dot(H, med_basis1)

            if textmode:
                print('reflex')

        angle = angle_v(med_cell2, med_cell1)
        single = single_angle_v(med_cell2, med_cell1)
        R = rotate_matrix(angle)
        if single == 1:
            R = np.transpose(R)
        med_basis1 = np.dot(R, med_basis1)

        alpha = acute_angle_v(med_basis1, med_basis2)
        if textmode:
            print(f'{alpha:.3f}')

        return float(alpha)

    def create_supercell(self, config):
        k11 = int(config[0])
        k12 = int(config[1])
        k21 = int(config[2])
        k22 = int(config[3])
        l11 = int(config[4])
        l12 = int(config[5])
        l21 = int(config[6])
        l22 = int(config[7])
        eps_oth = bool(config[8])
        G1 = k11 * self.g1 + k12 * self.g2
        G2 = k21 * self.g1 + k22 * self.g2
        M1 = l11 * self.m1 + l12 * self.m2
        M2 = l21 * self.m1 + l22 * self.m2
        gr_cell = {"V1": G1, "V1_abs": np.linalg.norm(G1), "n11": k11, "n12": k12, "n21": k21, "n22": k22,
                   "V2": G2, "V2_abs": np.linalg.norm(G2), "beta": acute_angle_v(G1, G2)}
        me_cell = {"V1": M1, "V1_abs": np.linalg.norm(M1), "n11": l11, "n12": l12, "n21": l21, "n22": l22,
                   "V2": M2, "V2_abs": np.linalg.norm(M2), "beta": acute_angle_v(M1, M2)}

        if not eps_oth:
            eps1 = (me_cell["V1_abs"] - gr_cell["V1_abs"]) / gr_cell["V1_abs"] * 100
            eps2 = (me_cell["V2_abs"] - gr_cell["V2_abs"]) / gr_cell["V2_abs"] * 100
        else:
            eps1 = (me_cell["V1_abs"] - gr_cell["V2_abs"]) / gr_cell["V2_abs"] * 100
            eps2 = (me_cell["V2_abs"] - gr_cell["V1_abs"]) / gr_cell["V1_abs"] * 100

        alpha = self.compute_alpha(gr_cell, me_cell, [self.g1, self.g2], [self.m1, self.m2])
        S = me_cell["V1_abs"] * me_cell["V2_abs"] * np.sin(np.radians(me_cell["beta"]))

        return {"me": me_cell, "gr": gr_cell, "eps1": eps1, "eps2": eps2, "eps_oth": eps_oth, "alpha": alpha, "S": S}

    def read_cell_from_file(self, filepath, title):
        try:
            with open(Path(filepath), 'r') as f:
                if title == "me":
                    tmp = f.readline().split()
                    self.title_me = tmp[0]
                    self.lat_me = tmp[1]
                    self.me_a_3d = float(f.readline())
                    self.compute_me_a()
                    b1 = self.m1
                    b2 = self.m2
                elif title == "gr":
                    self.gr_a = float(f.readline())
                    self.compute_gr_a()
                    b1 = self.g1
                    b2 = self.g2
                else:
                    return False, None

                cells = []
                tmp = f.readline()
                for line in f:
                    tmp = line.split()
                    n11 = int(tmp[0])
                    n12 = int(tmp[1])
                    n21 = int(tmp[2])
                    n22 = int(tmp[3])
                    V1 = n11 * b1 + n12 * b2
                    V2 = n21 * b1 + n22 * b2
                    cell = {"V1": V1, "V1_abs": np.linalg.norm(V1), "n11": n11, "n12": n12, "n21": n21, "n22": n22,
                            "V2": V2, "V2_abs": np.linalg.norm(V2), "beta": acute_angle_v(V1, V2)}
                    cells.append(cell)
                return True, cells
        except:
            return False, None

    def read_supercells_from_csv(self, filepath):
        df = pd.read_csv(filepath)
        mg_supercells = []
        for i in range(df.shape[0]):
            config = [df.iloc[i]["k11"], df.iloc[i]["k12"], df.iloc[i]["k21"], df.iloc[i]["k22"],
                      df.iloc[i]["l11"], df.iloc[i]["l12"], df.iloc[i]["l21"], df.iloc[i]["l22"], df.iloc[i]["eps_oth"]]
            mg_supercells.append(self.create_supercell(config))

        self.mg_supercells = mg_supercells
        return mg_supercells

    def write_supercells_in_csv(self, mg_supercells=False, directory=Path('./'), id=0):
        directory = Path(directory)
        if type(mg_supercells) is bool:
            mg_supercells = self.mg_supercells

        with open(directory / Path(f'{self.title_me}_{self.lat_me}_{id}.csv'), 'w') as file:
            file.write('alpha,k11,k12,k21,k22,l11,l12,l21,l22,eps_oth,eps1,eps2,L1,L2,S,beta\n')
            for i in range(len(mg_supercells)):
                alpha = mg_supercells[i]["alpha"]
                k11 = mg_supercells[i]["gr"]["n11"]
                k12 = mg_supercells[i]["gr"]["n12"]
                k21 = mg_supercells[i]["gr"]["n21"]
                k22 = mg_supercells[i]["gr"]["n22"]
                l11 = mg_supercells[i]["me"]["n11"]
                l12 = mg_supercells[i]["me"]["n12"]
                l21 = mg_supercells[i]["me"]["n21"]
                l22 = mg_supercells[i]["me"]["n22"]
                eps1 = mg_supercells[i]["eps1"]
                eps2 = mg_supercells[i]["eps2"]
                L1 = mg_supercells[i]["me"]["V1_abs"]
                L2 = mg_supercells[i]["me"]["V2_abs"]
                S = mg_supercells[i]["S"]
                beta = mg_supercells[i]["me"]["beta"]
                eps_oth = mg_supercells[i]["eps_oth"]
                numbers = [alpha, k11, k12, k21, k22, l11, l12, l21, l22, int(eps_oth), eps1, eps2, L1, L2, S, beta]
                file.write(format_write(numbers) + '\n')
            file.write('\n')
            file.write('\n')

    def search_cell(self, radius, title, directory=Path("./"), textmode=False, file_save=False, id=0,
                    beta_fix=[60.], eq_abs=True, beta_min=20., beta_max=100.):
        directory = Path(directory)

        if title == "gr":
            a = self.gr_a
            b1 = self.g1
            b2 = self.g2
            title_save = "graphene"
            tmp = int(m.ceil(radius / a / np.cos(np.radians(30))))
        if title == "me":
            b1 = self.m1
            b2 = self.m2
            title_save = "metall"
            tmp = self.compute_m_max(radius)

        n1_max = tmp
        n1_min = -tmp
        n2_max = tmp
        n2_min = -tmp
        if textmode:
            print(f'{self.title_me}_{self.lat_me}_{id}' + f': Start collecting {title}_vectors', end='\r')
        vectors = []
        for n1 in range(n1_min, n1_max):
            for n2 in range(n2_min, n2_max):
                V = n1 * b1 + n2 * b2
                V_abs = np.linalg.norm(V)
                if V_abs <= radius and V_abs != 0:
                    vector = [V, V_abs, n1, n2]
                    vectors.append(vector)
        if textmode:
            print(f'{self.title_me}_{self.lat_me}_{id}' + f': N of {title}_vectors =', len(vectors), '         ')

        N_all = len(vectors) ** 2 / 2
        l = 0
        cells = []
        last_print_time = time.time()
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                l += 1
                current_time = time.time()
                if textmode and current_time - last_print_time >= 0.1:
                    print(f'{self.title_me}_{self.lat_me}_{id}' + f': Group in {title}_cells: ',
                          f'{l / N_all * 100:.8f}', '%', end='\r')
                    last_print_time = current_time
                beta = acute_angle_v(vectors[i][0], vectors[j][0])

                if self.good_cell(vectors[i][1], vectors[j][1], beta, beta_fix, eq_abs, beta_min, beta_max):
                    good = True
                    cell = {"V1": vectors[i][0], "V1_abs": vectors[i][1], "n11": vectors[i][2], "n12": vectors[i][3],
                            "V2": vectors[j][0], "V2_abs": vectors[j][1], "n21": vectors[j][2], "n22": vectors[j][3],
                            "beta": beta}
                    for k in range(len(cells)):
                        if self.eq_cell(cell, cells[k]):
                            good = False
                            break
                    if good:
                        cells.append(cell)
        if textmode:
            print(f'{self.title_me}_{self.lat_me}_{id}' + f': Group in {title}_cells: ', f'{100:.8f}', '%', end='\r')
            print(f'{self.title_me}_{self.lat_me}_{id}' + f': N of {title}_cells =', len(cells), '                 ')

        if file_save:
            try:
                if isinstance(file_save, bool):
                    file_name = f'{self.title_me}_{self.lat_me}_{id}_{title_save}_save.txt'
                else:
                    file_name = file_save
                with open(directory / Path(file_name), 'w', newline="") as f:
                    if title == "me":
                        f.write(f"{self.title_me} {self.lat_me}\n")
                        f.write(f"{self.me_a_3d}\n")
                    else:
                        f.write(f"{a}\n")
                    f.write(f"{len(cells)}\n")
                    for cell in cells:
                        s_tmp = cell["V1_abs"] * cell["V2_abs"] * np.sin(
                            np.radians(acute_angle_v(cell["V1"], cell["V2"])))
                        f.write(
                            f'{cell["n11"]} {cell["n12"]} {cell["n21"]} {cell["n22"]} {s_tmp:.3f} {acute_angle_v(cell["V1"], cell["V2"]):.3f}\n')
            except:
                print(f'Error in save {title} file')
        return cells

    def func_opt_eps(self, eps1, eps2):
        """чем меньше, тем лучше"""
        return min([abs(eps1), abs(eps2)]) * abs(eps1 - eps2)

    def search_supercell(self, radius: float = 20, eps_max: float = 2.5, eps_min: float = 0., id: int = 0,
                         beta_fix: float = [60.], eq_abs: bool = True,
                         textmode: bool = True, beta_min: float = 20., beta_max: float = 100., csv: bool = False,
                         eq_eps: float = 1.e-1,
                         directory_res=Path("./"), graphene_save: bool = False, graphene_from_file: str = "",
                         metal_save: bool = False,
                         metal_from_file: str = ""):

        directory_res = Path(directory_res)
        if eq(beta_min, 0., 1.e-4):
            beta_min = 1.e-2
        if eq(eq_eps, 0., 1.e-6):
            eq_eps = 1.e-5

        if type(beta_fix) is float or type(beta_fix) is int:
            beta_fix = [beta_fix]
        if beta_fix == True:
            beta_fix = [60.]

        if metal_from_file != "":
            metal_from_file, me_cells = self.read_cell_from_file(metal_from_file, "me")
            if textmode:
                if metal_from_file:
                    print(f'{self.title_me}_{self.lat_me}_{id}' + f': N of me_cells =', len(me_cells),
                          '                 ')
                else:
                    print(f'Metal file not found')
        if graphene_from_file != "":
            graphene_from_file, gr_cells = self.read_cell_from_file(graphene_from_file, "gr")
            if textmode:
                if graphene_from_file:
                    print(f'{self.title_me}_{self.lat_me}_{id}' + f': N of gr_cells =', len(gr_cells),
                          '                 ')
                else:
                    print(f'Grahpene file not found')

        if not metal_from_file:
            me_cells = self.search_cell(radius, "me", directory_res, textmode, metal_save, id, beta_fix, eq_abs,
                                        beta_min, beta_max)
        if not graphene_from_file:
            gr_cells = self.search_cell(radius * (1 + eps_max / 100), "gr", directory_res, textmode, graphene_save, id,
                                        beta_fix, eq_abs, beta_min, beta_max)

        N_all = len(me_cells) * len(gr_cells)
        l = 0
        last_print_time = time.time()
        mg_supercells = []
        for me_cell in me_cells:
            for gr_cell in gr_cells:
                if eq(me_cell["beta"], gr_cell["beta"], 1.e-5):
                    eps1 = (me_cell["V1_abs"] - gr_cell["V1_abs"]) / gr_cell["V1_abs"] * 100
                    eps2 = (me_cell["V2_abs"] - gr_cell["V2_abs"]) / gr_cell["V2_abs"] * 100
                    eps1_oth = (me_cell["V1_abs"] - gr_cell["V2_abs"]) / gr_cell["V2_abs"] * 100
                    eps2_oth = (me_cell["V2_abs"] - gr_cell["V1_abs"]) / gr_cell["V1_abs"] * 100

                    direct = False
                    other = False
                    if eq(eps1, eps2, eq_eps) and eps_min < abs(eps1) < eps_max and eps_min < abs(eps2) < eps_max:
                        direct = True
                    if eq(eps1_oth, eps2_oth, eq_eps) and eps_min < abs(eps1_oth) < eps_max and eps_min < abs(
                            eps2_oth) < eps_max:
                        other = True

                    if direct:
                        if other:
                            if self.func_opt_eps(eps1, eps2) < self.func_opt_eps(eps1_oth, eps2_oth):
                                mg_supercell = {"me": me_cell, "gr": gr_cell, "eps1": eps1, "eps2": eps2,
                                                "eps_oth": False}
                                mg_supercells.append(mg_supercell)
                            else:
                                mg_supercell = {"me": me_cell, "gr": gr_cell, "eps1": eps1_oth, "eps2": eps2_oth,
                                                "eps_oth": True}
                                mg_supercells.append(mg_supercell)
                        else:
                            mg_supercell = {"me": me_cell, "gr": gr_cell, "eps1": eps1, "eps2": eps2, "eps_oth": False}
                            mg_supercells.append(mg_supercell)
                    else:
                        if other:
                            mg_supercell = {"me": me_cell, "gr": gr_cell, "eps1": eps1_oth, "eps2": eps2_oth,
                                            "eps_oth": True}
                            mg_supercells.append(mg_supercell)

                l += 1
                current_time = time.time()
                if textmode and current_time - last_print_time >= 0.1:
                    print(f'{self.title_me}_{self.lat_me}_{id}' + f': Group in mg_supercells: ',
                          f'{l / N_all * 100:.8f}', '%', end='\r')
                    last_print_time = current_time

        if textmode:
            print(f'{self.title_me}_{self.lat_me}_{id}' + f': Group in mg_supercells: ', f'{100:.8f}', '%', end='\r')
            print(f'{self.title_me}_{self.lat_me}_{id}' + f': N of mg_supercells =', len(mg_supercells),
                  '                        ')

        for mg_supercell in mg_supercells:
            mg_supercell["alpha"] = self.compute_alpha(mg_supercell["gr"], mg_supercell["me"],
                                                       [self.g1, self.g2], [self.m1, self.m2])
            mg_supercell["S"] = mg_supercell["me"]["V1_abs"] * mg_supercell["me"]["V2_abs"] * np.sin(
                np.radians(mg_supercell["me"]["beta"]))

        mg_groups = group_el(mg_supercells, self.eq_mg_supercell)

        mg_supercells = []
        for group in mg_groups:
            keyed_group = np.array([group[i]["S"] for i in range(len(group))])
            min_index = np.argmin(keyed_group)
            mg_supercells.append(group[min_index])
        if textmode:
            print(f'{self.title_me}_{self.lat_me}_{id}' + ': N of unique mg_supercells =', len(mg_supercells),
                  '        ')

        mg_supercells.sort(key=lambda x: x["S"])
        print(f'{self.title_me}_{self.lat_me}_{id}' + ': Job done\n')

        self.mg_supercells = mg_supercells
        if csv:
            self.write_supercells_in_csv(directory=directory_res, id=id)
        return mg_supercells

    def build_supercell(self, config=False, mg_supercell=False, directory_res=Path('./'), n_me_layers=3, id: int = 0,
                        textmode=False,
                        supercell_save=True, save_nodef_graphene=False):
        directory_res = Path(directory_res)
        if type(mg_supercell) is bool:
            mg_supercell = self.create_supercell(config)

        angle_grme = mg_supercell["alpha"]
        M1, M2 = mg_supercell["me"]["V1"], mg_supercell["me"]["V2"]
        l11, l12, l21, l22 = int(mg_supercell["me"]["n11"]), int(mg_supercell["me"]["n12"]), int(
            mg_supercell["me"]["n21"]), int(mg_supercell["me"]["n22"])
        if not mg_supercell["eps_oth"]:
            G1, G2 = mg_supercell["gr"]["V1"], mg_supercell["gr"]["V2"]
            k11, k12, k21, k22 = int(mg_supercell["gr"]["n11"]), int(mg_supercell["gr"]["n12"]), int(
                mg_supercell["gr"]["n21"]), int(mg_supercell["gr"]["n22"])
        else:
            G1, G2 = mg_supercell["gr"]["V2"], mg_supercell["gr"]["V1"]
            k11, k12, k21, k22 = int(mg_supercell["gr"]["n21"]), int(mg_supercell["gr"]["n22"]), int(
                mg_supercell["gr"]["n11"]), int(mg_supercell["gr"]["n12"])
        G1_abs = np.linalg.norm(G1)
        G2_abs = np.linalg.norm(G2)
        M1_abs = np.linalg.norm(M1)
        M2_abs = np.linalg.norm(M2)
        eps1 = (M1_abs - G1_abs) / G1_abs  # G_new = (1+eps)G
        eps2 = (M2_abs - G2_abs) / G2_abs
        beta = acute_angle_v(M1, M2)

        #####  Graphene

        gr_surf_all = []
        min_g1 = min([0, k11, k21, k11 + k21]) - 1
        max_g1 = max([0, k11, k21, k11 + k21]) + 1
        min_g2 = min([0, k12, k22, k12 + k22]) - 1
        max_g2 = max([0, k12, k22, k12 + k22]) + 1
        g_second = np.array([self.g1[0] / 2, self.g1[0] * 3 ** 0.5 / 6])
        for i in range(min_g1, max_g1):
            for j in range(min_g2, max_g2):
                G = i * self.g1 + j * self.g2
                gr_surf_all.append(G)
                G = G + g_second
                gr_surf_all.append(G)

        gr_surface = []
        for atom in gr_surf_all:
            if self.atom_not_inside(atom, G1, G2):
                continue
            if eq(atom[0], G1[0], 1.e-5) and eq(atom[1], G1[1], 1.e-5):
                continue
            if eq(atom[0], G2[0], 1.e-5) and eq(atom[1], G2[1], 1.e-5):
                continue
            if self.on_bounary(atom, G1, G2):
                continue
            gr_surface.append([atom[0], atom[1], 0])

        if textmode:
            print(f'{self.title_me}{id}: Graphen: ', len(gr_surface))

        if save_nodef_graphene:
            swap = False
            if single_angle_v(G1, G2) == -1:
                swap = True
                tmp = G2
                G2 = G1
                G1 = tmp
            with open(directory_res / f'Gr{self.title_me}{id}_{angle_grme:.3f}_{beta:.3f}_nodef_graphene.xyz', 'w',
                      newline='') as file:
                file.write(f'{len(gr_surface)}')
                file.write('\n\n')
                for atom in gr_surface:
                    file.write(f'C {atom[0]:.16f} {atom[1]:.15f} {(self.z_dist / 2):.15f}\n')
                file.write('\n')
            with open(directory_res / f'Gr{self.title_me}{id}_{angle_grme:.3f}_{beta:.3f}_nodef_graphene.txt', 'w',
                      newline='') as file:
                file.write(r'CELL_PARAMETERS {angstrom}' + '\n')
                file.write(f'{G1[0]:.16f} {G1[1]:.16f} {0.:.16f}\n')
                file.write(f'{G2[0]:.16f} {G2[1]:.16f} {0.:.16f}\n')
                file.write(f'{0.:.16f} {0.:.16f} {self.z_dist:.16f}\n')
            if swap:
                tmp = G2
                G2 = G1
                G1 = tmp

        # Defortmation
        S = np.zeros((2, 2))
        S[0] = G1
        S[1] = G2
        S = np.transpose(S)
        gr_surface = linear_map(gr_surface, np.linalg.inv(S))
        Eps = np.zeros((2, 2))
        Eps[0][0] = 1 + eps1
        Eps[1][1] = 1 + eps2
        gr_surface = linear_map(gr_surface, Eps)
        gr_surface = linear_map(gr_surface, S)
        G1 *= 1 + eps1
        G2 *= 1 + eps2
        G1_abs = np.linalg.norm(G1)
        G2_abs = np.linalg.norm(G2)

        ####  Metal

        min_m1 = min([0, l11, l21, l11 + l21])
        max_m1 = max([0, l11, l21, l11 + l21])
        min_m2 = min([0, l12, l22, l12 + l22])
        max_m2 = max([0, l12, l22, l12 + l22])
        l1_me = max_m1 - min_m1
        l2_me = max_m2 - min_m2
        size1 = l1_me + 6
        size2 = l2_me + 6

        cell_me_all = self.build_me(size1, size2, n_me_layers)
        t = cell_me_all[0][2]
        for atom in cell_me_all:
            atom[2] -= t
            atom[2] += 1

        Med = (size2 // 2 - 1) * size1 + size1 // 2 - 1
        trans = -M1 / 2 - M2 / 2
        coord = [cell_me_all[Med][0] + trans[0], cell_me_all[Med][1] + trans[1]]

        cell_me_tmp = cell_me_all[0:size1 * size2]
        eps_nach = []
        for i in range(size1 * size2):
            a = np.array([cell_me_tmp[i][0] - coord[0], cell_me_tmp[i][1] - coord[1]])
            eps_nach.append(np.linalg.norm(a))
        j = np.argmin(eps_nach)
        coord_nach = [cell_me_tmp[j][0], cell_me_tmp[j][1]]
        for atom in cell_me_all:
            atom[0] -= coord_nach[0]
            atom[1] -= coord_nach[1]

        cell_me = []
        for atom in cell_me_all:
            if self.atom_not_inside(atom[:-1], M1, M2):
                continue
            if eq(atom[0], M1[0], 1.e-5) and eq(atom[1], M1[1], 1.e-5):
                continue
            if eq(atom[0], M2[0], 1.e-5) and eq(atom[1], M2[1], 1.e-5):
                continue
            if self.on_bounary(atom[:-1], M1, M2):
                continue
            cell_me.append([atom[0], atom[1], atom[2]])

        if textmode:
            print(f'{self.title_me}{id}: Me ready:', len(cell_me))

        # Recalculate med_gr
        recalculate = False
        beta_gr = angle_v(G1, G2)
        beta_me = angle_v(M1, M2)

        if not eq(beta_gr, beta_me, 1.e-5):
            for atom in gr_surface:
                atom[0] -= G1[0]
                atom[1] -= G1[1]
            G1 = -G1
            recalculate = True

        med_me = M1 + M2
        med_me /= np.linalg.norm(med_me)
        med_gr = G1 + G2
        med_gr /= np.linalg.norm(med_gr)

        # Reflex
        reflex = False
        if single_angle_v(M1, M2) != single_angle_v(G1, G2) and not eq(M1_abs, M2_abs, 1.e-5):
            i = np.array([-med_gr[1], med_gr[0]])
            H = np.eye(2) - 2 * np.outer(i, i)
            gr_surface = linear_map(gr_surface, H)
            reflex = True

        # Rotate Gr to Me
        alpha = angle_v(med_me, med_gr)
        single = single_angle_v(med_me, med_gr)
        R = rotate_matrix(alpha)
        if single == 1:
            R = np.transpose(R)

        G1 = np.dot(R, G1)
        G2 = np.dot(R, G2)
        gr_surface = linear_map(gr_surface, R)

        # Build supercell
        cell = np.zeros((3, 3))
        for i in range(2):
            cell[0][i] = M1[i]
        for i in range(2):
            cell[1][i] = M2[i]

        cell[2] = np.array([0, 0, self.z_dist])

        supercell = []
        max1 = 0
        for atom in cell_me:
            supercell.append([self.title_me, atom[0], atom[1], atom[2]])
            if atom[2] > max1:
                max1 = atom[2]

        for atom in gr_surface:
            atom[2] += (max1 + self.distance)
            supercell.append(['C', atom[0], atom[1], atom[2]])

        cell0 = np.array([cell[0][0], cell[0][1]])
        cell1 = np.array([cell[1][0], cell[1][1]])

        zero = [1, 0]
        angle = angle_v(cell0, zero)
        single = single_angle_v(cell0, zero)
        R = rotate_matrix(angle)
        if single == 1:
            new0 = np.dot(R, cell0)
            new1 = np.dot(R, cell1)
        else:
            new0 = np.dot(np.transpose(R), cell0)
            new1 = np.dot(np.transpose(R), cell1)

        if single_angle_v(zero, new1) == 1:
            for i in range(2):
                cell[0][i] = new0[i]
                cell[1][i] = new1[i]

            for atom in supercell:
                vector = np.array([atom[1], atom[2]])
                if single == 1:
                    vector = np.dot(R, vector)
                else:
                    vector = np.dot(np.transpose(R), vector)
                atom[1] = vector[0]
                atom[2] = vector[1]
        else:
            angle = angle_v(cell1, zero)
            single = single_angle_v(cell1, zero)
            R = rotate_matrix(angle)
            if single == 1:
                new0 = np.dot(R, cell0)
                new1 = np.dot(R, cell1)
            else:
                new0 = np.dot(np.transpose(R), cell0)
                new1 = np.dot(np.transpose(R), cell1)

            for i in range(2):
                cell[0][i] = new0[i]
                cell[1][i] = new1[i]

            for atom in supercell:
                vector = np.array([atom[1], atom[2]])
                if single == 1:
                    vector = np.dot(R, vector)
                else:
                    vector = np.dot(np.transpose(R), vector)
                atom[1] = vector[0]
                atom[2] = vector[1]

        if abs(cell[1][1]) < 1.e-12:
            for i in range(2):
                tmp = cell[0][i]
                cell[0][i] = cell[1][i]
                cell[1][i] = tmp

        cell[0][1] = 0

        v1 = np.array([cell[0][0], cell[0][1]])
        v2 = np.array([cell[1][0], cell[1][1]])
        S = np.linalg.norm(v1) * np.linalg.norm(v2) * np.sin(np.radians(angle_v(v1, v2)))

        if textmode:
            print(f'{self.title_me}{id}: Job done')
        try:
            os.makedirs(directory_res)
        except:
            pass

        code_str = 'none'
        if reflex and recalculate:
            code_str = 'reflex and recalculate'
        if reflex:
            code_str = 'reflex'
        if recalculate:
            code_str = 'recalculate'

        if supercell_save:
            with open(directory_res / f'Gr{self.title_me}{id}_{angle_grme:.3f}_{beta:.3f}.xyz', 'w',
                      newline='') as file:
                file.write(f'{len(supercell)}' + '\n')
                file.write('\n')
                for atom in supercell:
                    file.write(atom[
                                   0] + '  ' + f'{round(atom[1], 16):3.16f}  ' + f'{round(atom[2], 16):3.16f}  ' + f'{round(atom[3], 16):3.16f}' + '\n')

            with open(directory_res / f'Gr{self.title_me}{id}_{angle_grme:.3f}_{beta:.3f}.txt', 'w',
                      newline='') as file:
                file.write(r'CELL_PARAMETERS {angstrom}' + '\n')
                for i in range(3):
                    for j in range(3):
                        file.write(f'{round(cell[i][j], 16):3.16f}  ')
                    file.write('\n')
                file.write('\n')
                file.write(f"alpha = {angle_grme:.3f}, " + f"beta = {beta:.3f}; S = {S:.3f}\n")
                file.write(f'Gr: {k11} {k12} {k21} {k22}; ' + str(self.gr_a) + '\n')
                file.write(
                    f'{self.title_me}: {l11} {l12} {l21} {l22}; ' + str(self.me_a1) + ' ' + str(self.me_a2) + '\n')
                file.write(f'eps1 = {eps1 * 100:.3f}, eps2 = {eps2 * 100:.3f} \n \n')
                file.write(f'Number of atoms: {len(supercell)}' + '\n' + '\n')
                file.write(f'{code_str}' + '\n')

        # return supercell

    def build_supercells_from_csv(self, filepath, directory_res=Path('./'), n_me_layers=3, id_start: int = 0,
                                  textmode=False,
                                  supercell_save=True, save_nodef_graphene=False):
        directory_res = Path(directory_res)
        df = pd.read_csv(filepath)
        for i in range(df.shape[0]):
            config = [df.iloc[i]["k11"], df.iloc[i]["k12"], df.iloc[i]["k21"], df.iloc[i]["k22"],
                      df.iloc[i]["l11"], df.iloc[i]["l12"], df.iloc[i]["l21"], df.iloc[i]["l22"], df.iloc[i]["eps_oth"]]
            self.build_supercell(config=config, directory_res=directory_res, n_me_layers=n_me_layers,
                                 id=id_start + i, textmode=textmode, supercell_save=supercell_save,
                                 save_nodef_graphene=save_nodef_graphene)

    def build_supercells_list(self, mg_supercells=False, directory_res=Path('./'), n_me_layers=3, id_start: int = 0,
                              textmode=False,
                              supercell_save=True, save_nodef_graphene=False):
        directory_res = Path(directory_res)
        if type(mg_supercells) is bool:
            mg_supercells = self.mg_supercells
        for i in range(len(mg_supercells)):
            self.build_supercell(mg_supercell=mg_supercells[i], directory_res=directory_res, n_me_layers=n_me_layers,
                                 id=id_start + i, textmode=textmode, supercell_save=supercell_save,
                                 save_nodef_graphene=save_nodef_graphene)

    def compute_mismatch_in_basis(self, mg_supercell, basis=[[1., 0.], [0., 1.]]):
        # for default basis return matrix, where [0][0] - deformation orthogonal C-C, [1][1] - parallel C-C
        S = np.zeros((2, 2))
        S[0] = mg_supercell["gr"]["V1"]
        S[1] = mg_supercell["gr"]["V2"]
        S = np.transpose(S)
        Eps = np.zeros((2, 2))
        Eps[0][0] = 1 + mg_supercell["eps1"] / 100
        Eps[1][1] = 1 + mg_supercell["eps2"] / 100

        B = np.zeros((2, 2))
        B[0] = basis[0]
        B[1] = basis[1]
        B = np.transpose(B)

        return np.linalg.multi_dot([np.linalg.inv(B), S, Eps, np.linalg.inv(S), B])

    def compute_deform_of_cc(self, mg_supercell=False, config=False):
        if type(mg_supercell) is bool:
            mg_supercell = self.create_supercell(config)
        M1, M2 = mg_supercell["me"]["V1"], mg_supercell["me"]["V2"]
        if not mg_supercell["eps_oth"]:
            G1, G2 = mg_supercell["gr"]["V1"], mg_supercell["gr"]["V2"]
        else:
            G1, G2 = mg_supercell["gr"]["V2"], mg_supercell["gr"]["V1"]

        G1_abs = np.linalg.norm(G1)
        G2_abs = np.linalg.norm(G2)
        M1_abs = np.linalg.norm(M1)
        M2_abs = np.linalg.norm(M2)
        eps1 = (M1_abs - G1_abs) / G1_abs  # G_new = (1+eps)G
        eps2 = (M2_abs - G2_abs) / G2_abs

        gr_surface = []
        gr_surface.append(self.g2)
        gr_surface.append(np.array([self.g1[0] / 2, self.g1[0] * 3 ** 0.5 / 6]))
        gr_surface.append(self.g1)
        gr_surface.append(self.g1 + np.array([self.g1[0] / 2, self.g1[0] * 3 ** 0.5 / 6]))

        S = np.zeros((2, 2))
        S[0] = G1
        S[1] = G2
        S = np.transpose(S)
        gr_surface = linear_map(gr_surface, np.linalg.inv(S))
        Eps = np.zeros((2, 2))
        Eps[0][0] = 1 + eps1
        Eps[1][1] = 1 + eps2
        gr_surface = linear_map(gr_surface, Eps)
        gr_surface = linear_map(gr_surface, S)

        cc_arr = []
        for i in range(1, len(gr_surface)):
            cc_arr.append(np.linalg.norm(gr_surface[i] - gr_surface[i - 1]))

        cc_arr.sort()
        if abs(cc_arr[0] - cc_arr[1]) < abs(cc_arr[1] - cc_arr[2]):
            cc = cc_arr[2]
            cc_oth = (cc_arr[0] + cc_arr[1]) / 2
        else:
            cc = cc_arr[0]
            cc_oth = (cc_arr[2] + cc_arr[1]) / 2
        eps_cc = (cc - self.cc) / self.cc
        dzeta = self.cc / (2 * cc_oth)
        eps_orth_cc = ((1 - dzeta * dzeta) ** 0.5 / (3) ** 0.5 / dzeta) - 1

        return eps_cc * 100, eps_orth_cc * 100