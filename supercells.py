import os
import time
from pathlib import Path
import math as m
import numpy as np
import pandas as pd
from ase import Atoms

from ase.build import fcc100
from ase.build import fcc110
from ase.build import fcc111
from ase.build import bcc100
from ase.build import bcc110
from ase.build import bcc111
from ase.build import hcp0001
from ase.build import diamond100
from ase.build import diamond111


def group_el(array:list, elements_eq) -> list:
    """
    Groups equivalent elements of a list according to an equivalence function.

    Args:
        array (list): list to be grouped;
        elements_eq: equivalence function f(el_1, el_2) that takes 2 elements and returns
        'True' if the elements are equivalent.

    Returns:
        list: list containing sublists of equivalent elements.
    """
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

def format_write(numbers:list) -> str:
    """
    Creates a string of numbers spaced with equal intervals.

    Args:
        numbers: list of numbers to format.

    Returns:
        str: string of numbers.
    """
    string = ""
    num = len(numbers)
    for i in range(num):
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
        if i == num - 1:
            string += formatted_num
        else:
            string += formatted_num + "," + " " * num_spaces

    return string

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the angle between vectors in degrees.

    Args:
        v1: First vector as numpy array.
        v2: Second vector as numpy array.

    Returns:
        Angle between vectors in degrees [0, 180].
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Vectors must have non-zero magnitude")

    cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine = np.clip(cosine, -1.0, 1.0) #for numerical stability

    return np.degrees(np.arccos(cosine))

def acute_angle_v(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the acute angle between vectors in degrees.

    Args:
        v1: First vector as numpy array.
        v2: Second vector as numpy array.

    Returns:
        Acute angle between vectors in degrees [0, 90].
    """
    angle = angle_between_vectors(v1, v2)
    return min(angle, 180 - angle)

def vector_triple_orientation(v1: np.ndarray, v2: np.ndarray) -> int:
    """
    Determines whether vectors v1, v2 and a vector perpendicular to them form a right or left triple.

    Args:
        v1 (np.ndarray): first vector;
        v2 (np.ndarray): second vector.

    Returns:
        int: if '+1' - right triple, if '-1' - left triple.
    """
    v11 = np.array([v1[0], v1[1], 0])
    v12 = np.array([v2[0], v2[1], 0])
    cross_product = np.cross(v11, v12)
    return 1 if cross_product[2] > 0 else -1

def rotate_matrix(angle_degrees: float) -> np.ndarray:
    """
    Creates a 2D rotation matrix for the given angle.

    Args:
        angle_degrees: Rotation angle in degrees.

    Returns:
        2x2 rotation matrix.
    """
    angle_rad = np.radians(angle_degrees)
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)

    return np.array([[cos_val, -sin_val],
                     [sin_val, cos_val]])

def linear_map(system: list, map_matrix: np.ndarray) -> list:
    """
    Performs a two-dimensional linear transformation on a system of atoms.

    Args:
        system (list): list of atoms, where an atom is a list of coordinates [x, y];
        map_matrix (np.ndarray): linear transformation matrix.

    Returns:
        New list of transformed coordinates.
    """
    transformed = np.stack(system) @ map_matrix.T
    return [vec for vec in transformed]

def are_equal(a: float, b: float, tolerance: float = 1e-5) -> bool:
    """
    Checks equality of numbers with a given tolerance.

    Args:
        a: first number;
        b: second number;
        eps: required equality tolerance.

    Returns:
        True if |a - b| <= tolerance, False otherwise.
    """
    return abs(a - b) <= tolerance

def fractional_remainder(x: float, y:float) -> float:
    """
    Calculates the remainder of dividing a larger number by a smaller one.

    Args:
        x: First number.
        y: Second number.

    Returns:
        Fractional remainder in range [0, 1).

    Raises:
        ValueError: If both numbers are zero.
    """
    if x == 0 and y == 0:
        raise ValueError("At least one number must be non-zero")

    larger = max(abs(x), abs(y))
    smaller = min(abs(x), abs(y))

    if smaller == 0:
        return 0

    return (larger / smaller) % 1

class Supercell:
    """
    Class for finding and building supercells.

    Attributes:
        gs_supercells: list of supercells of type dict, where:
            'eps1' (float): mismatch parameter (in %) for the first supercell vector;
            'eps2' (float): mismatch parameter (in %) for the second supercell vector;
            'alpha' (float): rotation angle (in degrees) of the graphene lattice relative to the substrate lattice;
            'S' (float): supercell area (in angstroms);
            'sub', 'gr' (dict): substrate and graphene cells, where:
                'V1', 'V2' (np.ndarray): first and second cell vectors;
                'V1_abs', 'V2_abs' (float): lengths of the first and second cell vectors;
                'n11', 'n12': coordinates of the first cell vector in the lattice basis;
                'n21', 'n22': coordinates of the second cell vector in the lattice basis;
                'beta': angle between cell vectors
        other attributes see in __init__()
    """

    def __init__(self, title_sub: str = "", lat_sub: str = "", z_dist: float = 25., distance: float = 3):
        """
        Args:
            title_sub (str): substrate name;
            lat_sub (str): substrate surface name, implemented:
                * fcc111
                * fcc110
                * fcc100
                * bcc111
                * bcc110
                * bcc100
                * hcp0001
                * diamond100
                * diamond111
            z_dist (float): size of the generated cell along the Z axis in angstroms;
            distance (float): distance between the graphene plane and the substrate in angstroms.
        """
        self.z_dist = z_dist
        self.distance = distance
        self.gr_a_exp = 2.4612
        self.gr_a = self.gr_a_exp
        self._compute_gr_a()
        self.sub_a_3d_exp = 1.0
        self.sub_c_exp = 1.0
        if title_sub != "":
            self.title_sub = title_sub
            self._set_sub_a_3d_exp()
        self.sub_a_3d = self.sub_a_3d_exp
        self.sub_c = self.sub_c_exp
        self.lat_sub = lat_sub
        self._compute_sub_a()
        self.gs_supercells = []

    def _set_sub_a_3d_exp(self):
        """
        Set experimental value of lattice constant
        """
        match self.title_sub:
            case "Ag":
                self.sub_a_3d_exp = 4.086
            case "Au":
                self.sub_a_3d_exp = 4.0782
            case 'Pt':
                self.sub_a_3d_exp = 3.9236
            case 'Al':
                self.sub_a_3d_exp = 4.0496
            case 'Ni':
                self.sub_a_3d_exp = 3.524
            case 'Cu':
                self.sub_a_3d_exp = 3.6146
            case 'Ir':
                self.sub_a_3d_exp = 3.8392
            case 'Pd':
                self.sub_a_3d_exp = 3.8903
            case 'Fe':
                self.sub_a_3d_exp = 2.8665
            case 'Co':
                self.sub_a_3d_exp = 2.5071
                self.sub_c_exp = 4.0686

    def _compute_sub_a(self):
        """
        Calculates substrate lattice vectors and their lengths depending on the set surface type
        and lattice parameter.
        """
        if self.lat_sub == "":
            self.sub_a1 = 1.0
            self.sub_a2 = self.sub_a1
            self.s1 = np.array([self.sub_a1, 0.0])
            self.s2 = np.array([0.0, self.sub_a2])
        if self.lat_sub == "fcc100":
            self.sub_a1 = self.sub_a_3d * 2 ** 0.5 / 2
            self.sub_a2 = self.sub_a1
            self.s1 = np.array([self.sub_a1, 0.0])
            self.s2 = np.array([0.0, self.sub_a2])
        if self.lat_sub == "fcc110":
            self.sub_a1 = self.sub_a_3d
            self.sub_a2 = self.sub_a_3d * 2 ** 0.5 / 2
            self.s1 = np.array([self.sub_a1, 0.0])
            self.s2 = np.array([0.0, self.sub_a2])
        if self.lat_sub == "fcc111":
            self.sub_a1 = self.sub_a_3d * 2 ** 0.5 / 2
            self.sub_a2 = self.sub_a1
            self.s1 = np.array([self.sub_a1, 0.0])
            self.s2 = np.array([self.sub_a2 * np.cos(np.radians(60.0)), self.sub_a2 * np.sin(np.radians(60.0))])
        if self.lat_sub == "bcc100":
            self.sub_a1 = self.sub_a_3d
            self.sub_a2 = self.sub_a1
            self.s1 = np.array([self.sub_a1, 0.0])
            self.s2 = np.array([0.0, self.sub_a2])
        if self.lat_sub == "bcc110":
            angle_tmp = np.arccos(1 / 3 ** 0.5)
            self.sub_a1 = self.sub_a_3d
            self.sub_a2 = self.sub_a_3d * 3 ** 0.5 / 2
            self.s1 = np.array([self.sub_a1, 0.0])
            self.s2 = np.array([self.sub_a2 * np.cos(angle_tmp), self.sub_a2 * np.sin(angle_tmp)])
        if self.lat_sub == "bcc111":
            self.sub_a1 = self.sub_a_3d * 2 ** 0.5
            self.sub_a2 = self.sub_a1
            self.s1 = np.array([self.sub_a1, 0.0])
            self.s2 = np.array([self.sub_a2 * np.cos(np.radians(60.0)), self.sub_a2 * np.sin(np.radians(60.0))])
        if self.lat_sub == "hcp0001":
            self.sub_a1 = self.sub_a_3d
            self.sub_a2 = self.sub_a1
            self.s1 = np.array([self.sub_a1, 0.0])
            self.s2 = np.array([self.sub_a2 * np.cos(np.radians(60.0)), self.sub_a2 * np.sin(np.radians(60.0))])
        if self.lat_sub == "diamond100":
            self.sub_a1 = self.sub_a_3d * 2 ** 0.5 / 2
            self.sub_a2 = self.sub_a1
            self.s1 = np.array([self.sub_a1, 0.0])
            self.s2 = np.array([0.0, self.sub_a2])
        if self.lat_sub == "diamond111":
            self.sub_a1 = self.sub_a_3d * 2 ** 0.5 / 2
            self.sub_a2 = self.sub_a1
            self.s1 = np.array([self.sub_a1, 0.0])
            self.s2 = np.array([self.sub_a2 * np.cos(np.radians(60.0)), self.sub_a2 * np.sin(np.radians(60.0))])

    def _compute_s_max(self, radius: float) -> int:
        """
        Calculates the maximum coordinate value in the substrate lattice basis for searching within a given radius.
        Args:
            radius (float): required radius.
        """
        if self.lat_sub == "fcc100":
            return int(m.ceil(radius / self.sub_a1))
        if self.lat_sub == "fcc110":
            return int(m.ceil(radius / self.sub_a2))
        if self.lat_sub == "fcc111":
            return int(m.ceil(radius / self.sub_a1 / np.sin(np.radians(60.0))))
        if self.lat_sub == "bcc100":
            return int(m.ceil(radius / self.sub_a1))
        if self.lat_sub == "bcc110":
            return int(m.ceil(radius / self.sub_a2 / np.sin(np.arccos(1 / 3 ** 0.5))))
        if self.lat_sub == "bcc111":
            return int(m.ceil(radius / self.sub_a1 / np.sin(np.radians(60.0))))
        if self.lat_sub == "hcp0001":
            return int(m.ceil(radius / self.sub_a1 / np.sin(np.radians(60.0))))
        if self.lat_sub == "diamond100":
            return int(m.ceil(radius / self.sub_a1))
        if self.lat_sub == "diamond111":
            return int(m.ceil(radius / self.sub_a1 / np.sin(np.radians(60.0))))

    def _compute_gr_a(self):
        """
        Calculates graphene lattice vectors and their lengths depending on the graphene lattice parameter,
        also calculates the C-C bond length.
        """
        self.g1 = np.array([self.gr_a, 0.0])
        self.g2 = np.array([self.gr_a * np.cos(np.radians(60.0)), self.gr_a * np.sin(np.radians(60.0))])
        self.cc = np.linalg.norm(np.array([self.gr_a / 2, self.gr_a * 3 ** 0.5 / 6]))

    def set_distance(self, distance: float):
        """
        Sets the required distance between the graphene plane and the substrate.
        Args:
            distance (float): required distance.
        """
        self.distance = distance

    def set_gr(self, a: float):
        """
        Sets the graphene lattice for a given lattice parameter.
        Args:
            a (float): required lattice parameter.
        """
        self.gr_a = a
        self._compute_gr_a()

    def set_sub(self, title_sub: str = "", lat_sub: str = "", a: float = -1.0, c: float = -1.0):
        """
        Sets the substrate lattice with the required surface type and lattice parameter.
        Args:
            title_sub (str): substrate name;
            lat_sub (str): surface type, implemented types are listed in __init__() of this class;
            a (float): required lattice parameter 'a' (volumetric);
            c (float): required lattice parameter 'c' (volumetric) (if necessary).
        """
        if title_sub != "":
            self.title_sub = title_sub
            self._set_sub_a_3d_exp()
            self.sub_a_3d = self.sub_a_3d_exp
            self.sub_c = self.sub_c_exp
        if lat_sub != "":
            self.lat_sub = lat_sub
        if not a < 0:
            self.sub_a_3d = a
        if not c < 0:
            self.sub_c = c
        self._compute_sub_a()

    def _build_substrate(self, size1: int, size2: int, n_sub_layers: int) -> np.ndarray:
        """
        Builds a substrate of a given size.
        Args:
            size1: substrate size along the first axis;
            size2: substrate size along the second axis;
            n_sub_layers: number of substrate layers;
        Returns:
            np.ndarray: list of substrate atom coordinates.
        """
        if self.lat_sub == "fcc100":
            substrate = fcc100(self.title_sub, size=(size1, size2, n_sub_layers), a=self.sub_a_3d, vacuum=0.0)
        if self.lat_sub == "fcc110":
            substrate = fcc110(self.title_sub, size=(size1, size2, n_sub_layers), a=self.sub_a_3d, vacuum=0.0)
        if self.lat_sub == "fcc111":
            substrate = fcc111(self.title_sub, size=(size1, size2, n_sub_layers), a=self.sub_a_3d, vacuum=0.0)
        if self.lat_sub == "bcc100":
            substrate = bcc100(self.title_sub, size=(size1, size2, n_sub_layers), a=self.sub_a_3d, vacuum=0.0)
        if self.lat_sub == "bcc110":
            substrate = bcc110(self.title_sub, size=(size1, size2, n_sub_layers), a=self.sub_a_3d, vacuum=0.0)
        if self.lat_sub == "bcc111":
            substrate = bcc111(self.title_sub, size=(size1, size2, n_sub_layers), a=self.sub_a_3d, vacuum=0.0)
        if self.lat_sub == "hcp0001":
            substrate = hcp0001(self.title_sub, size=(size1, size2, n_sub_layers), a=self.sub_a_3d, c=self.sub_c, vacuum=0.0)
        if self.lat_sub == "diamond100":
            substrate = diamond100(self.title_sub, size=(size1, size2, n_sub_layers), a=self.sub_a_3d, vacuum=0.0)
        if self.lat_sub == "diamond111":
            substrate = diamond111(self.title_sub, size=(size1, size2, n_sub_layers), a=self.sub_a_3d, vacuum=0.0)

        return np.array(substrate.get_positions())

    def _eq_alpha(self, scell1: dict, scell2: dict) -> bool:
        """
        Determines the equivalence of the rotation angle of the graphene lattice relative to the substrate lattice of two supercells.
        Args:
            scell1 (dict): first supercell;
            scell2 (dict): second supercell.
        """
        if are_equal(scell1["alpha"], scell2["alpha"], 1.e-5):
            return True
        if round(abs(scell1["alpha"] - scell2["alpha"]), 5) % 60 <= 1.e-5:
            return True
        if round(abs(scell1["alpha"] + scell2["alpha"]), 5) % 60 <= 1.e-5:
            return True
        return False

    def _eq_eps(self, scell1: dict, scell2: dict) -> bool:
        """
        Determines the equivalence of mismatches of two supercells.
        Args:
            scell1 (dict): first supercell;
            scell2 (dict): second supercell.
        """
        if are_equal(scell1["eps1"], scell2["eps1"], 1.e-5) and are_equal(scell1["eps2"], scell2["eps2"], 1.e-5):
            return True
        if are_equal(scell1["eps1"], scell2["eps2"], 1.e-5) and are_equal(scell1["eps2"], scell2["eps1"], 1.e-5):
            return True
        return False

    def _proportional(self, scell1: dict, scell2: dict) -> bool:
        """
        Determines the proportionality of two supercells.
        Args:
            scell1 (dict): first supercell;
            scell2 (dict): second supercell.
        """
        divide1 = fractional_remainder(scell1["sub"]["V1_abs"], scell2["sub"]["V1_abs"])
        divide2 = fractional_remainder(scell1["sub"]["V2_abs"], scell2["sub"]["V2_abs"])
        divide3 = fractional_remainder(scell1["sub"]["V1_abs"], scell2["sub"]["V2_abs"])
        divide4 = fractional_remainder(scell1["sub"]["V2_abs"], scell2["sub"]["V1_abs"])
        return (divide1 <= 1.e-5 and divide2 <= 1.e-5) or (divide3 <= 1.e-5 and divide4 <= 1.e-5)

    def _on_bounary(self, atom, L1, L2) -> bool:
        """
        Determines if an atom is on the boundary of the supercell.
        Args:
            atom (np.ndarray): atom coordinates;
            L1 (np.ndarray): first supercell vector;
            L2 (np.ndarray): second supercell vector.
        """
        tmp_v = atom - L2
        if abs(acute_angle_v(tmp_v, L1)) < 1.e-5:
            return True
        tmp_v = atom - L1
        if abs(acute_angle_v(tmp_v, L2)) < 1.e-5:
            return True

    def _atom_not_inside(self, atom, L1, L2)-> bool:
        """
        Determines if an atom is outside the supercell;
        Args:
            atom (np.ndarray): atom coordinates;
            L1 (np.ndarray): first supercell vector;
            L2 (np.ndarray): second supercell vector.
        """
        eps = 1.e-5
        matrix = np.array([L1, L2]).T
        try:
            uv = np.linalg.solve(matrix, atom)
            u, v = uv
        except np.linalg.LinAlgError:
            return True
        if (-eps <= u <= 1 + eps) and (-eps <= v <= 1 + eps):
            return False
        else:
            return True

    def _good_cell(self, abs1: float, abs2: float, beta: float, beta_fix, eq_abs: bool,
                   beta_min: float, beta_max: float):
        """
        Checks if a cell meets the search conditions.
        Args:
            abs1 (float): length of the first cell vector;
            abs2 (float): length of the second cell vector;
            beta (float): angle between cell vectors;
            beta_fix: search parameter for fixing beta, see more in method search_supercell()
            eq_abs (bool): search parameter for cell rhombicity, see more in method search_supercell()
            beta_min (float): minimum value of beta;
            beta_max (float): maximum value of beta;
        """
        if eq_abs:
            if not are_equal(abs1, abs2, 1.e-5):
                return False
        if type(beta_fix) is bool:
            if beta < beta_min or beta > beta_max:
                return False
        else:
            good_beta = False
            for beta_fix_one in beta_fix:
                if are_equal(beta, beta_fix_one, 1.e-5):
                    good_beta = True
            if not good_beta:
                return False
        return True

    def _eq_cell(self, cell1: dict, cell2: dict) -> bool:
        """
        Determines the equivalence of two cells. Cells are equivalent if the lengths of the cell vectors
        and the angle between the vectors (beta) are equal.
        Args:
             cell1 (dict): first cell;
             cell2 (dict): second cell.
        """
        if are_equal(cell1["beta"], cell2["beta"], 1.e-10):
            if are_equal(cell1["V1_abs"], cell2["V1_abs"], 1.e-10) and are_equal(cell1["V2_abs"], cell2["V2_abs"], 1.e-10):
                return True
            if are_equal(cell1["V1_abs"], cell2["V2_abs"], 1.e-10) and are_equal(cell1["V2_abs"], cell2["V1_abs"], 1.e-10):
                return True
        return False

    def _eq_gs_supercell(self, scell1: dict, scell2: dict):
        """
        Determines the equivalence of two supercells. Supercells are equivalent if the beta angles,
        alpha angles, mismatch parameters are equivalent, and if the cells are proportional or geometrically equal.
        Args:
            scell1 (dict): first supercell;
            scell2 (dict): second supercell;
        """
        if not are_equal(scell1["sub"]["beta"], scell2["sub"]["beta"], 1.e-5):
            return False
        if not self._eq_alpha(scell1, scell2):
            return False
        if not self._eq_eps(scell1, scell2):
            return False
        if not self._proportional(scell1, scell2):
            return False
        return True

    def _compute_alpha(self, cell1: dict, cell2: dict, basis1, basis2) -> float:
        """
        Calculates the rotation angle of the lattice of one cell relative to the lattice of another - the alpha angle.
        Args:
            cell1 (dict): first cell;
            cell2 (dict): second cell;
            basis1 (list): list of two lattice vectors of the first cell;
            basis2 (list): list of two lattice vectors of the second cell;
        """
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
        if not are_equal(cell1["beta"], cell2["beta"], 1.e-5):
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
        if vector_triple_orientation(V11, V12) != vector_triple_orientation(V21, V22) and not are_equal(V21_abs, V22_abs, 1.e-5):
            i = np.array([-med_cell1[1], med_cell1[0]])
            H = np.eye(2) - 2 * np.outer(i, i)
            med_basis1 = np.dot(H, med_basis1)
        angle = angle_between_vectors(med_cell2, med_cell1)
        single = vector_triple_orientation(med_cell2, med_cell1)
        R = rotate_matrix(angle)
        if single == 1:
            R = np.transpose(R)
        med_basis1 = np.dot(R, med_basis1)
        alpha = acute_angle_v(med_basis1, med_basis2)
        return float(alpha)

    def create_supercell(self, config: list) -> dict:
        """
        Creates a supercell from a configuration list.
        Args:
            config: configuration list, see more in method build_supercell()
        Returns:
            dict: supercell, see more in the description of this class
        """
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
        M1 = l11 * self.s1 + l12 * self.s2
        M2 = l21 * self.s1 + l22 * self.s2
        gr_cell = {"V1": G1, "V1_abs": np.linalg.norm(G1), "n11": k11, "n12": k12, "n21": k21, "n22": k22,
                   "V2": G2, "V2_abs": np.linalg.norm(G2), "beta": acute_angle_v(G1, G2)}
        sub_cell = {"V1": M1, "V1_abs": np.linalg.norm(M1), "n11": l11, "n12": l12, "n21": l21, "n22": l22,
                   "V2": M2, "V2_abs": np.linalg.norm(M2), "beta": acute_angle_v(M1, M2)}

        if not eps_oth:
            eps1 = (sub_cell["V1_abs"] - gr_cell["V1_abs"]) / gr_cell["V1_abs"] * 100
            eps2 = (sub_cell["V2_abs"] - gr_cell["V2_abs"]) / gr_cell["V2_abs"] * 100
        else:
            eps1 = (sub_cell["V1_abs"] - gr_cell["V2_abs"]) / gr_cell["V2_abs"] * 100
            eps2 = (sub_cell["V2_abs"] - gr_cell["V1_abs"]) / gr_cell["V1_abs"] * 100

        alpha = self._compute_alpha(gr_cell, sub_cell, [self.g1, self.g2], [self.s1, self.s2])
        S = sub_cell["V1_abs"] * sub_cell["V2_abs"] * np.sin(np.radians(sub_cell["beta"]))

        return {"sub": sub_cell, "gr": gr_cell, "eps1": eps1, "eps2": eps2, "eps_oth": eps_oth, "alpha": alpha, "S": S}

    def read_cell_from_file(self, filepath, title: str) -> list:
        """
        Reads a cell from a file.
        Args:
            filepath (str or Path): path to the file;
            title: cell name - 'gr' or 'sub'.
        Returns:
            list: list of cells of type dict (see more in the description of this class)
        """
        try:
            with open(Path(filepath), 'r') as f:
                if title == "sub":
                    tmp = f.readline().split()
                    self.title_sub = tmp[0]
                    self.lat_sub = tmp[1]
                    self.sub_a_3d = float(f.readline())
                    self._compute_sub_a()
                    b1 = self.s1
                    b2 = self.s2
                elif title == "gr":
                    self.gr_a = float(f.readline())
                    self._compute_gr_a()
                    b1 = self.g1
                    b2 = self.g2
                else:
                    return False, None
                cells = []
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

    def read_supercells_from_csv(self, filepath: str) -> list:
        """
        Reads a list of supercells from a '.csv' file and writes it to the gs_supercells attribute.
        Args:
            filepath (str or Path): path to the file.
        Returns:
            list: list of supercells of type dict (see more in the description of this class)
        """
        df = pd.read_csv(filepath)
        gs_supercells = []
        for i in range(df.shape[0]):
            config = [df.iloc[i]["k11"], df.iloc[i]["k12"], df.iloc[i]["k21"], df.iloc[i]["k22"],
                      df.iloc[i]["l11"], df.iloc[i]["l12"], df.iloc[i]["l21"], df.iloc[i]["l22"], df.iloc[i]["eps_oth"]]
            gs_supercells.append(self.create_supercell(config))

        self.gs_supercells = gs_supercells
        return gs_supercells

    def write_supercells_in_csv(self, gs_supercells:list=False, directory=Path('./'), id:int=0):
        """
        Outputs a list of supercells to a '.csv' file.
        Args:
            gs_supercells: list of supercells, by default - outputs the value of the gs_supercells attribute;
            directory (str or Path): directory where the file will be created;
            id (int): id.
        """
        directory = Path(directory)
        if type(gs_supercells) is bool:
            gs_supercells = self.gs_supercells

        with open(directory / Path(f'{self.title_sub}_{self.lat_sub}_{id}.csv'), 'w') as file:
            file.write('alpha,k11,k12,k21,k22,l11,l12,l21,l22,eps_oth,eps1,eps2,L1,L2,S,beta\n')
            for i in range(len(gs_supercells)):
                alpha = gs_supercells[i]["alpha"]
                k11 = gs_supercells[i]["gr"]["n11"]
                k12 = gs_supercells[i]["gr"]["n12"]
                k21 = gs_supercells[i]["gr"]["n21"]
                k22 = gs_supercells[i]["gr"]["n22"]
                l11 = gs_supercells[i]["sub"]["n11"]
                l12 = gs_supercells[i]["sub"]["n12"]
                l21 = gs_supercells[i]["sub"]["n21"]
                l22 = gs_supercells[i]["sub"]["n22"]
                eps1 = gs_supercells[i]["eps1"]
                eps2 = gs_supercells[i]["eps2"]
                L1 = gs_supercells[i]["sub"]["V1_abs"]
                L2 = gs_supercells[i]["sub"]["V2_abs"]
                S = gs_supercells[i]["S"]
                beta = gs_supercells[i]["sub"]["beta"]
                eps_oth = gs_supercells[i]["eps_oth"]
                numbers = [alpha, k11, k12, k21, k22, l11, l12, l21, l22, int(eps_oth), eps1, eps2, L1, L2, S, beta]
                file.write(format_write(numbers) + '\n')
            file.write('\n')
            file.write('\n')

    def _search_cell(self, radius, title, beta_fix=[60.], eq_abs=True, beta_min=20., beta_max=100.,
                     id=0, textmode=False, file_save=False, directory=Path("./")) -> list:
        """
        Searches for cells for a given radius and search settings.
            radius: given radius;
            title: cell name - 'gr' or 'sub'
            beta_fix: search parameter for fixing beta, see more in method search_supercell()
            eq_abs (bool): search parameter for cell rhombicity, see more in method search_supercell()
            beta_min (float): minimum value of beta;
            beta_max (float): maximum value of beta;
            id: id
            textmode: if True, search information will be displayed on the screen
            file_save: if True, the cell search result will be written to a file
            directory: directory where the search result will be written (if necessary)
        """
        directory = Path(directory)
        if title == "gr":
            a = self.gr_a
            b1 = self.g1
            b2 = self.g2
            title_save = "graphene"
            tmp = int(m.ceil(radius / a / np.cos(np.radians(30))))
        if title == "sub":
            b1 = self.s1
            b2 = self.s2
            title_save = "substrate"
            tmp = self._compute_s_max(radius)

        n1_max = tmp
        n1_min = -tmp
        n2_max = tmp
        n2_min = -tmp
        if textmode:
            print(f'{self.title_sub}_{self.lat_sub}_{id}' + f': Start collecting {title}_vectors', end='\r')
        vectors = []
        for n1 in range(n1_min, n1_max):
            for n2 in range(n2_min, n2_max):
                V = n1 * b1 + n2 * b2
                V_abs = np.linalg.norm(V)
                if V_abs <= radius and V_abs != 0:
                    vector = [V, V_abs, n1, n2]
                    vectors.append(vector)
        if textmode:
            print(f'{self.title_sub}_{self.lat_sub}_{id}' + f': N of {title}_vectors =', len(vectors), '         ')

        N_all = len(vectors) ** 2 / 2
        l = 0
        cells = []
        last_print_time = time.time()
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                l += 1
                current_time = time.time()
                if textmode and current_time - last_print_time >= 0.1:
                    print(f'{self.title_sub}_{self.lat_sub}_{id}' + f': Group in {title}_cells: ',
                          f'{l / N_all * 100:.8f}', '%', end='\r')
                    last_print_time = current_time
                beta = acute_angle_v(vectors[i][0], vectors[j][0])
                if self._good_cell(vectors[i][1], vectors[j][1], beta, beta_fix, eq_abs, beta_min, beta_max):
                    good = True
                    cell = {"V1": vectors[i][0], "V1_abs": vectors[i][1], "n11": vectors[i][2], "n12": vectors[i][3],
                            "V2": vectors[j][0], "V2_abs": vectors[j][1], "n21": vectors[j][2], "n22": vectors[j][3],
                            "beta": beta}
                    for k in range(len(cells)):
                        if self._eq_cell(cell, cells[k]):
                            good = False
                            break
                    if good:
                        cells.append(cell)
        if textmode:
            print(f'{self.title_sub}_{self.lat_sub}_{id}' + f': Group in {title}_cells: ', f'{100:.8f}', '%', end='\r')
            print(f'{self.title_sub}_{self.lat_sub}_{id}' + f': N of {title}_cells =', len(cells), '                 ')

        if file_save:
            try:
                if isinstance(file_save, bool):
                    file_name = f'{self.title_sub}_{self.lat_sub}_{id}_{title_save}_save.txt'
                else:
                    file_name = file_save
                with open(directory / Path(file_name), 'w', newline="") as f:
                    if title == "sub":
                        f.write(f"{self.title_sub} {self.lat_sub}\n")
                        f.write(f"{self.sub_a_3d}\n")
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

    def _func_opt_eps(self, eps1: float, eps2: float) -> float:
        """
        Function whose minimum corresponds to the best supercell in terms of comparing
        the mismatch parameter and anisotropy.
        """
        return min([abs(eps1), abs(eps2)]) * abs(eps1 - eps2)

    def search_supercell(self, radius: float = 20, eps_max: float = 2.5, eps_min: float = 0., id: int = 0,
                         beta_fix = [60.], beta_min: float = 20., beta_max: float = 100., eq_abs: bool = True,
                         eq_eps: float = 1.e-1, textmode: bool = True, csv: bool = False, directory_res=Path("./"),
                         graphene_save: bool = False, graphene_from_file: str = "",
                         substrate_save: bool = False, substrate_from_file: str = "") -> list:
        """
        Searches for supercells for a given radius and search settings,
        writes the result to the gs_supercell attribute.
        Args:
            radius (float): search radius (in angstroms) - supercell vector lengths will not exceed the radius;
            eps_max (float): maximum mismatch parameter (in %)
            eps_min (float): minimum mismatch parameter (in %)
            id (ind): search id
            beta_fix (list or bool): if False, the beta angle (in degrees) is completely unfixed and can vary
            from beta_min to beta_max; if [beta1, beta2, ...], the beta angle can only take values from the list;
            beta_min (float): minimum value of the beta angle (in degrees);
            beta_max (float): maximum value of the beta angle (in degrees);
            eq_abs (bool): if 'True', cells will only be rhombic, if 'False', then not;
            eq_eps (float): maximum anisotropy - difference between eps1 and eps2 (in %);
            textmode (str): if True, search information will be displayed on the screen;
            csv (bool): if True, the search result is output to a .csv file;
            directory_res: directory where the search result file will be written (if necessary);
            graphene_save, substrate_save (bool): if True, the cell search result will be written to a file;
            graphene_from_file, substrate_from_file (str or Path): if not an empty string, the method will read cells from the file.
        Returns:
            list: list of supercells of type dict (see more in the class description).
        """
        directory_res = Path(directory_res)
        if are_equal(beta_min, 0., 1.e-4):
            beta_min = 1.e-2
        if are_equal(eq_eps, 0., 1.e-6):
            eq_eps = 1.e-5
        if type(beta_fix) is float or type(beta_fix) is int:
            beta_fix = [beta_fix]
        if type(beta_fix) is bool and beta_fix == True:
            beta_fix = [60.]

        if substrate_from_file != "":
            substrate_from_file, sub_cells = self.read_cell_from_file(substrate_from_file, "sub")
            if textmode:
                if substrate_from_file:
                    print(f'{self.title_sub}_{self.lat_sub}_{id}' + f': N of sub_cells =', len(sub_cells), '                          ')
                else:
                    print(f'Substrate file not found')
        if graphene_from_file != "":
            graphene_from_file, gr_cells = self.read_cell_from_file(graphene_from_file, "gr")
            if textmode:
                if graphene_from_file:
                    print(f'{self.title_sub}_{self.lat_sub}_{id}' + f': N of gr_cells =', len(gr_cells), '                          ')
                else:
                    print(f'Graphene file not found')
        if not substrate_from_file:
            sub_cells = self._search_cell(radius, "sub", beta_fix, eq_abs, beta_min, beta_max, id,
                                          textmode, substrate_save, directory_res)
        if not graphene_from_file:
            gr_cells = self._search_cell(radius * (1 + eps_max / 100), "gr", beta_fix, eq_abs, beta_min, beta_max, id,
                                         textmode, graphene_save, directory_res)

        N_all = len(sub_cells) * len(gr_cells)
        l = 0
        last_print_time = time.time()
        gs_supercells = []
        for sub_cell in sub_cells:
            for gr_cell in gr_cells:
                if are_equal(sub_cell["beta"], gr_cell["beta"], 1.e-5):
                    eps1 = (sub_cell["V1_abs"] - gr_cell["V1_abs"]) / gr_cell["V1_abs"] * 100
                    eps2 = (sub_cell["V2_abs"] - gr_cell["V2_abs"]) / gr_cell["V2_abs"] * 100
                    eps1_oth = (sub_cell["V1_abs"] - gr_cell["V2_abs"]) / gr_cell["V2_abs"] * 100
                    eps2_oth = (sub_cell["V2_abs"] - gr_cell["V1_abs"]) / gr_cell["V1_abs"] * 100
                    direct = False
                    other = False
                    if are_equal(eps1, eps2, eq_eps) and eps_min < abs(eps1) < eps_max and eps_min < abs(eps2) < eps_max:
                        direct = True
                    if are_equal(eps1_oth, eps2_oth, eq_eps) and eps_min < abs(eps1_oth) < eps_max and eps_min < abs(
                            eps2_oth) < eps_max:
                        other = True
                    if direct:
                        if other:
                            if self._func_opt_eps(eps1, eps2) < self._func_opt_eps(eps1_oth, eps2_oth):
                                gs_supercell = {"sub": sub_cell, "gr": gr_cell, "eps1": eps1, "eps2": eps2,
                                                "eps_oth": False}
                                gs_supercells.append(gs_supercell)
                            else:
                                gs_supercell = {"sub": sub_cell, "gr": gr_cell, "eps1": eps1_oth, "eps2": eps2_oth,
                                                "eps_oth": True}
                                gs_supercells.append(gs_supercell)
                        else:
                            gs_supercell = {"sub": sub_cell, "gr": gr_cell, "eps1": eps1, "eps2": eps2, "eps_oth": False}
                            gs_supercells.append(gs_supercell)
                    else:
                        if other:
                            gs_supercell = {"sub": sub_cell, "gr": gr_cell, "eps1": eps1_oth, "eps2": eps2_oth,
                                            "eps_oth": True}
                            gs_supercells.append(gs_supercell)
                l += 1
                current_time = time.time()
                if textmode and current_time - last_print_time >= 0.1:
                    print(f'{self.title_sub}_{self.lat_sub}_{id}' + f': Group in gs_supercells: ',
                          f'{l / N_all * 100:.8f}', '%', end='\r')
                    last_print_time = current_time
        if textmode:
            print(f'{self.title_sub}_{self.lat_sub}_{id}' + f': Group in gs_supercells: ', f'{100:.8f}', '%', end='\r')
            print(f'{self.title_sub}_{self.lat_sub}_{id}' + f': N of gs_supercells =', len(gs_supercells), '                            ')

        for gs_supercell in gs_supercells:
            gs_supercell["alpha"] = self._compute_alpha(gs_supercell["gr"], gs_supercell["sub"],
                                                        [self.g1, self.g2], [self.s1, self.s2])
            gs_supercell["S"] = gs_supercell["sub"]["V1_abs"] * gs_supercell["sub"]["V2_abs"] * np.sin(
                np.radians(gs_supercell["sub"]["beta"]))

        gs_groups = group_el(gs_supercells, self._eq_gs_supercell)
        gs_supercells = []
        for group in gs_groups:
            keyed_group = np.array([group[i]["S"] for i in range(len(group))])
            min_index = np.argmin(keyed_group)
            gs_supercells.append(group[min_index])
        if textmode:
            print(f'{self.title_sub}_{self.lat_sub}_{id}' + ': N of unique gs_supercells =', len(gs_supercells), '                   ')
        gs_supercells.sort(key=lambda x: x["S"])
        print(f'{self.title_sub}_{self.lat_sub}_{id}' + ': Job done\n')

        self.gs_supercells = gs_supercells
        if csv:
            self.write_supercells_in_csv(directory=directory_res, id=id)
        return gs_supercells

    def build_supercell(self, config:list=False, gs_supercell:dict=False, n_sub_layers:int=3, id:int=0,
                        textmode:bool=False, supercell_save:bool=True, directory_res=Path('./'),
                        save_nodef_graphene:bool=False):
        """
        Builds a supercell according to the configuration list config or gs_supercell object of type dict.
        Args:
            config (list): configuration list of 9 numbers: [k11, k12, k21, k22, l11, l12, l21, l22, eps_oth], where
            the first 8 numbers are the coordinates of the graphene and substrate cells in the basis of the corresponding lattice vectors,
            eps_oth - if 0, the first graphene vector is matched with the first substrate vector,
            if 1 - the first graphene vector is matched with the second substrate vector;
            gs_supercell (dict): supercell of type dict (see more in the description of this class);
            n_sub_layers (int): number of substrate layers;
            id (int): build id;
            textmode (str): if True, build information will be displayed on the screen;
            supercell_save (bool): if True, the build result will be written to a file;
            directory_res (str or Path): directory where the search result file will be written (if necessary);
            save_nodef_graphene (bool): if True, the build result of undeformed graphene will be written to a file.
        Returns:
            ase.Atoms: built cupercell in ase.Atoms format
        """
        directory_res = Path(directory_res)
        if type(gs_supercell) is bool:
            gs_supercell = self.create_supercell(config)

        angle_grsub = gs_supercell["alpha"]
        M1, M2 = gs_supercell["sub"]["V1"], gs_supercell["sub"]["V2"]
        l11, l12, l21, l22 = int(gs_supercell["sub"]["n11"]), int(gs_supercell["sub"]["n12"]), int(
            gs_supercell["sub"]["n21"]), int(gs_supercell["sub"]["n22"])
        if not gs_supercell["eps_oth"]:
            G1, G2 = gs_supercell["gr"]["V1"], gs_supercell["gr"]["V2"]
            k11, k12, k21, k22 = int(gs_supercell["gr"]["n11"]), int(gs_supercell["gr"]["n12"]), int(
                gs_supercell["gr"]["n21"]), int(gs_supercell["gr"]["n22"])
        else:
            G1, G2 = gs_supercell["gr"]["V2"], gs_supercell["gr"]["V1"]
            k11, k12, k21, k22 = int(gs_supercell["gr"]["n21"]), int(gs_supercell["gr"]["n22"]), int(
                gs_supercell["gr"]["n11"]), int(gs_supercell["gr"]["n12"])
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
            if self._atom_not_inside(atom, G1, G2):
                continue
            if are_equal(atom[0], G1[0], 1.e-5) and are_equal(atom[1], G1[1], 1.e-5):
                continue
            if are_equal(atom[0], G2[0], 1.e-5) and are_equal(atom[1], G2[1], 1.e-5):
                continue
            if self._on_bounary(atom, G1, G2):
                continue
            gr_surface.append([atom[0], atom[1], 0])
        if textmode:
            print(f'{self.title_sub}{id}: Graphen: ', len(gr_surface))

        if save_nodef_graphene:
            swap = False
            if vector_triple_orientation(G1, G2) == -1:
                swap = True
                tmp = G2
                G2 = G1
                G1 = tmp
            with open(directory_res / f'Gr{self.title_sub}{id}_{angle_grsub:.3f}_{beta:.3f}_nodef_graphene.xyz', 'w',
                      newline='') as file:
                file.write(f'{len(gr_surface)}')
                file.write('\n\n')
                for atom in gr_surface:
                    file.write(f'C {atom[0]:.16f} {atom[1]:.15f} {(self.z_dist / 2):.15f}\n')
                file.write('\n')
            with open(directory_res / f'Gr{self.title_sub}{id}_{angle_grsub:.3f}_{beta:.3f}_nodef_graphene.txt', 'w',
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

        ####  Substrate

        min_m1 = min([0, l11, l21, l11 + l21])
        max_m1 = max([0, l11, l21, l11 + l21])
        min_m2 = min([0, l12, l22, l12 + l22])
        max_m2 = max([0, l12, l22, l12 + l22])
        l1_sub = max_m1 - min_m1
        l2_sub = max_m2 - min_m2
        size1 = l1_sub + 6
        size2 = l2_sub + 6
        cell_sub_all = self._build_substrate(size1, size2, n_sub_layers)
        t = cell_sub_all[0][2]
        for atom in cell_sub_all:
            atom[2] -= t
            atom[2] += 1

        Med = (size2 // 2 - 1) * size1 + size1 // 2 - 1
        trans = -M1 / 2 - M2 / 2
        coord = [cell_sub_all[Med][0] + trans[0], cell_sub_all[Med][1] + trans[1]]
        cell_sub_tmp = cell_sub_all[0:size1 * size2]
        eps_nach = []
        for i in range(size1 * size2):
            a = np.array([cell_sub_tmp[i][0] - coord[0], cell_sub_tmp[i][1] - coord[1]])
            eps_nach.append(np.linalg.norm(a))
        j = np.argmin(eps_nach)
        coord_nach = [cell_sub_tmp[j][0], cell_sub_tmp[j][1]]
        for atom in cell_sub_all:
            atom[0] -= coord_nach[0]
            atom[1] -= coord_nach[1]

        cell_sub = []
        for atom in cell_sub_all:
            if self._atom_not_inside(atom[:-1], M1, M2):
                continue
            if are_equal(atom[0], M1[0], 1.e-5) and are_equal(atom[1], M1[1], 1.e-5):
                continue
            if are_equal(atom[0], M2[0], 1.e-5) and are_equal(atom[1], M2[1], 1.e-5):
                continue
            if self._on_bounary(atom[:-1], M1, M2):
                continue
            cell_sub.append([atom[0], atom[1], atom[2]])
        if textmode:
            print(f'{self.title_sub}{id}: Substrate ready:', len(cell_sub))

        # Recalculate med_gr
        recalculate = False
        beta_gr = angle_between_vectors(G1, G2)
        beta_sub = angle_between_vectors(M1, M2)
        if not are_equal(beta_gr, beta_sub, 1.e-5):
            for atom in gr_surface:
                atom[0] -= G1[0]
                atom[1] -= G1[1]
            G1 = -G1
            recalculate = True
        med_sub = M1 + M2
        med_sub /= np.linalg.norm(med_sub)
        med_gr = G1 + G2
        med_gr /= np.linalg.norm(med_gr)

        # Reflex
        reflex = False
        if vector_triple_orientation(M1, M2) != vector_triple_orientation(G1, G2) and not are_equal(M1_abs, M2_abs, 1.e-5):
            i = np.array([-med_gr[1], med_gr[0]])
            H = np.eye(2) - 2 * np.outer(i, i)
            gr_surface = linear_map(gr_surface, H)
            reflex = True

        # Rotate Gr to Substrate
        alpha = angle_between_vectors(med_sub, med_gr)
        single = vector_triple_orientation(med_sub, med_gr)
        R = rotate_matrix(alpha)
        if single == 1:
            R = np.transpose(R)
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
        for atom in cell_sub:
            supercell.append([self.title_sub, atom[0], atom[1], atom[2]])
            if atom[2] > max1:
                max1 = atom[2]
        for atom in gr_surface:
            atom[2] += (max1 + self.distance)
            supercell.append(['C', atom[0], atom[1], atom[2]])

        cell0 = np.array([cell[0][0], cell[0][1]])
        cell1 = np.array([cell[1][0], cell[1][1]])
        zero = [1, 0]
        angle = angle_between_vectors(cell0, zero)
        single = vector_triple_orientation(cell0, zero)
        R = rotate_matrix(angle)
        if single == 1:
            new0 = np.dot(R, cell0)
            new1 = np.dot(R, cell1)
        else:
            new0 = np.dot(np.transpose(R), cell0)
            new1 = np.dot(np.transpose(R), cell1)
        if vector_triple_orientation(zero, new1) == 1:
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
            angle = angle_between_vectors(cell1, zero)
            single = vector_triple_orientation(cell1, zero)
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
        S = np.linalg.norm(v1) * np.linalg.norm(v2) * np.sin(np.radians(angle_between_vectors(v1, v2)))
        if textmode:
            print(f'{self.title_sub}{id}: Job done')
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
            with open(directory_res / f'Gr{self.title_sub}{id}_{angle_grsub:.3f}_{beta:.3f}.xyz', 'w',
                      newline='') as file:
                file.write(f'{len(supercell)}' + '\n')
                file.write('\n')
                for atom in supercell:
                    file.write(atom[
                                   0] + '  ' + f'{round(atom[1], 16):3.16f}  ' + f'{round(atom[2], 16):3.16f}  ' + f'{round(atom[3], 16):3.16f}' + '\n')
            with open(directory_res / f'Gr{self.title_sub}{id}_{angle_grsub:.3f}_{beta:.3f}.txt', 'w',
                      newline='') as file:
                file.write(r'CELL_PARAMETERS {angstrom}' + '\n')
                for i in range(3):
                    for j in range(3):
                        file.write(f'{round(cell[i][j], 16):3.16f}  ')
                    file.write('\n')
                file.write('\n')
                file.write(f"alpha = {angle_grsub:.3f}, " + f"beta = {beta:.3f}; S = {S:.3f}\n")
                file.write(f'Gr: {k11} {k12} {k21} {k22}; ' + str(self.gr_a) + '\n')
                file.write(
                    f'{self.title_sub}: {l11} {l12} {l21} {l22}; ' + str(self.sub_a1) + ' ' + str(self.sub_a2) + '\n')
                file.write(f'eps1 = {eps1 * 100:.3f}, eps2 = {eps2 * 100:.3f} \n \n')
                file.write(f'Number of atoms: {len(supercell)}' + '\n' + '\n')
                file.write(f'{code_str}' + '\n')

        symbols = [atom[0] for atom in supercell]
        positions = [[atom[1], atom[2], atom[3]] for atom in supercell]
        v1 = [cell[0][0], cell[0][1], cell[0][2]]
        v2 = [cell[1][0], cell[1][1], cell[1][2]]
        v3 = [cell[2][0], cell[2][1], cell[2][2]]

        supercell_atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=[v1, v2, v3],
            pbc=True
        )
        return supercell_atoms

    def build_supercells_from_csv(self, filepath, n_sub_layers: int = 3, id_start: int = 0,
                                  textmode: bool = False, supercell_save: bool = True, directory_res=Path('./'),
                                  save_nodef_graphene: bool = False):
        """
        Builds supercells from a .csv file.

        Args:
            filepath (str or Path): path to the file;
            n_sub_layers (int): number of substrate layers;
            id_start (int): initial build id;
            textmode (str): if True, build information will be displayed on the screen;
            supercell_save (bool): if True, the build result will be written to a file;
            directory_res (str or Path): directory where the search result file will be written (if necessary);
            save_nodef_graphene (bool): if True, the build result of undeformed graphene will be written to a file.

        Returns:
            list: list of built supercells in ase.Atoms format
        """
        directory_res = Path(directory_res)
        df = pd.read_csv(filepath)
        supercells = []
        for i in range(df.shape[0]):
            config = [df.iloc[i]["k11"], df.iloc[i]["k12"], df.iloc[i]["k21"], df.iloc[i]["k22"],
                      df.iloc[i]["l11"], df.iloc[i]["l12"], df.iloc[i]["l21"], df.iloc[i]["l22"], df.iloc[i]["eps_oth"]]
            supercell = self.build_supercell(config=config, directory_res=directory_res, n_sub_layers=n_sub_layers,
                                             id=id_start + i, textmode = textmode, supercell_save=supercell_save,
                                             save_nodef_graphene=save_nodef_graphene)
            supercells.append(supercell)
        return supercells

    def build_supercells_list(self, gs_supercells: list = False, n_sub_layers: int = 3, id_start: int = 0,
                              textmode:bool=False, supercell_save:bool=True, directory_res=Path('./'),
                              save_nodef_graphene: bool=False):
        """
        Builds supercells from the gs_supercells list or from the gs_supercells attribute of the class.

        Args:
            gs_supercells (dict): list of supercells of type dict (see more in the description of this class)
            n_sub_layers (int): number of substrate layers;
            id_start (int): initial build id;
            textmode (str): if True, build information will be displayed on the screen;
            supercell_save (bool): if True, the build result will be written to a file;
            directory_res (str or Path): directory where the search result file will be written (if necessary);
            save_nodef_graphene (bool): if True, the build result of undeformed graphene will be written to a file.

        Returns:
            list: list of built supercells in ase.Atoms format
        """

        directory_res = Path(directory_res)
        if type(gs_supercells) is bool:
            gs_supercells = self.gs_supercells
        supercells = []
        for i in range(len(gs_supercells)):
            supercell = self.build_supercell(gs_supercell=gs_supercells[i], directory_res=directory_res, n_sub_layers=n_sub_layers,
                                             id=id_start + i, textmode=textmode, supercell_save=supercell_save,
                                             save_nodef_graphene=save_nodef_graphene)
            supercells.append(supercell)
        return supercells

    def compute_mismatch_in_basis(self, gs_supercell:dict, basis:list=[[1., 0.], [0., 1.]]):
        """
        Calculates the mismatch matrix in the required basis.
        Args:
            gs_supercell (dict): supercell of type dict (see more in the description of this class);
            basis (list): list of two basis vectors. By default in the unit basis
            the elements of the mismatch matrix: '00' - deformation orthogonal to the C-C bond, '11' - deformation along the C-C bond,
            '01' and '10' - shear deformation.
        Returns:
            np.ndarray: mismatch matrix (in %).
        """
        S = np.zeros((2, 2))
        S[0] = gs_supercell["gr"]["V1"]
        S[1] = gs_supercell["gr"]["V2"]
        S = np.transpose(S)
        Eps = np.zeros((2, 2))
        Eps[0][0] = 1 + gs_supercell["eps1"] / 100
        Eps[1][1] = 1 + gs_supercell["eps2"] / 100

        B = np.zeros((2, 2))
        B[0] = basis[0]
        B[1] = basis[1]
        B = np.transpose(B)
        return np.linalg.multi_dot([np.linalg.inv(B), S, Eps, np.linalg.inv(S), B])

    def compute_deform_of_cc(self, gs_supercell: dict = False, config: list = False):
        """
        Calculates the deformation of the supercell along the C-C bond and orthogonal to it.
        Supercell - according to the configuration list config or gs_supercell object of type dict.

        Args:
            config (list): configuration list of 9 numbers: [k11, k12, k21, k22, l11, l12, l21, l22, eps_oth], where
            the first 8 numbers are the coordinates of the graphene and substrate cells in the basis of the corresponding lattice vectors,
            eps_oth - if 0, the first graphene vector is matched with the first substrate vector,
            if 1 - the first graphene vector is matched with the second substrate vector;
            gs_supercell (dict): supercell of type dict (see more in the description of this class);

        Returns:
            float, float: deformation along the C-C bond, deformation orthogonal to the C-C bond (in %).
        """
        if type(gs_supercell) is bool:
            gs_supercell = self.create_supercell(config)
        M1, M2 = gs_supercell["sub"]["V1"], gs_supercell["sub"]["V2"]
        if not gs_supercell["eps_oth"]:
            G1, G2 = gs_supercell["gr"]["V1"], gs_supercell["gr"]["V2"]
        else:
            G1, G2 = gs_supercell["gr"]["V2"], gs_supercell["gr"]["V1"]
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