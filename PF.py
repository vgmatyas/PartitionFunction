import numpy as np
import sys
import multiprocessing
from PFClasses import GlobalFunctions
from PFClasses import BasisClass
from PFClasses import ModelClass
from PFClasses import SectorClass

with open("InputData/Basis10D.txt", "r") as InBasis:
     Basis = np.loadtxt(InBasis)

with open("InputData/GSO10D.txt", "r") as InGSO:
     GSO = np.loadtxt(InGSO)

AModel = ModelClass(Basis,GSO)

print(AModel.basis_prod_matrix())
