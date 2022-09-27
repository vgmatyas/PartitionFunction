import numpy as np
from PFClasses import GlobalFunctions
from PFClasses import BasisClass
from PFClasses import ModelClass
from PFClasses import SectorClass

with open("Basis.txt", "r") as InBasis:
     Basis = np.loadtxt(InBasis)
with open("GSORizosA.txt", "r") as InGSO:
     GSO = np.loadtxt(InGSO)


ABasis = BasisClass(Basis)

AModel = ModelClass(Basis,ABasis.random_gso())

AModel.tachyon_check()



"""
for row in BC.sectors()[1]:
 SC = SectorClass(Basis,GSO,row)
 SC_PF = SC.sector_partition_function(1)
 if SC_PF[9,9] != 0:
   print(SC.b_sector,SC_PF[9,9])
"""
