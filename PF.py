import numpy as np
import sys
from PFClasses import GlobalFunctions
from PFClasses import BasisClass
from PFClasses import ModelClass
from PFClasses import SectorClass

#Import file with basis vectors and GSO phase matrix
with open("BasisE1.txt", "r") as InBasis:
     Basis = np.loadtxt(InBasis)

"""
with open("GSOGolden.txt", "r") as InGSO:
     GSO = np.loadtxt(InGSO)
"""

#Define an instance of the class of models with a certain basis
ABasisSet = BasisClass(Basis)

A = 0
SUSY = 0
#Define an instance of a specific model with basis set and a random GSO matrix
while (A != 0 or SUSY == 0):
    AModel = ModelClass(Basis,ABasisSet.random_gso())

    [PF,CoC] = AModel.partition_function(1)

    print(CoC)

    A = PF[9,9]
    SUSY = np.sum(PF[1::,1::])

    with open("GSO.txt","w+") as Out:
            np.savetxt(Out,AModel.gso, delimiter=' ',fmt="%d")

    with open("Q.txt","w+") as Out:
            np.savetxt(Out,PF, delimiter=' ',fmt="%d")

sys.exit()
c=0
NonTachGSOs = []
while c<10:
    AModel = ModelClass(Basis,ABasisSet.random_gso())
    if AModel.tachyon_check() == True:
        NonTachGSOs.append(AModel.gso)
        c+=1
with open("NonTachGSOs.txt","w+") as Out:
        np.savetxt(Out,np.concatenate(NonTachGSOs,axis=0), delimiter=' ',fmt="%d")

#Print the q-expansion of the partition function (to second order in q)
print(AModel.partition_function(0)[0])
