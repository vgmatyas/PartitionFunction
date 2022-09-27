import numpy as np
from PFClasses import GlobalFunctions
from PFClasses import BasisClass
from PFClasses import ModelClass
from PFClasses import SectorClass

#Import file with basis vectors and GSO phase matrix
with open("Basis.txt", "r") as InBasis:
     Basis = np.loadtxt(InBasis)
with open("GSORizosA.txt", "r") as InGSO:
     GSO = np.loadtxt(InGSO)


#Define an instance of the class of models with a certain basis
ABasisSet = BasisClass(Basis)

#Define a specific model with basis set and GSO matrix
AModel = ModelClass(Basis,GSO)


#Define an instance of a specific model with basis set and a random GSO matrix
AModel = ModelClass(Basis,ABasisSet.random_gso())

#Check model for tachyons
AModel.tachyon_check()

#Print the q-expansion of the partition function (to second order in q)
print(AModel.partition_function(1)[0])

#Print the worldsheet vacuum energy (to second order in q)
print(AModel.partition_function(1)[1])


#Define an instance of a sectro within a specific model with a basis set, GSO matrix and a sector
ASector = SectorClass(Basis,GSO,[0,0,0,0,0,0,0,0,1])

#Print partition function contribution of the given sector
print(ASector.partition_function(1)[0])

#Print vacuum energy contribution of the given sector
print(ASector.partition_function(1)[1])
