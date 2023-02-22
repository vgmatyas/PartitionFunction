import numpy as np
import itertools
import random
import sys
from numpy.polynomial import polynomial as p

with open("FixedData/QCoeffPowersRe.txt", "r") as InQCRe:
     QCRe = np.loadtxt(InQCRe)
with open("FixedData/QCoeffPowersIm.txt", "r") as InQCIm:
     QCIm = np.loadtxt(InQCIm)
with open("FixedData/QCoeffEta.txt", "r") as InQCEt:
     QCEt = np.loadtxt(InQCEt)
with open("FixedData/QInt.txt", "r") as InQInt:
     QInt = np.loadtxt(InQInt)


class GlobalFunctions:

    @staticmethod
    def basis_prod(B1,B2,CompDim):
        BP = 0.5*np.dot(B1[0:8+CompDim*2],B2[0:8+CompDim*2]) - 0.5*np.dot(B1[8+CompDim*2:8+CompDim*4],B2[8+CompDim*2:8+CompDim*4]) - np.dot(B1[8+CompDim*4:8+CompDim*4+16],B2[8+CompDim*4:8+CompDim*4+16])
        return BP

    @staticmethod
    def PolyMul(Poly):
        if len(Poly) > 2:
          TRes = p.polymul(Poly[-1],Poly[-2])[:len(Poly[0])]
          return GlobalFunctions.PolyMul(np.vstack((Poly[:-2],np.pad(TRes,(0,len(Poly[0])-len(TRes))))))
        if len(Poly) == 2:
          TRes = p.polymul(Poly[0],Poly[1])[:len(Poly[0])]
          return np.array(np.pad(TRes,(0,len(Poly[0])-len(TRes))))

    @staticmethod
    def GSOSec(Sector,BSector,SectorUnRed,GSO,a,b,CompDim):
        if Sector[a][0] == 1:
          SDelta1 = -1
        elif Sector[a][0]  == 0:
          SDelta1 = 1
        if Sector[b][0] == 1:
          SDelta2 = -1
        elif Sector[b][0]  == 0:
          SDelta2 = 1
        SGSO1 = (SDelta1**(np.sum(BSector[b])-1) * SDelta2**(np.sum(BSector[a])-1))
        SGSO2 = np.around(np.exp(1j*np.pi*GlobalFunctions.basis_prod((Sector[a][:]-SectorUnRed[a][:]),SectorUnRed[b][:],CompDim)/2))
        SGSO3 = 1
        for k in range(BSector.shape[1]):
         for l in range(BSector.shape[1]):
           TSGSO3 = GSO[k][l]**(BSector[a][k]*BSector[b][l])
           SGSO3 = SGSO3 * TSGSO3
        SecGSO = SGSO1 * SGSO2 * SGSO3
        return SecGSO




class BasisClass:

    def __init__(self,basis):
        BasisClass.basis = basis

    def num_basis(self):
        return self.basis.shape[0]

    def comp_dim(self):
        return int((self.basis.shape[1]-24)/4)

    def num_sector(self):
        NumBas=self.num_basis()
        CBasis = np.zeros(NumBas)
        for i in range(NumBas):
         for k in range(self.basis.shape[1]):
          if self.basis[i][k] % 1 != 0:
            CBasis[i] = CBasis[i]+1
        NumSec = 1
        for i in range(NumBas):
            if CBasis[i] == 0:
                NumSec = NumSec*2
                CBasis[i] = int(2)
            elif CBasis[i] != 0:
                NumSec = NumSec*4
                CBasis[i] = int(4)
        return NumSec, CBasis.astype(int)

    def sectors(self):
        NumBas = self.num_basis()
        NumSec =self.num_sector()[0]
        Sector = np.zeros((NumSec,self.basis.shape[1]))
        BSector = np.zeros((NumSec,NumBas))
        rngs = self.num_sector()[1]
        for i,t in enumerate(itertools.product(*[range(i) for i in rngs])):
            Sector[i,:] = sum([self.basis[i,:] * t[i] for i in range(len(t))])
            BSector[i,:] = t
        SectorUnRed = Sector.copy()
        SectorUnRed[0][:] = 2
        Sector = Sector % 2
        for i in range(NumSec):
            for j in range(Sector.shape[1]):
                if Sector[i][j] == 1.5:
                    Sector[i][j] = -0.5
        BSector[0][0] = 2
        return Sector, BSector.astype(int), SectorUnRed

    def sector_mass(self):
        Sec = self.sectors()[0]
        CompDim = self.comp_dim()
        M = np.zeros((Sec.shape[0],2))
        SL = Sec.copy()
        SR = Sec.copy()
        SL[::,8+CompDim*2:8+CompDim*4+16] = 0
        SR[::,0:8+CompDim*2] = 0
        for i in range(Sec.shape[0]):
             M[i][0] = -0.5 + GlobalFunctions.basis_prod(SL[i,::],SL[i,::],CompDim)/8
             M[i][1] = -1 - GlobalFunctions.basis_prod(SR[i,::],SR[i,::],CompDim)/8
        return M

    def random_gso(self):
         CompDim = self.comp_dim()
         Basis = self.basis
         NumG = int((Basis.shape[0]*(Basis.shape[0]-1)+2)/2)
         G = np.zeros(NumG)
         for j in range(NumG):
           R = random.random()
           if R < 0.5:
               G[j] = -1
           if R > 0.5:
               G[j] = +1
         GSO = np.zeros((Basis.shape[0],Basis.shape[0]),dtype=np.complex128)
         GSO[0][0] = G[0]
         c=0
         for i in range(Basis.shape[0]):
          for j in range(Basis.shape[0]):
            TBP = GlobalFunctions.basis_prod(Basis[i],Basis[j],CompDim)
            if i<j:
              c += 1
              if TBP%2 == 0:
                GSO[i][j] = G[c]
              if TBP%2 != 0:
                GSO[i][j] = 1j * G[c]
         for i in range(Basis.shape[0]):
          for j in range(Basis.shape[0]):
            if i>j:
              GSO[i][j] = np.around(np.exp(1j*np.pi*GlobalFunctions.basis_prod(Basis[j],Basis[i],CompDim)/2))*np.conj(GSO[j][i])
            if i==j:
              GSO[i][j] = -np.around(np.exp(1j*np.pi*GlobalFunctions.basis_prod(Basis[i],Basis[j],CompDim)/4))*GSO[i][0]

         if np.sum(np.imag(GSO)) == 0:
            return np.real(GSO).astype(int)

         if np.sum(np.imag(GSO)) != 0:
            return GSO

    def theta_matrix_real(self):
        Sector = self.sectors()[0]
        CompDim = self.comp_dim()
        SectorL=Sector[::,:8+CompDim*2:]
        SectorRR=Sector[::,8+CompDim*2:8+CompDim*4:]
        SectorRC=Sector[::,8+CompDim*4:8+CompDim*4+16:]
        CSectorL = 1 - SectorL
        CSectorRR = 1 - SectorRR
        CSectorRC = 1 - SectorRC
        MT1 = np.dot(SectorL, SectorL.T)/2
        MT4 = np.dot(CSectorL, SectorL.T)/2
        MT2 = np.dot(SectorL, CSectorL.T)/2
        MT3 = np.dot(CSectorL, CSectorL.T)/2
        MTb1 = np.dot(SectorRR, SectorRR.T)/2 + np.dot(SectorRC, SectorRC.T)
        MTb2 = np.dot(SectorRR, CSectorRR.T)/2 + np.dot(SectorRC, CSectorRC.T)
        MTb3 = np.dot(CSectorRR, CSectorRR.T)/2 + np.dot(CSectorRC, CSectorRC.T)
        MTb4 = np.dot(CSectorRR, SectorRR.T)/2 + np.dot(CSectorRC, SectorRC.T)
        return MT1,MT2,MT3,MT4,MTb1,MTb2,MTb3,MTb4

    def theta_matrix_complex(self):
        Sector = self.sectors()[0]
        CompDim = self.comp_dim()
        MT1 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MT2 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MT3 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MT4 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb1 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb2 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb3 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb4 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb5 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb6 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb7 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb8 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb9 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb10 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb11 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb12 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb13 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb14 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb15 = np.zeros((Sector.shape[0],Sector.shape[0]))
        MTb16 = np.zeros((Sector.shape[0],Sector.shape[0]))
        for i in range(Sector.shape[0]):
           for j in range(Sector.shape[0]):
               for k in range(8+CompDim*2):
                   if Sector[i][k]==1 and Sector[j][k]==1:
                       MT1[i][j] += 1/2
                   if Sector[i][k]==1 and Sector[j][k]==0:
                       MT2[i][j] += 1/2
                   if Sector[i][k]==0 and Sector[j][k]==0:
                       MT3[i][j] += 1/2
                   if Sector[i][k]==0 and Sector[j][k]==1:
                       MT4[i][j] += 1/2
               for k in range(8+CompDim*2,8+CompDim*4):
                   if Sector[i][k]==1 and Sector[j][k]==1:
                       MTb1[i][j] += 1/2
                   if Sector[i][k]==1 and Sector[j][k]==0:
                       MTb2[i][j] += 1/2
                   if Sector[i][k]==0 and Sector[j][k]==0:
                       MTb3[i][j] += 1/2
                   if Sector[i][k]==0 and Sector[j][k]==1:
                       MTb4[i][j] += 1/2
               for k in range(8+CompDim*4,8+CompDim*4+16):
                   if Sector[i][k]==1 and Sector[j][k]==1:
                       MTb1[i][j] += 1
                   if Sector[i][k]==1 and Sector[j][k]==0:
                       MTb2[i][j] += 1
                   if Sector[i][k]==0 and Sector[j][k]==0:
                       MTb3[i][j] += 1
                   if Sector[i][k]==0 and Sector[j][k]==1:
                       MTb4[i][j] += 1
                   if Sector[i][k]==1 and Sector[j][k]==1/2:
                       MTb5[i][j] += 1
                   if Sector[i][k]==1 and Sector[j][k]==-1/2:
                       MTb6[i][j] += 1
                   if Sector[i][k]==0 and Sector[j][k]==1/2:
                       MTb7[i][j] += 1
                   if Sector[i][k]==0 and Sector[j][k]==-1/2:
                       MTb8[i][j] += 1
                   if Sector[i][k]==1/2 and Sector[j][k]==1:
                       MTb9[i][j] += 1
                   if Sector[i][k]==1/2 and Sector[j][k]==0:
                       MTb10[i][j] += 1
                   if Sector[i][k]==1/2 and Sector[j][k]==1/2:
                       MTb11[i][j] += 1
                   if Sector[i][k]==1/2 and Sector[j][k]==-1/2:
                       MTb12[i][j] += 1
                   if Sector[i][k]==-1/2 and Sector[j][k]==1:
                       MTb13[i][j] += 1
                   if Sector[i][k]==-1/2 and Sector[j][k]==0:
                       MTb14[i][j] += 1
                   if Sector[i][k]==-1/2 and Sector[j][k]==1/2:
                       MTb15[i][j] += 1
                   if Sector[i][k]==-1/2 and Sector[j][k]==-1/2:
                       MTb16[i][j] += 1
        return MT1,MT2,MT3,MT4,MTb1,MTb2,MTb3,MTb4,MTb5,MTb6,MTb7,MTb8,MTb9,MTb10,MTb11,MTb12,MTb13,MTb14,MTb15,MTb16





class ModelClass(BasisClass):

    def __init__(self,basis,gso):
        super().__init__(basis)
        self.gso = gso

    def gso_check(self):
        n1 = 0
        n2 = 0
        for i in range(self.num_basis()):
            for k in range(self.num_basis()):
                TGSO1 = -np.round(np.exp(1j*np.pi*GlobalFunctions.basis_prod(self.basis[i],self.basis[k],self.comp_dim())/4)*self.gso[i][0])
                TGSO2 = np.round(np.exp(1j*np.pi*GlobalFunctions.basis_prod(self.basis[i],self.basis[k],self.comp_dim())/2)*np.conj(self.gso[k][i]))
            if i == k and TGSO1 != self.gso[i][k]:
                n1 += 1
                #print(k)
            elif i != k and TGSO2 != self.gso[i][k]:
                n2 += 1
                #print(i,k)
        if n1 != 0 or n2 != 0:
            #print("~ Error: Basis GSOs not modular invariant!")
            return False
        else:
            #print("~ Basis GSOs OK!")
            return True

    def basis_prod_matrix(self):
        BP = np.zeros((self.num_basis(),self.num_basis()))
        for i in range(self.num_basis()):
            for k in range(self.num_basis()):
                BP[i,k] = GlobalFunctions.basis_prod(self.basis[i],self.basis[k],self.comp_dim())
        return BP



    def partition_function(self,q_order):

        TQ = QCRe[::,:32*(q_order+1)+1:] + 1j*QCIm[::,:32*(q_order+1)+1:]
        E =  QCEt[0,0:8*(q_order+1)+9]
        Eb = QCEt[1,0:8*(q_order+1)+9]

        CompDim = self.comp_dim()

        [NumSec,CBasis] = self.num_sector()

        [Sector,BSector,SectorUnRed] = self.sectors()

        if 4 not in CBasis:
            [MT1,MT2,MT3,MT4,MTb1,MTb2,MTb3,MTb4] = self.theta_matrix_real()
            TPF = np.zeros((32*(q_order+1)+1,32*(q_order+1)+1),dtype=np.complex128)
            for i in range(Sector.shape[0]):
               for j in range(Sector.shape[0]):
                  if MT1[i][j]==0 and MTb1[i][j]==0:
                    TPFT1 = GlobalFunctions.PolyMul(np.array([TQ[int(MT2[i][j]),::],TQ[25+int(MT3[i][j]),::],TQ[50+int(MT4[i][j]),::]]))
                    TPFT2 = GlobalFunctions.PolyMul(np.array([TQ[375+int(MTb2[i][j]),::],TQ[400+int(MTb3[i][j]),::],TQ[425+int(MTb4[i][j]),::]]))
                    TPFT = GlobalFunctions.GSOSec(Sector,BSector,SectorUnRed,self.gso,i,j,CompDim) * np.tensordot(TPFT1,TPFT2,axes=0)
                    TPF += TPFT
            QPF = np.around(np.real(TPF)/NumSec).astype(int)
            if np.sum(np.around(np.imag(TPF))) != 0:
                print("Error: Imaginary PF Does Not Vanish! ", np.sum(np.around(np.imag(TPF))))
            for i in range(QPF.shape[0]):
                for j in range(QPF.shape[1]):
                    if (i%4 != 0 or j%4 != 0) and QPF[i][j] != 0:
                      print("Error: Disallowed Terms in q-Expansion! ",(i-32),(j-32),QPF[i][j])
            QPF = QPF[0::4,0::4]
            A = np.pad(QPF,((8,0),(8,0)), mode='constant')
            B = np.tensordot(E,Eb,axes=0)
            C = np.tensordot(A,B,axes=0)
            D = np.zeros((C.shape[0],C.shape[0],2*C.shape[0]-1,2*C.shape[0]-1),dtype=int)
            for k in range(D.shape[0]):
             for l in range(D.shape[1]):
               D[k,l,k:k+D.shape[0],l:l+D.shape[1]] = C[k,l,:,:]
            D = D.sum(axis=0).sum(axis=0)[8:int(8*(q_order+1))+8+1,8:int(8*(q_order+1))+8+1]
            DOut = np.pad(D,((1,0),(1,0)), mode='constant')
            DOut[0][0] = 8
            for i in range(1,DOut.shape[0]):
              DOut[i][0] = i-(8+1)
              DOut[0][i] = i-(8+1)

        else:
            [MT1,MT2,MT3,MT4,MTb1,MTb2,MTb3,MTb4,MTb5,MTb6,MTb7,MTb8,MTb9,MTb10,MTb11,MTb12,MTb13,MTb14,MTb15,MTb16] = self.theta_matrix_complex()
            TPF = np.zeros((32*(q_order+1)+1,32*(q_order+1)+1),dtype=np.complex128)
            for i in range(Sector.shape[0]):
               for j in range(Sector.shape[0]):
                  if MT1[i][j]==0 and MTb1[i][j]==0:
                    TPFT1 = GlobalFunctions.PolyMul(np.array([TQ[int(MT2[i][j]),::],TQ[25+int(MT3[i][j]),::],TQ[50+int(MT4[i][j]),::]]))
                    TPFT2 = GlobalFunctions.PolyMul(np.array([TQ[375+int(MTb2[i][j]),::],TQ[400+int(MTb3[i][j]),::],TQ[425+int(MTb4[i][j]),::],TQ[450+int(MTb5[i][j]),::],
                                                TQ[475+int(MTb6[i][j]),::],TQ[500+int(MTb7[i][j]),::],TQ[525+int(MTb8[i][j]),::],TQ[550+int(MTb9[i][j]),::],
                                                TQ[575+int(MTb10[i][j]),::],TQ[600+int(MTb11[i][j]),::],TQ[625+int(MTb12[i][j]),::],TQ[650+int(MTb13[i][j]),::],
                                                TQ[675+int(MTb14[i][j]),::],TQ[700+int(MTb15[i][j]),::],TQ[725+int(MTb16[i][j]),::]]))
                    TPFT = GlobalFunctions.GSOSec(Sector,BSector,SectorUnRed,self.gso,i,j,CompDim) * np.tensordot(TPFT1,TPFT2,axes=0)
                    TPF += TPFT
            QPF = np.around(np.real(TPF)/NumSec).astype(int)
            if np.sum(np.around(np.imag(TPF))) != 0:
                print("Error: Imaginary PF Does Not Vanish! ", np.sum(np.around(np.imag(TPF))))
            for i in range(QPF.shape[0]):
                for j in range(QPF.shape[1]):
                    if (i%4 != 0 or j%4 != 0) and QPF[i][j] != 0:
                      print("Error: Disallowed Terms in q-Expansion! ",(i-32),(j-32),QPF[i][j])
            QPF = QPF[0::4,0::4]
            A = np.pad(QPF,((8,0),(8,0)), mode='constant')
            B = np.tensordot(E,Eb,axes=0)
            C = np.tensordot(A,B,axes=0)
            D = np.zeros((C.shape[0],C.shape[0],2*C.shape[0]-1,2*C.shape[0]-1),dtype=int)
            for k in range(D.shape[0]):
             for l in range(D.shape[1]):
               D[k,l,k:k+D.shape[0],l:l+D.shape[1]] = C[k,l,:,:]
            D = D.sum(axis=0).sum(axis=0)[8:int(8*(q_order+1))+8+1,8:int(8*(q_order+1))+8+1]
            DOut = np.pad(D,((1,0),(1,0)), mode='constant')
            DOut[0][0] = 8
            for i in range(1,DOut.shape[0]):
              DOut[i][0] = i-(8+1)
              DOut[0][i] = i-(8+1)


        QIntegral = QInt[0:int(8*(q_order+1))+1,0:int(8*(q_order+1))+1]
        m=0
        for i in range(D.shape[0]):
         for j in range(D.shape[0]):
          if i!=j and QIntegral[i][j]==9 and D[i][j]!=0:
              print("~ Error: Divergent Terms Partition Function!")
              print(i+1,j+1,D[i][j])
              #sys.exit()
          if i==j and i<8 and j<8 and D[i][j] != 0:
              m+=1
              CoC = 999999
        if m==0:
         CoC = 0
         for i in range(D.shape[0]):
          for j in range(D.shape[0]):
           CoC += D[i][j] * QIntegral[i][j]

        return DOut,CoC


    def tachyon_check(self):

        q_order = 1

        TQ = QCRe[::,:32*(q_order+1)+1:] + 1j*QCIm[::,:32*(q_order+1)+1:]
        E =  QCEt[0,0:8*(q_order+1)+9]
        Eb = QCEt[1,0:8*(q_order+1)+9]

        CompDim = self.comp_dim()

        [NumSec,CBasis] = self.num_sector()

        [Sector,BSector,SectorUnRed] = self.sectors()

        SectorMass = self.sector_mass()

        Tach4 = []
        Tach3 = []
        Tach2 = []
        Tach1 = []

        if 4 not in CBasis:
            [MT1,MT2,MT3,MT4,MTb1,MTb2,MTb3,MTb4] = self.theta_matrix_real()
            TPF = np.zeros((32*(q_order+1)+1,32*(q_order+1)+1),dtype=np.complex128)
            for i in range(Sector.shape[0]):
              if SectorMass[i][0]<0 and SectorMass[i][1]<0:
                for j in range(Sector.shape[0]):
                      if MT1[i][j]==0 and MTb1[i][j]==0:
                        TPFT1 = GlobalFunctions.PolyMul(np.array([TQ[int(MT2[i][j]),::],TQ[25+int(MT3[i][j]),::],TQ[50+int(MT4[i][j]),::]]))
                        TPFT2 = GlobalFunctions.PolyMul(np.array([TQ[375+int(MTb2[i][j]),::],TQ[400+int(MTb3[i][j]),::],TQ[425+int(MTb4[i][j]),::]]))
                        TPFT = GlobalFunctions.GSOSec(Sector,BSector,SectorUnRed,self.gso,i,j,CompDim) * np.tensordot(TPFT1,TPFT2,axes=0)
                        TPF += TPFT
                QPF = np.around(np.real(TPF)/NumSec).astype(int)
                if np.sum(np.around(np.imag(TPF))) != 0:
                    print("Error: Imaginary PF Does Not Vanish! ", np.sum(np.around(np.imag(TPF))))
                for m in range(QPF.shape[0]):
                    for n in range(QPF.shape[1]):
                        if (m%4 != 0 or n%4 != 0) and QPF[m][n] != 0:
                          print("Error: Disallowed Terms in q-Expansion! ",(i-32),(j-32),QPF[m][n])
                QPF = QPF[0::4,0::4]
                A = np.pad(QPF,((8,0),(8,0)), mode='constant')
                B = np.tensordot(E,Eb,axes=0)
                C = np.tensordot(A,B,axes=0)
                D = np.zeros((C.shape[0],C.shape[0],2*C.shape[0]-1,2*C.shape[0]-1),dtype=int)
                for k in range(D.shape[0]):
                 for l in range(D.shape[1]):
                   D[k,l,k:k+D.shape[0],l:l+D.shape[1]] = C[k,l,:,:]
                D = D.sum(axis=0).sum(axis=0)[8:int(8*(q_order+1))+8+1,8:int(8*(q_order+1))+8+1]
                if D[7][7]!=0:
                   Tach1.append([i,D[7][7]])
                if D[6][6]!=0:
                   Tach2.append([i,D[6][6]])
                if D[5][5]!=0:
                   Tach3.append([i,D[5][5]])
                if D[4][4]!=0:
                   Tach4.append([i,D[4][4]])

            print("------- Tachyon Checker -------")
            print("Tachyons @ -1/2:")
            for i in range(len(Tach4)):
                print(BSector[Tach4[i][0]]," ",Tach4[i][1])
            if len(Tach4)==0:
                print("None")

            print("Tachyons @ -3/8:")
            for i in range(len(Tach3)):
                print(BSector[Tach3[i][0]]," ",Tach3[i][1])
            if len(Tach3)==0:
                print("None")

            print("Tachyons @ -1/4:")
            for i in range(len(Tach2)):
                print(BSector[Tach2[i][0]]," ",Tach2[i][1])
            if len(Tach2)==0:
                print("None")

            print("Tachyons @ -1/8:")
            for i in range(len(Tach1)):
                print(BSector[Tach1[i][0]]," ",Tach1[i][1])
            if len(Tach1)==0:
                print("None")



        else:
            [MT1,MT2,MT3,MT4,MTb1,MTb2,MTb3,MTb4,MTb5,MTb6,MTb7,MTb8,MTb9,MTb10,MTb11,MTb12,MTb13,MTb14,MTb15,MTb16] = self.theta_matrix_complex()
            TPF = np.zeros((32*(q_order+1)+1,32*(q_order+1)+1),dtype=np.complex128)
            for i in range(Sector.shape[0]):
              if SectorMass[i][0]<0 and SectorMass[i][1]<0:
                for j in range(Sector.shape[0]):
                  if MT1[i][j]==0 and MTb1[i][j]==0:
                    TPFT1 = GlobalFunctions.PolyMul(np.array([TQ[int(MT2[i][j]),::],TQ[25+int(MT3[i][j]),::],TQ[50+int(MT4[i][j]),::]]))
                    TPFT2 = GlobalFunctions.PolyMul(np.array([TQ[375+int(MTb2[i][j]),::],TQ[400+int(MTb3[i][j]),::],TQ[425+int(MTb4[i][j]),::],TQ[450+int(MTb5[i][j]),::],
                                                TQ[475+int(MTb6[i][j]),::],TQ[500+int(MTb7[i][j]),::],TQ[525+int(MTb8[i][j]),::],TQ[550+int(MTb9[i][j]),::],
                                                TQ[575+int(MTb10[i][j]),::],TQ[600+int(MTb11[i][j]),::],TQ[625+int(MTb12[i][j]),::],TQ[650+int(MTb13[i][j]),::],
                                                TQ[675+int(MTb14[i][j]),::],TQ[700+int(MTb15[i][j]),::],TQ[725+int(MTb16[i][j]),::]]))
                    TPFT = GlobalFunctions.GSOSec(Sector,BSector,SectorUnRed,self.gso,i,j,CompDim) * np.tensordot(TPFT1,TPFT2,axes=0)
                    TPF += TPFT
                QPF = QPF[0::4,0::4]
                A = np.pad(QPF,((8,0),(8,0)), mode='constant')
                B = np.tensordot(E,Eb,axes=0)
                C = np.tensordot(A,B,axes=0)
                D = np.zeros((C.shape[0],C.shape[0],2*C.shape[0]-1,2*C.shape[0]-1),dtype=int)
                for k in range(D.shape[0]):
                 for l in range(D.shape[1]):
                   D[k,l,k:k+D.shape[0],l:l+D.shape[1]] = C[k,l,:,:]
                D = D.sum(axis=0).sum(axis=0)[8:int(8*(q_order+1))+8+1,8:int(8*(q_order+1))+8+1]
                if D[7][7]!=0:
                   Tach1.append([i,D[7][7]])
                if D[6][6]!=0:
                   Tach2.append([i,D[6][6]])
                if D[5][5]!=0:
                   Tach3.append([i,D[5][5]])
                if D[4][4]!=0:
                   Tach4.append([i,D[4][4]])

            print("Tachyons @ -1/2:")
            for i in range(len(Tach4)):
                print(BSector[Tach4[i][0]]," ",Tach4[i][1])
            if len(Tach4)==0:
                print("None")

            print("Tachyons @ -3/8:")
            for i in range(len(Tach3)):
                print(BSector[Tach3[i][0]]," ",Tach3[i][1])
            if len(Tach3)==0:
                print("None")

            print("Tachyons @ -1/4:")
            for i in range(len(Tach2)):
                print(BSector[Tach2[i][0]]," ",Tach2[i][1])
            if len(Tach2)==0:
                print("None")

            print("Tachyons @ -1/8:")
            for i in range(len(Tach1)):
                print(BSector[Tach1[i][0]]," ",Tach1[i][1])
            if len(Tach1)==0:
                print("None")


        NTach = len(Tach1) + len(Tach2) + len(Tach3) + len(Tach4)
        print("--- "+str(NTach)+" Tachyonic Sectors Found ---")
        if NTach == 0:
            Ret = True
        else:
            Ret = False

        return Ret

class SectorClass(ModelClass):

    def __init__(self,basis,gso,b_sector):
        super().__init__(basis,gso)
        self.b_sector = b_sector

    def sec_convert(self):
        print(self.sectors()[0])
        print(self.b_sector)
        SecNum = np.where((self.sectors()[1]== self.b_sector).all(axis=1))[0][0]
        return self.sectors()[0][SecNum]

    def sec_find(self):
        SecNum = np.where((self.sectors()[1]== self.b_sector).all(axis=1))[0][0]
        return SecNum


    def sector_delta(self):
        SDelta = np.zeros((self.shape[0],1))
        for i in range(Sector.shape[0]):
            if Sector[i][0] == 1:
                SDelta[i] = -1
            elif Sector[i][0]  == 0:
                SDelta[i] = 1
        return SDelta

    def partition_function(self,q_order):

        TQ = QCRe[::,:32*(q_order+1)+1:] + 1j*QCIm[::,:32*(q_order+1)+1:]
        E =  QCEt[0,0:8*(q_order+1)+9]
        Eb = QCEt[1,0:8*(q_order+1)+9]

        CompDim = self.comp_dim()

        [NumSec,CBasis] = self.num_sector()

        [Sector,BSector,SectorUnRed] = self.sectors()

        if 4 not in CBasis:
            [MT1,MT2,MT3,MT4,MTb1,MTb2,MTb3,MTb4] = self.theta_matrix_real()
            TPF = np.zeros((32*(q_order+1)+1,32*(q_order+1)+1),dtype=np.complex128)
            i = self.sec_find()
            for j in range(Sector.shape[0]):
                  if MT1[i][j]==0 and MTb1[i][j]==0:
                    TPFT1 = GlobalFunctions.PolyMul(np.array([TQ[int(MT2[i][j]),::],TQ[25+int(MT3[i][j]),::],TQ[50+int(MT4[i][j]),::]]))
                    TPFT2 = GlobalFunctions.PolyMul(np.array([TQ[375+int(MTb2[i][j]),::],TQ[400+int(MTb3[i][j]),::],TQ[425+int(MTb4[i][j]),::]]))
                    TPFT = GlobalFunctions.GSOSec(Sector,BSector,SectorUnRed,self.gso,i,j,CompDim) * np.tensordot(TPFT1,TPFT2,axes=0)
                    TPF += TPFT
            QPF = np.around(np.real(TPF)/NumSec).astype(int)
            if np.sum(np.around(np.imag(TPF))) != 0:
                print("Error: Imaginary PF Does Not Vanish! ", np.sum(np.around(np.imag(TPF))))
            for i in range(QPF.shape[0]):
                for j in range(QPF.shape[1]):
                    if (i%4 != 0 or j%4 != 0) and QPF[i][j] != 0:
                      print("Error: Disallowed Terms in q-Expansion! ",(i-32),(j-32),QPF[i][j])
            QPF = QPF[0::4,0::4]
            A = np.pad(QPF,((8,0),(8,0)), mode='constant')
            B = np.tensordot(E,Eb,axes=0)
            C = np.tensordot(A,B,axes=0)
            D = np.zeros((C.shape[0],C.shape[0],2*C.shape[0]-1,2*C.shape[0]-1),dtype=int)
            for k in range(D.shape[0]):
             for l in range(D.shape[1]):
               D[k,l,k:k+D.shape[0],l:l+D.shape[1]] = C[k,l,:,:]
            D = D.sum(axis=0).sum(axis=0)[8:int(8*(q_order+1))+8+1,8:int(8*(q_order+1))+8+1]
            DOut = np.pad(D,((1,0),(1,0)), mode='constant')
            DOut[0][0] = 8
            for i in range(1,DOut.shape[0]):
              DOut[i][0] = i-(8+1)
              DOut[0][i] = i-(8+1)

        else:
            [MT1,MT2,MT3,MT4,MTb1,MTb2,MTb3,MTb4,MTb5,MTb6,MTb7,MTb8,MTb9,MTb10,MTb11,MTb12,MTb13,MTb14,MTb15,MTb16] = self.theta_matrix_complex()
            TPF = np.zeros((32*(q_order+1)+1,32*(q_order+1)+1),dtype=np.complex128)
            i = self.sec_find()
            for j in range(Sector.shape[0]):
                  if MT1[i][j]==0 and MTb1[i][j]==0:
                    TPFT1 = GlobalFunctions.PolyMul(np.array([TQ[int(MT2[i][j]),::],TQ[25+int(MT3[i][j]),::],TQ[50+int(MT4[i][j]),::]]))
                    TPFT2 = GlobalFunctions.PolyMul(np.array([TQ[375+int(MTb2[i][j]),::],TQ[400+int(MTb3[i][j]),::],TQ[425+int(MTb4[i][j]),::],TQ[450+int(MTb5[i][j]),::],
                                                TQ[475+int(MTb6[i][j]),::],TQ[500+int(MTb7[i][j]),::],TQ[525+int(MTb8[i][j]),::],TQ[550+int(MTb9[i][j]),::],
                                                TQ[575+int(MTb10[i][j]),::],TQ[600+int(MTb11[i][j]),::],TQ[625+int(MTb12[i][j]),::],TQ[650+int(MTb13[i][j]),::],
                                                TQ[675+int(MTb14[i][j]),::],TQ[700+int(MTb15[i][j]),::],TQ[725+int(MTb16[i][j]),::]]))
                    TPFT = GlobalFunctions.GSOSec(Sector,BSector,SectorUnRed,self.gso,i,j,CompDim) * np.tensordot(TPFT1,TPFT2,axes=0)
                    TPF += TPFT
            QPF = np.around(np.real(TPF)/NumSec).astype(int)
            if np.sum(np.around(np.imag(TPF))) != 0:
                print("Error: Imaginary PF Does Not Vanish! ", np.sum(np.around(np.imag(TPF))))
            for i in range(QPF.shape[0]):
                for j in range(QPF.shape[1]):
                    if (i%4 != 0 or j%4 != 0) and QPF[i][j] != 0:
                      print("Error: Disallowed Terms in q-Expansion! ",(i-32),(j-32),QPF[i][j])
            QPF = QPF[0::4,0::4]
            A = np.pad(QPF,((8,0),(8,0)), mode='constant')
            B = np.tensordot(E,Eb,axes=0)
            C = np.tensordot(A,B,axes=0)
            D = np.zeros((C.shape[0],C.shape[0],2*C.shape[0]-1,2*C.shape[0]-1),dtype=int)
            for k in range(D.shape[0]):
             for l in range(D.shape[1]):
               D[k,l,k:k+D.shape[0],l:l+D.shape[1]] = C[k,l,:,:]
            D = D.sum(axis=0).sum(axis=0)[8:int(8*(q_order+1))+8+1,8:int(8*(q_order+1))+8+1]
            DOut = np.pad(D,((1,0),(1,0)), mode='constant')
            DOut[0][0] = 8
            for i in range(1,DOut.shape[0]):
              DOut[i][0] = i-(8+1)
              DOut[0][i] = i-(8+1)

        QIntegral = QInt[0:int(8*(q_order+1))+1,0:int(8*(q_order+1))+1]
        m=0
        for i in range(D.shape[0]):
           for j in range(D.shape[0]):
            if i!=j and QIntegral[i][j]==9 and D[i][j]!=0:
                print("~ Error: Divergent Terms Partition Function!")
                print(i+1,j+1,D[i][j])
                #sys.exit()
            if i==j and i<8 and j<8 and D[i][j] != 0:
                m+=1
                CoC = 999999
        if m==0:
           CoC = 0
           for i in range(D.shape[0]):
            for j in range(D.shape[0]):
             CoC += D[i][j] * QIntegral[i][j]

        return DOut, CoC


class IndexBasisClass:

    def __init__(self,basis,s_matrix):
        IndexBasisClass.basis = basis
        IndexBasisClass.index_basis = np.matmul(s_matrix,basis)%2
        IndexBasisClass.s_matrix = s_matrix

    def num_basis(self):
        return self.basis.shape[0]

    def comp_dim(self):
        return int((self.basis.shape[1]-24)/4)

    def num_sector(self):
        return 2**self.num_basis()

    def sectors(self):
        NumBas = self.num_basis()
        NumSec =self.num_sector()
        Sector = np.zeros((NumSec,self.basis.shape[1]))
        BSector = np.zeros((NumSec,NumBas))
        rngs = np.full(self.basis.shape[0],2)
        for i,t in enumerate(itertools.product(*[range(i) for i in rngs])):
            Sector[i,:] = sum([self.index_basis[i,:] * t[i] for i in range(len(t))])
            BSector[i,:] = t
        return Sector%2, BSector.astype(int)


    def theta_matrix(self):
        Sector = self.sectors()[0]
        CompDim = self.comp_dim()
        SectorL=Sector[::,:8+CompDim*2:]
        SectorRR=Sector[::,8+CompDim*2:8+CompDim*4:]
        SectorRC=Sector[::,8+CompDim*4:8+CompDim*4+16:]
        CSectorL = 1 - SectorL
        CSectorRR = 1 - SectorRR
        CSectorRC = 1 - SectorRC
        MT1 = np.dot(SectorL, SectorL.T)/2
        MT4 = np.dot(CSectorL, SectorL.T)/2
        MT2 = np.dot(SectorL, CSectorL.T)/2
        MT3 = np.dot(CSectorL, CSectorL.T)/2
        MTb1 = np.dot(SectorRR, SectorRR.T)/2 + np.dot(SectorRC, SectorRC.T)
        MTb2 = np.dot(SectorRR, CSectorRR.T)/2 + np.dot(SectorRC, CSectorRC.T)
        MTb3 = np.dot(CSectorRR, CSectorRR.T)/2 + np.dot(CSectorRC, CSectorRC.T)
        MTb4 = np.dot(CSectorRR, SectorRR.T)/2 + np.dot(CSectorRC, SectorRC.T)
        return MT1,MT2,MT3,MT4,MTb1,MTb2,MTb3,MTb4


class IndexModelClass(IndexBasisClass):

    def __init__(self,basis,s_matrix,p_matrix,gso):
        super().__init__(basis,s_matrix)
        self.gso = gso
        self.p_matrix = p_matrix
        temp_g_matrix = np.zeros((self.gso.shape[0],self.gso.shape[0]))
        for i in range(self.gso.shape[0]):
            for j in range(self.gso.shape[0]):
                if self.gso[i][j] == -1:
                    temp_g_matrix[i,j] = 1
                if self.gso[i][j] == +1:
                    temp_g_matrix[i,j] = 0
        self.g_matrix = temp_g_matrix

    def partition_function(self,q_order):

        TQ = QCRe[::,:32*(q_order+1)+1:]
        E =  QCEt[0,0:8*(q_order+1)+9]
        Eb = QCEt[1,0:8*(q_order+1)+9]

        GPTilde = np.matmul(np.matmul(np.linalg.inv(self.s_matrix),(self.g_matrix + self.p_matrix)),np.linalg.inv(self.s_matrix.T))%2

        CompDim = self.comp_dim()

        NumSec = self.num_sector()

        [Sector,BSector] = self.sectors()

        [MT1,MT2,MT3,MT4,MTb1,MTb2,MTb3,MTb4] = self.theta_matrix()

        TPF = np.zeros((32*(q_order+1)+1,32*(q_order+1)+1))
        for i in range(Sector.shape[0]):
           for j in range(Sector.shape[0]):
              if MT1[i][j]==0 and MTb1[i][j]==0:
                Chi = (BSector[i,0]+BSector[i,1])*(BSector[j,9]+BSector[j,10]+BSector[j,9]*BSector[j,10]) + (BSector[j,0]+BSector[j,1])*(BSector[i,9]*BSector[j,10]+BSector[i,10]*BSector[j,9])
                Phase =(BSector[i,0]+BSector[j,0]+(np.matmul(np.matmul(BSector[i],GPTilde),BSector[j])) + Chi)%2
                TPFT1 = GlobalFunctions.PolyMul(np.array([TQ[int(MT2[i][j]),::],TQ[25+int(MT3[i][j]),::],TQ[50+int(MT4[i][j]),::]]))
                TPFT2 = GlobalFunctions.PolyMul(np.array([TQ[375+int(MTb2[i][j]),::],TQ[400+int(MTb3[i][j]),::],TQ[425+int(MTb4[i][j]),::]]))
                TPFT = (-1)**(Phase) * np.tensordot(TPFT1,TPFT2,axes=0)
                TPF += TPFT
        QPF = np.around(np.real(TPF)/NumSec).astype(int)
        for i in range(QPF.shape[0]):
            for j in range(QPF.shape[1]):
                if (i%4 != 0 or j%4 != 0) and QPF[i][j] != 0:
                  print("Error: Disallowed Terms in q-Expansion! ",(i-32),(j-32),QPF[i][j])
        QPF = QPF[0::4,0::4]
        A = np.pad(QPF,((8,0),(8,0)), mode='constant')
        B = np.tensordot(E,Eb,axes=0)
        C = np.tensordot(A,B,axes=0)
        D = np.zeros((C.shape[0],C.shape[0],2*C.shape[0]-1,2*C.shape[0]-1),dtype=int)
        for k in range(D.shape[0]):
         for l in range(D.shape[1]):
           D[k,l,k:k+D.shape[0],l:l+D.shape[1]] = C[k,l,:,:]
        D = D.sum(axis=0).sum(axis=0)[8:int(8*(q_order+1))+8+1,8:int(8*(q_order+1))+8+1]
        DOut = np.pad(D,((1,0),(1,0)), mode='constant')
        DOut[0][0] = 8
        for i in range(1,DOut.shape[0]):
          DOut[i][0] = i-(8+1)
          DOut[0][i] = i-(8+1)
        return DOut
