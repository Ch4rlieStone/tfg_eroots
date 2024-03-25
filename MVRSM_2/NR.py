    #Vslack es un vector de dimension 1, Vpv es de dimensión nG, por tanto Vtot
    #es de dimensión nD+nG+1. Solo iteraremos los nD, los demás son datos.


    #Numeración de los buses: El primero es el slack. Tecnicamente el 0, en
    #matlab es el 1 pero lo sorteamos con una matriz aparte.
    #Luego los PQ (tienen tanto activa como reactiva scheduled, hay nD)
    #Luego los PV (solo tienen activa sch, hay nG)
    #La matriz de impedancias sigue la misma númeración (f1c1 es el slack, f2...nD+1c2...nD+1 son los PQ y los restantes los pv)

import numpy as np
import math

def NR(Y, Psch, Qsch, nD, nG, Vslack, Vpv):

    eps = 1
    V = np.ones((nD, 1)) #para iterar las reactivas de los PQ. Son las unicas tensiones incognitas
    
    if nG==0: #POR ALGUN MOTIVO HAY QUE DISTINGUIR ENTRE ESTOS DOS CASOS, OTHERWISE PYTHON DECIDE SUICIDARSE
        Vtot = np.concatenate((Vslack, V),)
        Vact = V #para iterar todas las potencias activas de los PQ y Pv. Las tensiones PV son dato.
    else:
        Vtot = np.concatenate((Vslack, V, Vpv)) #para iterar todas las tensiones.
        Vact = np.concatenate((V, Vpv)) #para iterar todas las potencias activas de los PQ y Pv. Las tensiones PV son dato.
        
    Th = np.zeros((nD + nG, 1)) #todos los desfases de los buses PQ y PV son incognitas
    Thtot = np.concatenate(([[0]],Th)) #incluimos el angulo del slack que siempre es 0
    PQsch = np.concatenate((Psch, Qsch))
    k=0
    #iteramos hasta que el delta sea menor a la precisión especificada
    while eps > 0.0000001:
        
        #añadimos una iteración y reseteamos las potencias para recalcularlas
        #con los datos de la última iteración.
        k=k+1
        P = np.zeros((nD + nG, 1))
        Qaux = np.zeros((nD + nG, 1))
        Thtot = np.concatenate(([[0]],Th))
        if nG==0:
            Vtot = np.concatenate((Vslack, V),)
            Vact = V #para iterar todas las potencias activas de los PQ y Pv. Las tensiones PV son dato.
        else:
            Vtot = np.concatenate((Vslack, V, Vpv)) #para iterar todas las tensiones.
            Vact = np.concatenate((V, Vpv)) #para iterar todas las potencias activas de los PQ y Pv. Las tensiones PV son dato.
        
        print(Vtot)
        print(Vact)
        #Calculamos las potencias activas en la iteración actual
        for i in range(0,(nD + nG)):
            for j in range(0,(nD+nG+1)):
                P[i] = P[i]+ Vtot[j]*(Y[i+1][j].real*math.cos(Thtot[i+1] - Thtot[j])+Y[i+1][j].imag*math.sin(Thtot[i+1] - Thtot[j]))
                
                #en la indexacion, hacemos i+1 ya que entre los Thtot tenemos
                #el slack mientras que para P no lo tenemos en cuenta. 
            
            P[i] = P[i]*Vact[i]
            
        
        #de la misma forma, pero esta vez cambiando la indexación y la matriz
        #de tensiones usada, calculamos la reactiva.
        
        
        
        for i in range(0,(nD + nG)):
            for j in range(0,(nD + nG + 1)):
                Qaux[i] = Qaux[i]+ Vtot[j]*(Y[i+1][j].real*math.sin(Thtot[i+1] - Thtot[j])-Y[i+1][j].imag*math.cos(Thtot[i+1] - Thtot[j]))
                #en la indexacion, hacemos i+1 ya que entre los Thtot tenemos
                #el slack mientras que para P no lo tenemos en cuenta. 
            
            Qaux[i] = Qaux[i]*Vact[i]
            
        
        Q = Qaux[0:nD]
       
        #Ahora hay que calcular la diferencia entre los scheduled y las
        #obtenidas en la iteración
        
        PQ = np.concatenate((P, Q))
        
        deltaPQ = PQsch - PQ

       
        #ahora solo falta la jacobiana y aplicar el NR
        
        H = np.zeros((nD + nG, nD + nG))
        N = np.zeros((nD + nG, nD))
        M = np.zeros((nD, nD + nG))
        L = np.zeros((nD, nD))
        
        
        #H
        for i in range(0,(nD + nG)):
            for j in range(0,(nD + nG)):
                if i==j:
                    H[i][i] = -Qaux[i]-Y[i+1][i+1].imag*(Vact[i]**2)
                else:
                    H[i][j] = Vact[i]*Vact[j]*(Y[i+1][j+1].real*math.sin(Th[i]-Th[j])-Y[i+1][j+1].imag*math.cos(Th[i]-Th[j]))

                
        
        
        #N
        for i in range(0,(nD + nG)):
            for j in range(0,nD):
                if i==j:
                    N[i][i] = P[i]+Y[i+1][i+1].real*(Vact[i]**2)
                else:
                    N[i][j] = Vact[i]*Vact[j]*(Y[i+1][j+1].real*math.cos(Th[i]-Th[j])+Y[i+1][j+1].imag*math.sin(Th[i]-Th[j]))

                
             
        
        #M
        for i in range(0,nD):
            for j in range(0,(nD + nG)):
                if i==j:
                    M[i][i] = P[i]-Y[i+1][i+1].real*(Vact[i]**2)
                else:
                    M[i][j] = -Vact[i]*Vact[j]*(Y[i+1][j+1].real*math.cos(Th[i]-Th[j])+Y[i+1][j+1].imag*math.sin(Th[i]-Th[j]))


                
        
        
        #L
        for i in range(0,nD):
            for j in range(0,nD): 
                if i==j:
                    L[i][i] = Qaux[i]-Y[i+1][i+1].imag*(Vact[i]**2)
                else:
                    L[i][j] = Vact[i]*Vact[j]*(Y[i+1][j+1].real*math.sin(Th[i]-Th[j])-Y[i+1][j+1].imag*math.cos(Th[i]-Th[j]))


               
        
        #construimos la jacobiana partiendo de las submatrices
        J = np.concatenate((np.concatenate((H, N), axis=1), np.concatenate((M, L), axis=1)))
       
        
        #Solo queda resolver el sistema matricial deltaPQ = J*delta(TH|deltaV/V)
        
        delta = np.dot((np.linalg.inv(J)),deltaPQ)
        
        deltaTh = delta[0:(nD + nG)];
        deltaV = np.multiply(delta[(nD + nG):(2*nD + nG)],V)
        
        Th = Th + deltaTh;
        V = V + deltaV;
        
        #k, Th, V;
       
        
        
        eps = float(max(abs(deltaPQ)));

        if nG==0:
            Vtot = np.concatenate((Vslack, V),)
  
        else:
            Vtot = np.concatenate((Vslack, V, Vpv))
            
        Thtot = np.concatenate(([[0]],Th))
        
    return Vtot,Thtot

    
    
    
    
