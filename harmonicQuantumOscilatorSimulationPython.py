from math import *
import numpy as np
import matplotlib.pylab as plt
import scipy as sp

"""CONSTANTS I DECLARACIÓ DE VARIABLES"""

m=1e-34
w=1e-1
pi=np.pi
dt=500
numeroiteracionst=100

t=np.arange(0,numeroiteracionst*dt,dt)

extrems=100  
elementsdelabase=100
hbarra=1.054571628e-34
a0=np.sqrt(hbarra/(m*w))

x=np.arange(-1*extrems,extrems,0.1) #DISCRETITZACIÓ DE L'ESPAI
x0=10 

"""Constrium la base del Hamiltonia"""
def vepH (x,n,a0):
    nvector=np.zeros(n+1) #el np.hermite és 0 indexed, com el python, pero el np.zeros no,per tenir en la posicio la posicio n necessitem el vector n+1 
    nvector[n]=1 #ens posa un 1 a la posicio n per a que hermite quedi com Hn
    Hermite=np.polynomial.hermite.hermval(x/a0,nvector)
    exp=np.exp(-x**2/(2*a0**2))
    normalitzacio=1/(np.sqrt(np.sqrt(pi)*(2**n)*sp.misc.factorial(n)*a0))
    return exp*Hermite*normalitzacio


#COMPROVACIO DE QUE HO FEM BE.
y0=vepH(x,0,a0)
y1=vepH(x,1,a0)

comprovacio1=sp.integrate.trapz(y0**2,x)   #si s'aproxima a 1 perfecte
comprovacio2=sp.integrate.trapz(y1**2,x)

#CREACIO DE L'ESTAT INICIAL
    
estatinicial=vepH(x-x0,0,a0)#NO USAREM L'OPERADOR TRANSLACIO, SIMPLEMENT DESPLAÇAREM LA X UN VALOR X0 CAP A LA DRETA

"""OBTENCIO DELS COEFICIENTS"""

baseH=np.zeros([len(x),elementsdelabase])

for i in range(len(x)): #cream una matriu de el nombre de x que hi ha columnes i 20 files
    for j in range(elementsdelabase):    #cada columna conté la funció propia de H harmònic, el numero de columna correspon amb el numero de l'estat
        baseH[i,j]=vepH(x[i],j,a0)
        

# multiplicar per lestat incinial PER A TENIR LA FUNCIO A INTEGRAR.
fi=np.zeros([len(x),elementsdelabase])

for j in range(elementsdelabase):
    for i in range(len(x)):
        fi[i,j]=baseH[i,j]*estatinicial[i]

#INTEGRACIO
c0=np.zeros([elementsdelabase])
for j in range(elementsdelabase):
    c0[j]=sp.integrate.simps(fi[:,j],x)
    
cquad=c0**2 #PER COMPROVAR QUE VAGI BE, MIRAR QUE LA SUMA DELS SEUS COMPONENTS NO DONI MAJOR QUE 1.
aproximat1=np.sum(cquad)

"""CONSTRUCCIO DE LA FUNCIO ORIGINAL COM A CL DE ELEMENTS DE LA BASE"""
construit=0

for i in range(elementsdelabase):    
    construit=c0[i]*vepH(x,i,a0)+construit


plt.figure(2)
plt.plot(x,construit)
plt.plot(x,estatinicial)
plt.plot(x,estatinicial-construit)
plt.show(2)
    
"""EVOLUCIO DELS COEFICIENTS"""


ct=np.zeros([len(t),elementsdelabase],dtype=complex)
E=np.zeros(elementsdelabase)

for i in range(elementsdelabase):
    E[i]=hbarra*w*(0.5+i)
       
for k in range(elementsdelabase):
    for i in range(len(t)):
        ct[i,k]=c0[k]*np.exp(complex(0,-E[k]*t[i]*1/hbarra))
                
onaambt=np.zeros([len(t),len(x)],dtype=complex)
onatreal=dict()
pdfx=dict()
integrantp=dict()
integrantp2=dict()
esperatx=np.zeros(len(t))
esperatxquad=np.zeros(len(t))  
esperatp=np.zeros(len(t))
esperatp2=np.zeros(len(t))


plt.figure(0)      
for i in range(len(t)):
    for k in range(elementsdelabase):
        onaambt[i,:]=ct[i,k]*vepH(x,k,a0)+onaambt[i,:]
    onatreal[i]=onaambt[i,:]
    pdfx[i]=(np.absolute(onaambt[i,:]))**2
    esperatx[i]=np.real(sp.integrate.simps(pdfx[i]*x,x))
    esperatxquad[i]=np.real(sp.integrate.simps(pdfx[i]*x*x,x))
    integrantp[i]=hbarra*complex(0,-1)*np.conjugate(onatreal[i])*np.gradient(onaambt[i,:],dt)
    esperatp[i]=np.real(sp.integrate.simps(integrantp[i],x))
    integrantp2[i]=-(hbarra**2)*np.conjugate(onatreal[i])*np.gradient(np.gradient(onaambt[i,:],dt),dt)
    esperatp2[i]=np.real(sp.integrate.simps(integrantp2[i],x))
    
    plt.pause(0.0001)
    plt.clf()
    plt.plot(x,np.real(onaambt[i,:]),color='blue',label='real')
    plt.plot(x,np.imag(onaambt[i,:]),color='red',label='imaginari')
    plt.plot(x,(np.absolute(onaambt[i,:])**2),color='black',label='mòdul')
    plt.xlim(-extrems/2,extrems/2)
    plt.ylim(-0.5,0.5)
    plt.legend()
    plt.show(0)
    
  
varianzax=(esperatxquad-esperatx**2)**0.5
varianzap=(esperatp2-esperatp**2)**0.5
plt.figure(1)
plt.title('esperat de x')
plt.xlabel('temps(s)')
plt.ylabel('distància(m)')
plt.grid()
plt.plot(t,esperatx)
plt.savefig('esperatx.png')
plt.show(1)

plt.figure(2)
plt.title('esperat de x cuadrat')
plt.xlabel('temps(s)')
plt.ylabel('distància(m)')
plt.grid()
plt.plot(t,esperatxquad)
plt.savefig('esperatx2.png')
plt.show(2)

plt.figure(3)
plt.title('variança x')
plt.xlabel('temps(s)')
plt.ylabel('distància(m)')
plt.grid()
plt.plot(t,varianzax)
plt.savefig('varianzax.png')
plt.show(3)

plt.figure(4)
plt.title('esperat de p')
plt.xlabel('temps(s)')
plt.ylabel('p(mKg/s)')
plt.grid()
plt.plot(t,esperatp)
plt.savefig('esperatp.png')
plt.show(4)

plt.figure(5)
plt.title('esperat de p cuadrat')
plt.xlabel('temps(s)')
plt.ylabel('p(mKg/s)')
plt.grid()
plt.plot(t,esperatp2)
plt.savefig('esperatp2.png')
plt.show(5)

plt.figure(6)
plt.title('variança p')
plt.xlabel('temps(s)')
plt.ylabel('p(mKg/s)')
plt.grid()
plt.plot(t,varianzap)
plt.savefig('varianzap.png')
plt.show(6)

plt.figure(7)
plt.title('multiplicacio de variançes')
plt.xlabel('temps(s)')
plt.ylabel('variança (hbarra)')
plt.grid()
plt.plot(t,varianzap*varianzax*hbarra)
plt.savefig('multiplicaciovariances.png')
plt.show(7)
