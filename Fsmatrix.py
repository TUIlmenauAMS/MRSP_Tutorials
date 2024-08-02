#Example for the extraction of the Fs Matrix from the MDCT polyphase matrix
from sympy import *
from numpy import *

z=symbols('z')
N=4

#baseband prototype filter g(n):
g=symbols('g:8');
print( "g=")
print(g)

#MDCT Polyphase matrix G. each column contains the impulse response, 

G=Matrix(zeros((N,N)));
for n in range(0,N):
   for k in range(0,N):
      G[k,n]=g[n]*2/N*cos(pi/N*(n-N/2+0.5)*(k+0.5))+z**(-1) *g[N+n]*2/N*cos(pi/N*(N+n-N/2+0.5)*(k+0.5)) 

#Transform matrix T:
T=Matrix(zeros((N,N)));
for n in range(0,N):
   for k in range(0,N):
      T[n,k]=cos(pi/N*(n+0.5)*(k+0.5));

#Compute the sparse Fa matrix:
Fs= T*G

#Print the Fa matrix with 1 digit after the decimal point and replacement of very small number by 0:
print( "Fs=")
print( Fs.evalf(2,chop=True))
#print(sympify(Fs))
#pprint(Fs)
