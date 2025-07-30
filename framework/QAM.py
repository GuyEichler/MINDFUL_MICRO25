from sympy import *
from scipy import special
import math as mt
#import sympy as sp
#x, y, z = symbols('x y z')

def QAM(n, efficiency, ber, path_loss, margin):
  p, M, N, x = symbols('p M N x')

  init_printing(use_unicode=True)

  # solve p = 2/log2(M)*(1-1/sqrt(M))*erfc(sqrt(3*log2(M)/(2*(M-1))*x/N)) for x

  print("n is", n)

  #Equation = Eq(x**2 - x, y)
  #n = 2

  p = ber
  M = 2**n
  N = 4.29*10**-21
  #guess = 4*10**-20
  #path_loss = 6 #60dB
  #margin = 2 #20dB
  efficiency = efficiency / n#%
  #val = erfcinv(1/500000)**2

  #val = special.erfcinv(1/500000)
  #val2 = special.erfc(mt.sqrt(3*mt.log2(M)/(2*(M-1))*x/N))
  #val3 = (p*mt.log2(M) - (p*mt.log2(M))/(1 - M) - (mt.sqrt(M)*p*mt.log2(M))/(1 - M))/(2*mt.log2(2))
  val2 = special.erfcinv((p*mt.log2(M) - (p*mt.log2(M))/(1 - M) - (mt.sqrt(M)*p*mt.log2(M))/(1 - M))/(2*mt.log2(2)))

  #print(val)
  print(val2)
  #print(val3)

  #Equation = Eq(2/log(M,2)*(1-1/sqrt(M))*erfc(sqrt(3*log(M,2)/(2*(M-1))*x/N)) , p)
  #Equation = Eq(2/log(M,2)*(1-1/sqrt(M))*sqrt(3*log(M,2)/(2*(M-1))*x/N) , p)
  # calc = erfc(x)
  # Equation = Eq(calc, p)
  # (-1 + M) N erfc^(-1)((p log(M) - (p log(M))/(1 - M) - (sqrt(M) p log(M))/(1 - M))/(2 log(2)))^2 log(4))/(3 log(M))
  #Equation = Eq(((-1 + M)*N*erfcinv((p*log(M,2) - (p*log(M,2))/(1 - M) - (sqrt(M)*p*log(M,2))/(1 - M))/(2*log(2,2)))**2*log(4,2))/(3*log(M,2)) , x)
  Equation = Eq(((-1 + M)*N*val2**2*log(4,2))/(3*log(M,2)) , x)

  print("Equation is:", Equation)
  results = solve(Equation, x)
  #results = sp.solve(calc, x)
  #results = nsolve(Equation, x, 0)
  numeric_res = list()

  for res in results:
    numeric_res.append(res.evalf())

  print(results)
  print(numeric_res)

  add_path_loss = numeric_res[0] * 10**path_loss
  add_margin = add_path_loss * 10**margin
  add_efficiency = add_margin * 100 / efficiency

  final = add_efficiency
  print("Final Eb is:", final/10**-12, "pJ")

  return final
