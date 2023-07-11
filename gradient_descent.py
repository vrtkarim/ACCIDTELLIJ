import numpy as np
import matplotlib.pyplot as plt

sattendu = np.array([0.7 for i in range (101)])
e = np.array([1.4 for i in range (101)])
p = np.array(np.arange(0,1.01,0.01))
s=e*p
erreur = (sattendu-s)**2

#Δpoid=𝛼.2.E.(S−Sortie attendu)
def deltap(sortie):
    #𝛼=0.1
    return 0.1 * 2 * 1.4 * (sortie - sortieattendu)
poid = 0
entree = 1.4
sortie=entree*poid
sortieattendu = 0.7
poids=[poid]
cout = [(sortieattendu-entree*poid)**2]

for i in range(101):
    poid = poid -deltap(sortie)
    poids.append(poid)
    cout.append((sortieattendu-entree*poid)**2)
    sortie=entree*poid
zero = cout.index(min(cout))
#la courbe du poid convenable
plt.plot(p, erreur, 'r--', label='cout')
for i in range(zero):
    plt.plot([poids[i], poids[i+1]],[cout[i], cout[i+1]], 'g-')
    plt.plot(poids[i], cout[i], 'go')
plt.plot(poids[zero],cout[zero] , 'bo',label='poid convenable')
plt.xlabel('$poid$', fontsize=12)
plt.ylabel('$erreur$', fontsize=12)
plt.title('cout en fonction du poid')
plt.legend()
plt.show()
