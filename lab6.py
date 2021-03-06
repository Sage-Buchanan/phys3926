import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt


def part2(k, L, Lw):
    k1 = k[0]
    k2 = k[1]
    k3 = k[2]
    k4 = k[3]

    l1 = L[0]
    l2 = L[1]
    l3 = L[2]
    l4 = L[3]
    
    K = np.array([[-k1 - k2, k2, 0],
                [k2 , -k2 - k3, k3],
                [0, k3, -k3 - k4]])
    
    b = np.array([-k1*l1 + k2*l2, -k2*l2 + k3*l3, -k3*l3 + k4*(l4 - Lw)])

    return(K, b)

## Part 3
# (i)
print("Part 3, i")
k1 = np.array([1,2,3,4])
l1 = np.array([1,1,1,1])
lw1 = 10
K1, b1 = part2(k1, l1, lw1)
if (np.linalg.det(K1) != 0):
    sol1 = np.linalg.solve(K1,b1)
    print("Equilibirum is at {:.2f}, {:.2f}, {:.2f}.".format(sol1[0], sol1[1], sol1[2]))
else:
    print("Uh oh, singular K matrix \nNo equilibirum exists")



# (ii)
print("\nPart 3, ii")
k2 = np.array([0,1,1,0])
l2 = np.array([2,2,1,1])
lw2 = 4
K2, b2 = part2(k2, l2, lw2)
if (np.linalg.det(K2) != 0):
    sol2 = np.linalg.solve(K2,b2)
    print("Equilibirum is at {:.2f}, {:.2f}, {:.2f}.".format(sol2[0], sol2[1], sol2[2]))
else:
    print("Uh oh, singular K matrix \nNo equilibirum exists")



# Part 4
def diff(t, x, m, k, L, Lw):
    K, b = part2(k, L , Lw)
    F = np.dot(K,x[0:3]) - b
    return(np.append(x[3:6], F/m))


results = sp.solve_ivp(diff, (0,10), np.array([7, 6.32, 8.28, 0, 0, 0]), args = ( np.array([1,1,1]), np.array([1,2,3,4]), np.array([1,1,1,1]), 10,  )  )
plt.plot(results.t, results.y[0])
plt.plot(results.t, results.y[1])
plt.plot(results.t, results.y[2])
plt.show()