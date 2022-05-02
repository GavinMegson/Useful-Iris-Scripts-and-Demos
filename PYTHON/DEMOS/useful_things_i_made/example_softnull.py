import numpy as np
from scipy import linalg as sp

# Indicates to use random example instead of manually entered values
use_rand = True

# Number of tx, rx antennas
M_tx = 9
M_rx = 9

# populate CSI values in a disgusting manual way here
if use_rand == False:
    H = [[ 0,0,0,0,0,0,0,0,0 ],
         [ 0,0,0,0,0,0,0,0,0 ],
         [ 0,0,0,0,0,0,0,0,0 ],
         [ 0,0,0,0,0,0,0,0,0 ],
         [ 0,0,0,0,0,0,0,0,0 ],
         [ 0,0,0,0,0,0,0,0,0 ],
         [ 0,0,0,0,0,0,0,0,0 ],
         [ 0,0,0,0,0,0,0,0,0 ],
         [ 0,0,0,0,0,0,0,0,0 ]]
    H = np.array(H)


# effective antennas
D_tx = 9

# signal to send without softnull (all 1s for this example)
#X_down = 0j + np.ones(M_tx)
X_down = 0j + np.ones(M_tx)
#    * D_tx/M_tx # keeping total power output even over number of antennas
# doing this a better way below

# MUMIMO precoder (sent to softnull for transmission) (all 1s for this example)
P_down = 0j + np.ones(D_tx)

# If using example random CSIs:
# vaguely random for example
if use_rand == True:
    amp = np.random.rand(M_tx,M_rx)
    phase = np.random.rand(M_tx,M_rx) * 2 * np.pi
    H = np.multiply(amp, np.exp(1j*phase))

# SVD
U, E, V_H = np.linalg.svd(H,full_matrices=False)

# make E into matrix (by default, numpy makes it an ordered list of eigenvalues) of first D_tx values
sig = sp.diagsvd(E,M_tx,M_rx)
for i in range(0, M_tx - D_tx):
    sig[i][i] = 0

print("U, sig, V_H:\n",U, "\n", sig,"\n",V_H)

# Reconstruct H without top D_tx dimensions from sig
P_new = np.matmul(U, np.matmul(sig, V_H))


print("SUM POWER OF previous:\n", np.sum(np.multiply(H,H.conjugate())))
print("SUM POWER OF previous but calculated the right way:\n", np.sum(np.multiply(np.sum(H,0),np.sum(H,0).conjugate())))
print("SUM POWER OF P_NEW:\n", np.sum(np.multiply(P_new,P_new.conjugate())))
afterSoft = np.sum(np.multiply(np.sum(P_new,0),np.sum(P_new,0).conjugate()))
print("SUM POWER OF P_NEW but calculated the right way:\n", afterSoft)
afterSoftPower = np.sum(np.multiply(np.sum(P_new * (M_tx/D_tx),0),np.sum(P_new * (M_tx/D_tx),0).conjugate()))
print("SUM POWER OF P_NEW but calculated the right way and also adjusted for power:\n", afterSoftPower)

# rescale for equivalent power output across all antennas, before and after
#H_new *= M_tx/D_tx

print("P_new:\n", P_new)



# get top D vectors (last D_tx columns of V)
P_self = np.delete(V_H.conj().T,slice(0,M_tx - D_tx),1)

# signals sent, after combining precoder used above softnull and softnull
# x_down = P_self * P_down * symbols_to_send
# M_tx x K_down (assume M_tx x 1)

# recast into original dimensions (onto physical antennas)

P_down.shape = (-1,1) # precoder (all 1s for example)
x_down = np.matmul(P_self,P_down) # assume symbols all 1

x_down.shape = (-1,1)

print("x_down:\n", x_down)
#print("alt x_down:\n", np.matmul(P_new,P_down))
'''

# actual signals to send with softnull:
x_down = np.matmul(P_new,P_down)

print("x down:\n", x_down)
'''
# calculate received signal without softnull
Rx_pre = 0j + np.zeros(M_rx)
for t in range(M_tx):
	for r in range(M_rx):
#		Rx_pre[r] += X_down[t]*H[t][r]
		Rx_pre[r] += H[t][r]

# received signal with softnull
Rx_post = 0j + np.zeros(M_rx)
for t in range(M_tx):
	for r in range(M_rx):
		Rx_post[r] += x_down[t]*H[t][r]
#for r in range(M_rx):
#    Rx_post[r] = (Rx_post[r])**2

print("Rx_pre sums:\n", Rx_pre[0], Rx_pre[1],Rx_pre[2],Rx_pre[3])
print("H summed axis 0:\n", np.sum(H,0))

print("H")
print(H)
print("E")
print(E)
print("softnull")
print(P_self)
print("V.H")
print(V_H.conj().T)

print("[Rx at each antenna], sum power squared")
print("No Softnull:")
print(sum(np.multiply(Rx_pre,Rx_pre.conjugate())))
print("With Softnull:")
print(sum(np.multiply(Rx_post,Rx_post.conjugate())))

print("With Softnull, but actually calculated properly:")
print(afterSoftPower)
print("unadjusted, for debugging")
print(afterSoft)
