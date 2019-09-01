# %% Cracking the Hill Cipher
#%% Key space
# Size of key matrix
m = 4
# Modulus
p = 31

keySpace = 1
for i in range(1, m+1):
    keySpace = keySpace * (p**m - p**(m-i))
print(f"{keySpace} possible keys")


# %% Cracking the cipher
# Imports and functions
import numpy as np
import pandas as pd

def modMatInv(A, m):
    """Returns the inverse A mod m where A is a matrix"""
    A = np.array(A)

    # Deteminant of A
    A_det = int(round(np.linalg.det(A)))
    # Adjunct matric of A
    A_adj = adj(A, m)

    return modInv(A_det, m) * A_adj % m

def adj(A, m):
    n = len(A)
    A_adj = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            minor_mat = get_minor(A, j, i)
            minor_det = int(round(np.linalg.det(minor_mat)))
            A_adj[i][j] = ((-1) ** (i + j) * minor_det) % m
    return A_adj

def modInv(a, m):
    """Returns the inverse of a mod m"""
    for i in range(1, m):
        if (i*a) % m == 1:
            return i
    raise ValueError(str(a) +" has no inverse mod " + str(m))

def get_minor(A, i, j) :  # Return matrix A with the ith row and jth column deleted
    n = len(A)
    A = np.array(A)
    minor = np.zeros([n - 1, n - 1])
    p = 0
    for s in range(len(minor)) :
        if p == i :
            p = p + 1
        q = 0
        for t in range(len(minor)) :
            if q == j : q = q + 1
            minor[s][t] = A[p][q]
            q = q + 1
        p = p + 1
    return minor


# %% Prep Matrices
# A = pd.Series('a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|,|.|?|!| '.split('|'))
alpha = 'abcdefghijklmnopqrstuvwxyz,.?! '
BLOCK_SIZE = 4

p = 'ctrl caps home pgup'.split()
c = 'hgpp hofl ttsi dacr'.split()

P = np.zeros([BLOCK_SIZE, BLOCK_SIZE]).astype(int)
P.fill(np.nan)
for row_i in range(BLOCK_SIZE) :
    for col_i in range(BLOCK_SIZE) :
        P[row_i, col_i] = alpha.find(p[row_i][col_i])
P = P.transpose()

C = np.zeros([BLOCK_SIZE, BLOCK_SIZE]).astype(int)
C.fill(np.nan)
for i in range(BLOCK_SIZE) :
    for j in range(BLOCK_SIZE) :
        C[i, j] = alpha.find(c[i][j])
C = C.transpose()

for i in range(BLOCK_SIZE) :
    print(f"{p[i]} ={P[:, i]}\t\t{c[i]} = {C[:, i]}")


#%% Extra Computation
m = 31

P_det = int(np.linalg.det(P).round())
print(f"Modulo Inverse = {modInv(P_det, m)}\n")

P_adj = adj(P, m).astype(int)
print(f"Adjunct Matrix =\n{P_adj}\n")

P_inv = modMatInv(P, m).astype(int)
print(f"Modulo Inverse Matrix =\n{P_inv}\n")


# %% Compute key matrix
m = 31
FILE_NAME = "key.csv"

#  CANNOT COMPUTE INVERSE LIKE THAT!
# TODO: Need to compute determinant with modulo: D = det(P)mod31

K = np.matmul(C, modMatInv(P, m)) % m
K = K.astype(int)
K_inv = modMatInv(K, m).astype(int)
print(f"Key Matrix =\n{K}\n")
print(f"Inverse Key Matrix =\n{K_inv}\n")

# Save key
pd.DataFrame(K).to_csv(
    f"/Users/kaisoon/Google Drive/Code/Python/COMP90043_Ass1/data/{FILE_NAME}",
    index=False)
pd.DataFrame(K_inv).to_csv(
    f"/Users/kaisoon/Google Drive/Code/Python/COMP90043_Ass1/data/inv_{FILE_NAME}",
    index=False)


# %% Check if the key works in encryption
FILE_NAME = "key.csv"
K = pd.read_csv(f"/Users/kaisoon/Google Drive/Code/Python/COMP90043_Ass1/data/{FILE_NAME}")

CC = np.matmul(K, P) % m

if np.array_equal(CC, C) :
    print("Key worked in encryption!!")
else :
    print("Key does not work in encryption...")


# %% Check if the key works in decryption
FILE_NAME = "key.csv"
K_inv = pd.read_csv(f"/Users/kaisoon/Google Drive/Code/Python/COMP90043_Ass1/data/inv_{FILE_NAME}")

PP = np.matmul(K_inv, C) % m

if np.array_equal(PP, P) :
    print("Key worked in decryption!!")
else :
    print("Key does not work in decryption...")



# %% Decode message
m = 31
FILE_NAME = "key.csv"
K = pd.read_csv(f"/Users/kaisoon/Google Drive/Code/Python/COMP90043_Ass1/data/{FILE_NAME}")
K_inv = pd.read_csv(f"/Users/kaisoon/Google Drive/Code/Python/COMP90043_Ass1/data/inv_{FILE_NAME}")

c = '!LPUMYAIJ?.MPA.DVRFUTNRUZYEFM?QVKJTOBTDRIAN!?SLQBKESZOSFRAAWYPI.VBOLLMWAWEMQ.JYBOITGNJRIFYGEGIBC?RB?UN?MORI,'.lower()
assert len(c) == 108

# Convert all characters to its corresponding integer
c_i = []
for char in c :
    c_i.append(alpha.find(char))
# Split integers into groups of 4
C = [c_i[i:i + BLOCK_SIZE] for i in range(0, len(c_i), BLOCK_SIZE)]
C = np.array(C).transpose().astype(int)

# Decryption
P = np.matmul(K_inv, C) % m
P = pd.DataFrame(P)
P.to_csv("/Users/kaisoon/Google Drive/Code/Python/COMP90043_Ass1/data/plainText_decrypted_i.csv", index=False)
# Plaintext Matrix
P_alpha = pd.DataFrame(np.zeros(C.shape)).astype(int)
for i in range(P_alpha.shape[1]):
    for j in range(P_alpha.shape[0]):
        # print(alpha.find(C_mat[i][j]))
        P_alpha[i][j] = alpha[P[i][j]]

P_alpha.to_csv("/Users/kaisoon/Google Drive/Code/Python/COMP90043_Ass1/data/plainText_decrypted.csv", index=False)

# Translate decoded message
p = ''
P = P.transpose()
for index, row in P.iterrows():
    for cell in row :
        p = p + alpha[cell]

print(p)


#%% Visualise ciphertext matrix
c_split = [c[i:i+BLOCK_SIZE] for i in range(0, len(c), BLOCK_SIZE)]

# Ciphertext Matrix
C_mat = []
for group in c_split:
    col = []
    for i in range(4):
        col.append(group[i])
    C_mat.append(col)
C_mat = pd.DataFrame(C_mat).transpose()

C_mat.to_csv("/Users/kaisoon/Google Drive/Code/Python/COMP90043_Ass1/data/cipherText.csv", index=False)

# Ciphertext Matrix in integers
C_mat_i = pd.DataFrame(np.zeros(C_mat.shape)).astype(int)
for i in range(C_mat_i.shape[1]):
    for j in range(C_mat_i.shape[0]):
        # print(alpha.find(C_mat[i][j]))
        C_mat_i[i][j] = alpha.find(C_mat[i][j])

C_mat_i.to_csv("/Users/kaisoon/Google Drive/Code/Python/COMP90043_Ass1/data/cipherText_i.csv", index=False)

