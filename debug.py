#%% Decode message
m = 31
FILE_NAME = "key.csv"
K = pd.read_csv(f"/Users/kaisoon/Google Drive/Code/Python/COMP90043_CaS/data/{FILE_NAME}")
K_inv = modMatInv(K, m).astype(int)

c = '!LPUMYAIJ?.MPA.DVRFUTNRUZYEFM?QVKJTOBTDRIAN!?SLQBKESZOSFRAAWYPI.VBOLLMWAWEMQ.JYBOITGNJRIFYGEGIBC?RB?UN?MORI,'.lower()
assert len(c) == 108

# Convert all characters to its corresponding integer
c_i = []
for char in c:
    c_i.append(alpha.find(char))
# Split integers into groups of 4
C = [c_i[i:i+BLOCK_SIZE] for i in range(0, len(c_i), BLOCK_SIZE)]
C = np.array(C).transpose().astype(int)

# Decryption
P = np.matmul(K_inv, C)%m

# Translate decoded message
p = ''
P = P.transpose()
for row in P:
    for cell in row:
        p = p + alpha[cell]

print(p)