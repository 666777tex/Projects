import numpy as np

# clean version of the main jupyter notebook file 

# Creating a generalized network to work with any amount oflayer

L = int(input("Layers: "))

n = []
n.append(int(input("Nodes in Input Layer:")))
for i in range(1, L):
    n.append(int(input("Nodes in Hidden Layer" + str(i) + ":")))
n.append(int(input("Nodes in Output Layer:")))

WBs = {}
for i in range(1, L+1):
    WBs["W" + str(i)] = np.random.randn(int(n[i]), int(n[i-1]))
    WBs["b" + str(i)] = np.random.randn(int(n[i]), 1)

def standard_scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std
def prepare_data():
    X = np.array([
        [150, 70],
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69]
    ])
    X = standard_scale(X)
    y = np.array([0,1,1,0,0,1,1,0,1,0])
    m = 10
    A0 = X.T
    Y = y.reshape(n[L], m)

    return A0, Y, m

def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))

ZAs = {}
def feed_forward(A0):
    ZAs["A0"] = A0
    for i in range(1, L+1):
        ZAs["Z" + str(i)] = WBs["W" + str(i)] @ ZAs["A" + str(i-1)] + WBs["b" + str(i)] 
        ZAs["A" + str(i)] = sigmoid(ZAs["Z" + str(i)])

    y_hat = ZAs["A" + str(L)]
    
    return y_hat, ZAs

def cost(y_hat, y):
  """
  y_hat should be a n^L x m matrix
  y should be a n^L x m matrix
  """
  # 1. losses is a n^L x m
  losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )

  m = y_hat.reshape(-1).shape[0]

  # 2. summing across axis = 1 means we sum across rows, 
  #   making this a n^L x 1 matrix
  summed_losses = (1 / m) * np.sum(losses, axis=1)

  # 3. unnecessary, but useful if working with more than one node
  #   in output layer
  return np.sum(summed_losses)

def backprop_layer_output(y_hat, Y, m, AL_, WL, L = L):
    AL = y_hat

    # step 1. calculate dC/dZL
    dC_dZL = (1/m) * (AL - Y)
    assert dC_dZL.shape == (n[L], m)

    # step 2. calculate dC/dWL = dC/dZL * dZL/dWL
    dZL_dWL = AL_
    assert dZL_dWL.shape == (n[L-1], m)

    dC_dWL = dC_dZL @ dZL_dWL.T
    assert dC_dWL.shape == (n[L], n[L-1])

    # step 3. calculate dC/dbL = np.sum(dC/dZL, axis=1, keepdism=True)
    dC_dbL = np.sum(dC_dZL, axis=1, keepdims=True)
    assert dC_dbL.shape == (n[L], 1)

    # step 4. calculate propagator dC/dAL_ = dC/dZL * dZ3/dAL_
    dZ3_dAL_ = WL
    dC_dAL_ = WL.T @ dC_dZL
    assert dC_dAL_.shape == (n[L-1], m)

    return dC_dWL, dC_dbL, dC_dAL_

def backprop_hidden_layers(prop, AL, AL_, WL, L, m):
    dAL_dZL = AL * (1 - AL)
    dC_dZL = prop * dAL_dZL
    assert dC_dZL.shape == (n[L], m)

    dZL_dWL = AL_
    assert dZL_dWL.shape == (n[L-1], m)

    dC_dWL = dC_dZL @ dZL_dWL.T
    assert dC_dWL.shape == (n[L], n[L-1])

    dC_dbL = np.sum(dC_dWL, axis=1, keepdims=True)
    assert dC_dbL.shape == (n[L], 1)

    if L > 1:
        dZL_dAL_ = WL
        dC_dAL_ = WL.T @ dC_dZL
        assert dC_dAL_.shape == (n[L], m)
        return dC_dWL, dC_dbL, dC_dAL_
    
    return dC_dWL, dC_dbL

def train():
    global W3, W2, W1, b1, b2, b3

    epochs = 1000
    alpha = 0.1
    costs = []

    for e in range(epochs):

        A0, Y, m = prepare_data()

        # 1. feed forward
        y_hat, ZAs = feed_forward(A0)

        # 2. cost calculation
        error = cost(y_hat, Y)
        costs.append(error)
        
        layer = L
        partial = {}
        #output layer calculation
        dW, db, dA = backprop_layer_output(
            y_hat, Y, m, AL_ = ZAs["A" + str(layer-1)], WL = WBs["W"+str(layer)], L = layer)
        partial["dC_dW" + str(layer)] = dW 
        partial["dC_db"+str(layer)] = db
        partial["dC_dA" + str(layer-1)] = dA
        #updating weight and bias for output layer 
        WBs["W"+str(layer)] = WBs["W"+str(layer)] - (alpha * partial["dC_dW" +str(layer)])
        WBs["b" +str(layer)] = WBs["b" +str(layer)] - (alpha * partial["dC_db" +str(layer)])
        layer -= 1

        # other hidden layer calculation and updating weights and biases
        while layer > 1:
            dW, db, dA = backprop_hidden_layers(
                partial["dC_dA" + str(layer)], AL = ZAs["A"+str(layer)], AL_ = ZAs["A" + str(layer - 1)], WL = WBs["W" + str(layer)], L =layer, m = m)
            partial["dC_dW" + str(layer)] = dW 
            partial["dC_db"+str(layer)] = db
            partial["dC_dA" + str(layer-1)] = dA
            WBs["W"+ str(layer)] = WBs["W"+ str(layer)] - (alpha * partial["dC_dW" + str(layer)])
            WBs["b" +str(layer)] = WBs["b" +str(layer)] - (alpha * partial["dC_db" +str(layer)])
            layer -= 1

        # last prop layer (first hidden layer)
        dW, db = backprop_hidden_layers(
                partial["dC_dA" + str(layer)], AL = ZAs["A"+str(layer)], AL_ = ZAs["A" + str(layer - 1)], WL = WBs["W" + str(layer)], L =layer, m = m)
        partial["dC_dW" + str(layer)] = dW 
        partial["dC_db"+str(layer)] = db
        WBs["W"+ str(layer)] = WBs["W"+ str(layer)] - (alpha * partial["dC_dW" + str(layer)])
        WBs["b" +str(layer)] = WBs["b" +str(layer)] - (alpha * partial["dC_db" +str(layer)])

        if e % 20 == 0:
            print(f"epoch {e}: cost = {error:4f}")

        return costs
    
costs = train() 