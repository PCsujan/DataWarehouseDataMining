import numpy as np

# ---------- Sigmoid Activation and Derivative ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ---------- Input and Output ----------
X = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
])

# Convert targets from [-1, 1] to [0, 1] for sigmoid
y = np.array([[-1], [1], [1], [-1]])
y = (y + 1) / 2  

# ---------- Initialize Weights ----------
np.random.seed(1)
input_size = 2
hidden_size = 2
output_size = 1
lr = 0.1  

# Weights
W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
b1 = np.zeros((1, hidden_size))
W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
b2 = np.zeros((1, output_size))

# ---------- Training Loop ----------
epochs = 10000
for epoch in range(epochs):
    # Forward Pass
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Backward Pass
    error = y - A2
    dA2 = error * sigmoid_derivative(A2)

    dW2 = np.dot(A1.T, dA2)
    db2 = np.sum(dA2, axis=0, keepdims=True)

    dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dA1)
    db1 = np.sum(dA1, axis=0, keepdims=True)

    # Update weights
    W2 += lr * dW2
    b2 += lr * db2
    W1 += lr * dW1
    b1 += lr * db1

    # Optional: Print loss occasionally
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} Loss: {loss:.4f}")

# ---------- Prediction ----------
A2_final = sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2)
predicted = (A2_final > 0.5).astype(int)

# Convert back from [0,1] to [-1,1]
predicted = predicted * 2 - 1
print("\nPredicted Outputs:")
for i in range(len(X)):
    print(f"Input: {X[i]} => Prediction: {predicted[i][0]}")
