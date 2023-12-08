
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.neural_networks import CircuitQNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def compute_loss_and_gradients(params, batch_data, batch_labels, qnn):
    # Forward pass: compute the predictions
    predictions = qnn.forward(batch_data, params)

    # Compute the loss (e.g., binary cross-entropy)
    loss = binary_cross_entropy_loss(predictions, batch_labels)

    # Compute the gradients
    grad = Gradient().convert(qnn.operator, params)
    gradients = qnn.backward(grad, batch_data)

    return loss, gradients

def sgd_optimizer_qml(initial_params, data, labels, qnn, learning_rate=0.01, max_iters=100, batch_size=10, tol=1e-6):
    params = initial_params
    n_samples = len(data)
    prev_loss = float('inf')

    for iteration in range(max_iters):
        # Randomly select a mini-batch
        indices = np.random.choice(n_samples, batch_size, replace=False)
        batch_data = data[indices]
        batch_labels = labels[indices]

        # Compute the loss and gradients for the mini-batch
        loss, gradients = compute_loss_and_gradients(params, batch_data, batch_labels, qnn)

        # Check for convergence
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

        # Update parameters
        params -= learning_rate * gradients

        # Print the loss at each iteration
        print(f"Iteration {iteration}, Loss: {loss}")

    return params

# Load the dataset
data = pd.read_csv('snowfall_prediction_data.csv')
features = data[['Feature1', 'Feature2', 'Feature3']].values
labels = data['Label'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Quantum Feature Map and Variational Circuit
feature_dim = 3
feature_map = ZZFeatureMap(feature_dim)
var_circuit = TwoLocal(feature_dim, 'ry', 'cz', reps=2)

# Quantum Instance
backend = Aer.get_backend('aer_simulator')
quantum_instance = QuantumInstance(backend)

# Quantum Neural Network
qnn = CircuitQNN(feature_map.compose(var_circuit), input_params=feature_map.parameters,
                 weight_params=var_circuit.parameters, quantum_instance=quantum_instance)

# Initial parameters
initial_params = np.random.rand(var_circuit.num_parameters)

# Call the SGD optimizer function
trained_params = sgd_optimizer_qml(initial_params, X_train_scaled, y_train, qnn, learning_rate=0.01, max_iters=100, batch_size=10)
