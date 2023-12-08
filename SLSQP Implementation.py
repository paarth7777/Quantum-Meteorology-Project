
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import SLSQP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
data = pd.read_csv('snowfall_prediction_data.csv')
features = data[['Temperature', 'Humidity', 'Time']].values
labels = data['Snowing'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Quantum Feature Map
feature_dim = 3
feature_map = ZZFeatureMap(feature_dim)

# Adjusted Variational Circuit
var_circuit = EfficientSU2(feature_dim, reps=3)  # More complex ansatz

# Quantum Instance
backend = Aer.get_backend('aer_simulator')
quantum_instance = QuantumInstance(backend)

# Quantum Neural Network
qnn = CircuitQNN(feature_map.compose(var_circuit), input_params=feature_map.parameters,
                 weight_params=var_circuit.parameters, quantum_instance=quantum_instance)

# Adjusted Optimizer
optimizer = SLSQP(maxiter=150, tol=0.01)

# VQC with adjusted settings
vqc = VQC(feature_map=feature_map, ansatz=var_circuit, optimizer=optimizer,
          quantum_instance=quantum_instance)

# Train the model
vqc.fit(X_train_scaled, y_train)

# Evaluate the model
score = vqc.score(X_test_scaled, y_test)
print(f'Model accuracy: {score}')
