#/bin/python3
import numpy as np
import sympy as sp

import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator,AutoMinorLocator

from IPython import display

from typing import Sequence, Tuple, List, Iterator, Optional

import cirq
from cirq import Y, PhasedXZGate, PhasedFSimGate
import cirq_google
import qsimcirq

from scipy.interpolate import interp1d

class MyNoiseModel(cirq.NoiseModel):
    def __init__(self, depolorizationErrorRate, phaseDampingErrorRate, amplitudeDampingErrorRate):
        super().__init__()

        self.depolorizationErrorRate = depolorizationErrorRate
        self.phaseDampingErrorRate = phaseDampingErrorRate
        self.amplitudeDampingErrorRate = amplitudeDampingErrorRate

    
    def noisy_operation(self, op):
        
        if cirq.is_measurement(op):
            return op
        
        qubits = op.qubits
        n_qubits =  len(qubits)
        
        # Apply depolorizing Noise
        depolarizeChannel = cirq.depolarize(self.depolorizationErrorRate, n_qubits = n_qubits).on(*qubits)
        
        # Apply phase Damping and Amplitude Damping on Each Qubit
        phaseDamping = [cirq.phase_damp(self.phaseDampingErrorRate).on(q) for q in qubits]
        amplitudeDamping = [cirq.amplitude_damp(self.amplitudeDampingErrorRate).on(q) for q in qubits]
        
        return [op, depolarizeChannel] + phaseDamping + amplitudeDamping

def simulate_dtc_circuit_list(
    circuit_list: Sequence[cirq.Circuit],
    noise_md: cirq.NoiseModel
    ) -> np.ndarray:
    """
    Simulates a sequence of quantum circuits with a given noise model and returns
    the state probabilities at specific moments corresponding to the lengths of
    the circuits in the provided list.

    Args:
        circuit_list: A sequence of Cirq circuits to be simulated.
        noise_md: A Cirq NoiseModel to apply during simulation.

    Returns:
        A NumPy array containing the state probabilities at specified moments.
    """
    # Initialize the simulator with the provided noise model
    simulator = cirq.Simulator(noise=noise_md)

    # Determine the set of moment indices corresponding to the end of each circuit
    circuit_positions = {len(circuit) - 1 for circuit in circuit_list}

    # Select the longest circuit for simulation
    circuit = circuit_list[-1]

    # List to store the state probabilities at specified moments
    probabilities = []

    # Simulate the circuit moment by moment
    for k, step in enumerate(simulator.simulate_moment_steps(circuit=circuit)):
        # If the current moment index matches one of the circuit positions, record the state probabilities
        if k in circuit_positions:
            probabilities.append(np.abs(step.state_vector()) ** 2)

    return np.asarray(probabilities)

def get_polarizations(
    probabilities: np.ndarray,
    num_qubits: int,
    initial_states: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes qubit polarizations from a matrix of measurement probabilities.

    A polarization is the marginal probability for a qubit to be measured as 0 or 1,
    scaled to the range [-1, 1]. If `initial_states` are given, results are autocorrelated
    (i.e., compared to their initial value).

    Args:
        probabilities: np.ndarray of shape (samples, cycles, 2**num_qubits)
            Probability to measure each bit string at each cycle.
        num_qubits: Number of qubits in the system.
        initial_states: Optional np.ndarray of shape (samples, num_qubits)
            Initial values (0 or 1) for each qubit in each sample.

    Returns:
        np.ndarray of shape (samples, cycles, num_qubits) with qubit polarizations.
    """
    polarizations = []

    for qubit_index in range(num_qubits):
        # Bit mask for qubit being 0
        shift_by = num_qubits - qubit_index - 1
        zero_state_indices = [
            i for i in range(2**num_qubits) if ((i >> shift_by) & 1) == 0
        ]

        # Compute polarization: (2 * P(q=0) - 1)
        polarization = (
            2.0 * np.sum(
                probabilities.take(indices=zero_state_indices, axis=-1),
                axis=-1
            ) - 1.0
        )
        polarizations.append(polarization)

    # Convert list of arrays into a final polarization tensor
    polarizations = np.moveaxis(np.asarray(polarizations), 0, -1)  # shape: (samples, cycles, qubits)

    # Apply initial state autocorrelation if provided
    if initial_states is not None:
        initial_states = 1 - 2.0 * initial_states  # map 0 → +1, 1 → -1
        polarizations *= initial_states[:, None, :]  # broadcast over cycles

    return polarizations
