import numpy as np 
import cirq 
import cirq_google
import qsimcirq
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,AutoMinorLocator
from typing import List, Tuple, Dict,Sequence, Any, Optional
from DTC_UTIL import MyNoiseModel, simulate_dtc_circuit_list, get_polarizations
from cirq import Y

def create_dtc_circuit(
    qubits: Sequence[cirq.Qid],
    cycles: int,
    g_value: float,
    theta: float,
    phi: float,
    alpha: float,
    beta: float,
    local_fields: float,
    dimensions: int = 1
) -> List[cirq.Circuit]:
    circuit = cirq.Circuit(cirq.Moment(Y(q) for q in qubits))
    
    u_layer = cirq.Moment(
        cirq.PhasedXZGate(
            x_exponent=g_value,
            z_exponent=local_fields,
            axis_phase_exponent=0.0
        )(q) for q in qubits
    )
    
    coupling_gate = cirq.PhasedFSimGate.from_fsim_rz(
        theta=theta,
        phi=phi,
        rz_angles_before=(alpha, alpha),
        rz_angles_after=(beta, beta),
    )

    if dimensions == 1:
        even_pairs = [(q,q+1) for i, q in enumerate(qubits[:-1]) if i%2==0]
        odd_pairs = [(q,q+1) for i, q in enumerate(qubits[:-1]) if i%2==1]
        cycle = [u_layer,
                 cirq.Moment(coupling_gate(*pair) for pair in even_pairs),
                 cirq.Moment(coupling_gate(*pair) for pair in odd_pairs)]
    else:
        h_evens, h_odds, v_evens, v_odds = _grid_coupling_2d(qubits)
        cycle = [u_layer,
                 cirq.Moment(coupling_gate(*pair) for pair in h_evens),
                 cirq.Moment(coupling_gate(*pair) for pair in h_odds),
                 cirq.Moment(coupling_gate(*pair) for pair in v_evens),
                 cirq.Moment(coupling_gate(*pair) for pair in v_odds)]

    circuits = [circuit.copy()]
    for _ in range(cycles):
        circuit.append(cycle)
        circuits.append(circuit.copy())
    
    return circuits

def _grid_coupling_2d(qubits: List[cirq.GridQubit]) -> tuple:
    h_evens, h_odds, v_evens, v_odds = [], [], [], []
    
    # Horizontal couplings
    for q in qubits:
        right = cirq.GridQubit(q.row, q.col + 1)
        if right in qubits:
            if (q.row + q.col) % 2 == 0:
                h_evens.append((q, right))
            else:
                h_odds.append((q, right))
    
    # Vertical couplings
    for q in qubits:
        down = cirq.GridQubit(q.row + 1, q.col)
        if down in qubits:
            if (q.row + q.col) % 2 == 0:
                v_evens.append((q, down))
            else:
                v_odds.append((q, down))
    
    return h_evens, h_odds, v_evens, v_odds

def run_dtc_simulation(
    circuits: List[cirq.Circuit],
    noise_params: dict = None
) -> np.ndarray:
    noise_model = MyNoiseModel(
        depolorizationErrorRate=noise_params.get('depolarizing', 0.0),
        phaseDampingErrorRate=noise_params.get('phase_damping', 0.0),
        amplitudeDampingErrorRate=noise_params.get('amplitude_damping', 0.0),
    ) if noise_params else None 
    
    probabilities = simulate_dtc_circuit_list(circuits, noise_model)
    return get_polarizations(
        probabilities=probabilities,
        num_qubits=len(circuits[0].all_qubits())
    )

def plot_dtc_results(
    polarizations: np.ndarray,
    cycles: int,
    qubit_layout: str ='1D'
):
    dtc_z = np.transpose(polarizations)
    plt.figure(figsize=(12, 8))
    plt.imshow(dtc_z, aspect='auto')
    plt.xlabel('Floquet Cycle', fontsize=16) 
    plt.ylabel('Qubit Index' if qubit_layout=='1D' else 'Grid Position', fontsize=16)   
    plt.title('Qubit Polarization Dynamics', fontsize=20)
    plt.colorbar(label='Z-Polarization')
    
    avg_polarization = np.mean(polarizations, axis=1)
    spectrum = np.abs(np.fft.fft(avg_polarization))  # Added np.abs() here
    
    plt.figure(figsize=(12, 8))
    plt.plot(np.linspace(0, 1, len(spectrum)), spectrum)
    plt.xlabel('Frequency (ω/ω₀)', fontsize=16)
    plt.ylabel('Spectral Power', fontsize=16)
    plt.title('Subharmonic Response', fontsize=20)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Subharmonic Frequency')
    plt.legend()

SIM_PARAMS = {
    'cycles': 100,
    'g_value': 0.9,
    'theta': np.pi,
    'phi': np.pi,
    'alpha': np.pi,
    'beta': np.pi,
    'local_fields': 0.001  # Fixed typo from 'local_feilds'
}

# 1D Simulation
# qubits_1d = cirq.LineQubit.range(8)
# circuits_1d = create_dtc_circuit(qubits_1d, dimensions=1, **SIM_PARAMS)
# results_1d = run_dtc_simulation(circuits_1d)
# plot_dtc_results(results_1d, SIM_PARAMS['cycles'], qubit_layout='1D')

# 2D Simulation
# qubits_2d = cirq.GridQubit.square(4)
# circuits_2d = create_dtc_circuit(qubits_2d, dimensions=2, **SIM_PARAMS)
# results_2d = run_dtc_simulation(circuits_2d, noise_params={'depolarizing': 0.001})
# plot_dtc_results(results_2d, SIM_PARAMS['cycles'], qubit_layout='2D')
