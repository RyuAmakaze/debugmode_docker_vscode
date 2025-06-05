import numpy as np
import torch

import config

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    CRXGate,
    CRYGate,
    CU3Gate,
    RXXGate,
)
try:  # Qiskit <2.0 uses IsingXYGate, >=2.0 renamed it
    from qiskit.circuit.library import IsingXYGate
except Exception:  # pragma: no cover - handle version differences
    try:
        from qiskit.circuit.library import XXPlusYYGate as IsingXYGate
    except Exception:
        from qiskit.circuit.library import XYGate as IsingXYGate

def multi_rz(qc: QuantumCircuit, qubits: list[int], theta: float):
    # CNOT チェーン
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])
    # 最後の量子ビットに RZ をかける
    qc.rz(theta, qubits[-1])
    # CNOT チェーンを戻す
    for i in reversed(range(len(qubits) - 1)):
        qc.cx(qubits[i], qubits[i + 1])

def data_to_circuit(angles, params=None, entangling=False):
    """Return a QuantumCircuit encoding ``angles`` via Y rotations.

    Parameters
    ----------
    angles : Sequence[float] or torch.Tensor
        Rotation angles for RY gates on each qubit.
    params : Sequence[float] or torch.Tensor, optional
        Rotation angles for each additional layer.  If ``params`` is
        one-dimensional and ``entangling`` is ``False`` the values are
        added directly to ``angles`` for backwards compatibility.  If
        ``params`` is two-dimensional it is interpreted as
        ``(num_layers, n_qubits)`` with each layer applied sequentially
        using ``RZ`` rotations.
    entangling : bool, optional
        If ``True`` a chain of ``CX`` gates is inserted after each
        parameterized layer to introduce entanglement.

    Notes
    -----
    If qiskit is not installed, this function raises ``ImportError``.
    """

    if torch.is_tensor(angles):
        angles = angles.detach().cpu().tolist()
    angles = np.array(angles, dtype=float)
    n_qubits = angles.shape[0]

    # Backwards compatible path: single parameter vector without entanglement
    if params is not None and not entangling and np.ndim(params) == 1:
        if torch.is_tensor(params):
            params = params.detach().cpu().tolist()
        angles = angles + np.array(params, dtype=float)
        qc = QuantumCircuit(n_qubits)
        for i, theta in enumerate(angles):
            qc.ry(float(theta), i)
        return qc

    qc = QuantumCircuit(n_qubits)
    for i, theta in enumerate(angles):
        qc.ry(float(theta), i)

    if params is not None:
        if torch.is_tensor(params):
            params = params.detach().cpu().tolist()
        params = np.array(params, dtype=float)
        params = np.atleast_2d(params)
        for layer in params:
            for q, theta in enumerate(layer):
                qc.ry(float(theta), q)
            if entangling and n_qubits > 1:
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)
    return qc


def circuit_state_probs(circuit):
    """Simulate ``circuit`` and return measurement probabilities.

    When CUDA and a GPU-enabled ``AerSimulator`` are available the
    simulation is executed on the GPU for improved performance.  If the
    GPU simulator is unavailable the function falls back to the default
    :class:`~qiskit.quantum_info.Statevector` implementation.
    """

    if circuit.num_qubits > 24:
        raise ValueError(
            f"circuit_state_probs: {circuit.num_qubits} qubits exceeds the 24 qubit limit of Statevector simulation"
        )

    probs = None

    if torch.cuda.is_available():
        try:
            from qiskit_aer import AerSimulator

            sim = AerSimulator(method="statevector", device="GPU")
            circ = circuit.copy()
            circ.save_statevector()
            result = sim.run(circ).result()
            state = result.get_statevector()
            probs = state.probabilities(qargs=list(range(circuit.num_qubits))[::-1])
        except Exception:
            probs = None

    if probs is None:
        state = Statevector.from_instruction(circuit)
        probs = state.probabilities(qargs=list(range(circuit.num_qubits))[::-1])

    return torch.tensor(probs, dtype=torch.float32)

def parameter_shift_gradients(angles, params, shift=np.pi / 2, entangling=False):
    """Return probabilities and gradients via the parameter-shift rule.

    Parameters
    ----------
    angles : Sequence[float] or torch.Tensor
        Data-encoding rotation angles for each qubit.
    params : Sequence[float] or torch.Tensor
        Additional learned rotation angles. Can be ``(n_qubits,)`` or
        ``(num_layers, n_qubits)``.
    shift : float, optional
        Shift amount for the parameter-shift rule (default: ``π/2``).
    entangling : bool, optional
        If ``True`` entangling ``CX`` gates are inserted between layers.
    """


    if torch.is_tensor(angles):
        angles = angles.detach().cpu().tolist()
    angles = np.array(angles, dtype=float)

    if torch.is_tensor(params):
        params = params.detach().cpu().tolist()
    params = np.array(params, dtype=float)

    # Backwards compatible path: single layer without entanglement
    if params.ndim == 1 and not entangling:
        base_circuit = data_to_circuit(angles, params, entangling=False)
        base_probs = circuit_state_probs(base_circuit)

        grads = []
        for i in range(len(params)):
            shift_vec = np.zeros_like(params)
            shift_vec[i] = shift
            plus_circ = data_to_circuit(angles, params + shift_vec, entangling=False)
            minus_circ = data_to_circuit(angles, params - shift_vec, entangling=False)
            plus_probs = circuit_state_probs(plus_circ)
            minus_probs = circuit_state_probs(minus_circ)
            grad = 0.5 * (plus_probs - minus_probs)
            grads.append(grad)
        grads = torch.stack(grads, dim=0)
        return base_probs, grads

    # Multi-layer or entangling case
    params = np.atleast_2d(params)
    num_layers, n_qubits = params.shape

    base_circuit = data_to_circuit(angles, params, entangling=entangling)
    base_probs = circuit_state_probs(base_circuit)

    grads = torch.zeros(num_layers, n_qubits, base_probs.numel(), dtype=base_probs.dtype)

    for layer in range(num_layers):
        for q in range(n_qubits):
            shift_mat = np.zeros_like(params)
            shift_mat[layer, q] = shift
            plus_circ = data_to_circuit(angles, params + shift_mat, entangling=entangling)
            minus_circ = data_to_circuit(angles, params - shift_mat, entangling=entangling)
            plus_probs = circuit_state_probs(plus_circ)
            minus_probs = circuit_state_probs(minus_circ)
            grad = 0.5 * (plus_probs - minus_probs)
            grads[layer, q] = grad

    return base_probs, grads


def adaptive_entangling_circuit(
    x,
    *,
    n_qubits=config.NUM_QUBITS,
    features_per_layer=config.FEATURES_PER_LAYER,
    lambdas=None,
    gamma=1.0,
    delta=1.0,
):
    """Return a multi-stage entangling circuit for ``x``.

    This helper implements the six-stage design discussed in the
    project notes. The number of qubits and required features can be
    configured via :mod:`config`.

    Parameters
    ----------
    x : Sequence[float] or torch.Tensor
        Input features for a single layer.  At least
        ``features_per_layer`` elements are required.
    n_qubits : int, optional
        Number of qubits used in the circuit.
    features_per_layer : int, optional
        Number of features consumed from ``x``.  By default this is
        :data:`config.FEATURES_PER_LAYER`.
    lambdas : Sequence[float], optional
        Per-qubit scaling factors for stage 3. If ``None`` all ones are
        used.
    gamma : float, optional
        Scaling factor for the long-range entangling gate (stage 4).
    delta : float, optional
        Scaling factor for the global ``MultiRZ`` gate (stage 5).
    """


    if torch.is_tensor(x):
        x = x.detach().cpu().tolist()
    x = np.array(x, dtype=float)

    if len(x) < features_per_layer:
        raise ValueError(
            f"adaptive_entangling_circuit requires at least {features_per_layer} features"
        )

    if lambdas is None:
        lambdas = np.ones(n_qubits)
    else:
        lambdas = np.asarray(lambdas, dtype=float)

    qc = QuantumCircuit(n_qubits)

    # Stage 0: local encoding
    for j in range(n_qubits):
        angle = np.pi * x[j]
        qc.ry(float(angle), j)

    # Stage 1: immediate neighbor entanglement
    for j in range(n_qubits):
        x_a = x[j]
        x_b = x[(j + 1) % n_qubits]
        angle = np.pi * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
        qc.append(CRXGate(angle), [j, (j + 1) % n_qubits])

    # Stage 2: next-nearest neighbor correlations
    for j in range(n_qubits):
        vals = [x[j], x[(j + 1) % n_qubits], x[(j + 2) % n_qubits]]
        angle = np.pi * float(np.mean(vals))
        qc.append(CRYGate(angle), [j, (j + 2) % n_qubits])

    # Stage 3: adaptive CRot with layer-dependent scaling
    for j in range(n_qubits):
        x_a = x[j]
        x_b = x[(j + 1) % n_qubits]
        scale = lambdas[j % len(lambdas)]
        angle = np.pi * scale * 0.5 * (x_a + x_b)
        qc.append(CU3Gate(angle, 0.0, 0.0), [j, (j + 3) % n_qubits])

    # Stage 4: long-range entanglement via IsingXY-like coupling
    half = n_qubits // 2
    for j in range(half):
        qc.append(IsingXYGate(np.pi * gamma), [j, j + half])

    # Stage 5: global multi-qubit rotation
    global_angle = np.pi * delta * x[min(features_per_layer - 1, len(x) - 1)]
    multi_rz(qc, list(range(n_qubits)), global_angle)

    return qc
