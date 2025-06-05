import argparse
import torch
from config import NUM_QUBITS, NUM_LAYERS, NUM_OUTPUT_QUBITS
from model import QuantumLLPModel
from quantum_utils import data_to_circuit


def load_model(path: str) -> QuantumLLPModel:
    """Load a QuantumLLPModel from ``path``."""
    model = QuantumLLPModel(
        n_qubits=NUM_QUBITS,
        num_layers=NUM_LAYERS,
        entangling=NUM_LAYERS > 1,
        n_output_qubits=NUM_OUTPUT_QUBITS,
    )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def main(model_path: str, output: str | None) -> None:
    model = load_model(model_path)

    angles = torch.zeros(NUM_QUBITS + NUM_OUTPUT_QUBITS)
    circuit = data_to_circuit(angles, model.params.detach(), entangling=model.entangling)

    if output:
        circuit.draw(output="mpl", filename=output)
        print(f"Circuit diagram saved to {output}")
    else:
        print(circuit.draw())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw quantum circuit from a saved model")
    parser.add_argument("model_path", nargs="?", default="trained_quantum_llp.pt", help="Path to saved state dict")
    parser.add_argument("-o", "--output", help="Output image path (e.g., circuit.png)")
    args = parser.parse_args()
    main(args.model_path, args.output)
