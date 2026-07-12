"""TensorNetworkBuilder: construct tensor networks from quantum circuits."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .noise_model import ErrorType, NoiseModel
from .error_sampling import ErrorSet


@dataclass
class Tensor:
    """A single tensor node in the network."""
    data: np.ndarray        # actual array data
    indices: List[str]      # index labels, one per leg
    name: str = ""          # human-readable tag (e.g. "gate_3", "ket_0")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape


@dataclass
class TNNetwork:
    """An uncontracted tensor network: a collection of Tensor nodes."""
    tensors: List[Tensor] = field(default_factory=list)
    n_qubits: int = 0
    n_gates: int = 0

    def topology_hash(self) -> str:
        """Hash based on tensor shapes and index labels (order-preserving)."""
        parts = []
        for t in self.tensors:
            parts.append(f"{t.shape}:{','.join(t.indices)}")
        return "|".join(parts)

    def index_graph(self) -> Dict[str, List[str]]:
        """Return {index_label: [tensor_names_that_use_it]} for topology checks."""
        graph: Dict[str, List[str]] = {}
        for t in self.tensors:
            for idx in t.indices:
                graph.setdefault(idx, []).append(t.name)
        return graph

    def copy(self) -> "TNNetwork":
        return TNNetwork(
            tensors=[Tensor(t.data.copy(), list(t.indices), t.name) for t in self.tensors],
            n_qubits=self.n_qubits,
            n_gates=self.n_gates,
        )


class TensorNetworkBuilder:
    """
    Builds TNNetwork objects from quantum circuits.

    Circuits are represented as dicts:
        {"n_qubits": n, "gates": [{"qubits": (q,...), "unitary": ndarray}, ...]}

    This format avoids a hard Qiskit dependency for CPU/test usage.
    When Qiskit is available, use `from_qiskit_circuit()`.
    """

    @staticmethod
    def build(
        circuit: dict,
        error_set: Optional[ErrorSet] = None,
        mode: str = "noiseless",
    ) -> TNNetwork:
        """
        Build a tensor network from a circuit dict.

        Parameters
        ----------
        circuit: dict with keys "n_qubits" and "gates"
        error_set: optional {gate_idx: error_matrix} for noisy networks
        mode: "noiseless" | "insert" | "fuse"
            - noiseless: no error tensors
            - insert: add error operators as extra nodes (changes topology)
            - fuse: multiply error into gate tensor (preserves topology)
        """
        n = circuit["n_qubits"]
        gates = circuit["gates"]

        # Index naming scheme:
        #   wire_{q}_{layer} = qubit q after layer `layer`
        # layer 0 = initial |0>, layer advances each time qubit q is acted on

        qubit_layer = [0] * n  # current output layer for each qubit

        def _wire(q: int, layer: int) -> str:
            return f"w_{q}_{layer}"

        tensors: List[Tensor] = []

        # Initial |0> state tensors — rank-1 (ket vector)
        for q in range(n):
            ket = np.array([1.0 + 0j, 0.0 + 0j])
            tensors.append(Tensor(ket, [_wire(q, 0)], f"ket_{q}"))

        for gate_idx, gate in enumerate(gates):
            qubits = gate["qubits"]
            U = gate["unitary"].astype(complex)
            nq = len(qubits)

            if nq == 1:
                q = qubits[0]
                in_layer = qubit_layer[q]
                out_layer = in_layer + 1
                qubit_layer[q] = out_layer

                # Reshape 2x2 -> shape (2,2): [out, in]
                gate_tensor = U.reshape(2, 2)

                if mode == "fuse" and error_set and gate_idx in error_set:
                    K = error_set[gate_idx].astype(complex)
                    # Fuse: K @ U — same shape (2,2)
                    gate_tensor = (K @ U).reshape(2, 2)
                    error_set = dict(error_set)  # don't mutate caller's dict
                    del error_set[gate_idx]

                indices = [_wire(q, out_layer), _wire(q, in_layer)]
                tensors.append(Tensor(gate_tensor, indices, f"gate_{gate_idx}"))

                if mode == "insert" and error_set and gate_idx in error_set:
                    K = error_set[gate_idx].astype(complex)
                    err_out = out_layer + 1
                    qubit_layer[q] = err_out
                    err_tensor = K.reshape(2, 2)
                    err_indices = [_wire(q, err_out), _wire(q, out_layer)]
                    tensors.append(Tensor(err_tensor, err_indices, f"err_{gate_idx}"))

            elif nq == 2:
                q0, q1 = qubits
                in0, in1 = qubit_layer[q0], qubit_layer[q1]
                out0, out1 = in0 + 1, in1 + 1
                qubit_layer[q0] = out0
                qubit_layer[q1] = out1

                # Reshape 4x4 -> shape (2,2,2,2): [out0, out1, in0, in1]
                gate_tensor = U.reshape(2, 2, 2, 2)

                if mode == "fuse" and error_set and gate_idx in error_set:
                    K = error_set[gate_idx].astype(complex)
                    gate_tensor = (K @ U).reshape(2, 2, 2, 2)
                    error_set = dict(error_set)
                    del error_set[gate_idx]

                indices = [
                    _wire(q0, out0), _wire(q1, out1),
                    _wire(q0, in0), _wire(q1, in1),
                ]
                tensors.append(Tensor(gate_tensor, indices, f"gate_{gate_idx}"))

                if mode == "insert" and error_set and gate_idx in error_set:
                    K = error_set[gate_idx].astype(complex)
                    err_out0, err_out1 = out0 + 1, out1 + 1
                    qubit_layer[q0] = err_out0
                    qubit_layer[q1] = err_out1
                    err_tensor = K.reshape(2, 2, 2, 2)
                    err_indices = [
                        _wire(q0, err_out0), _wire(q1, err_out1),
                        _wire(q0, out0), _wire(q1, out1),
                    ]
                    tensors.append(Tensor(err_tensor, err_indices, f"err_{gate_idx}"))

        # Final measurement bra tensors — |0> bra (measuring in computational basis)
        # We leave output wires open so the contraction yields a probability amplitude.
        # The open output indices are the "dangling" wires at the current qubit_layer.
        # (No explicit bra tensors needed; sampling is done during contraction.)

        network = TNNetwork(tensors=tensors, n_qubits=n, n_gates=len(gates))
        return network

    @staticmethod
    def _gate_tensor(gate: dict, error_op: Optional[np.ndarray] = None) -> np.ndarray:
        """Reshape (and optionally fuse an error operator into) a gate's unitary.

        Mirrors the reshape/fusion convention used by `build()`'s "fuse" mode:
        single-qubit -> (2,2) = [out, in]; two-qubit -> (2,2,2,2) = [out0,out1,in0,in1].
        """
        U = gate["unitary"].astype(complex)
        nq = len(gate["qubits"])
        shape = tuple([2] * nq + [2] * nq)
        if error_op is not None:
            K = error_op.astype(complex)
            return (K @ U).reshape(shape)
        return U.reshape(shape)

    @staticmethod
    def build_network_state(
        circuit: dict,
        error_set: Optional[ErrorSet] = None,
        mode: str = "fuse",
        num_hypersamples: int = 1,
        dtype: str = "complex128",
    ):
        """
        Build a `cuquantum.tensornet.experimental.NetworkState` from a circuit dict.

        Unlike `build()`, this applies each gate directly to the state via
        `apply_tensor_operator` in circuit order (NetworkState has no explicit
        wire-graph to construct) so the resulting contraction is bounded to
        whatever `where`/`fixed` modes a later marginal query requests, not to
        the full 2^n state.

        Parameters
        ----------
        circuit: dict with keys "n_qubits" and "gates"
        error_set: optional {gate_idx: error_matrix}; only used when mode="fuse"
        mode: "noiseless" | "fuse"
            - noiseless: apply only coherent gates (error_set ignored)
            - fuse: fuse each gate_idx in error_set into its gate tensor
        num_hypersamples: path optimizer iterations, forwarded to TNConfig

        Returns
        -------
        (state, tensor_ids, coherent_operands)
            state: the NetworkState
            tensor_ids: {gate_idx: tensor_id} for every gate (for update_tensor_operator)
            coherent_operands: {gate_idx: ndarray} the error-free tensor for each gate
                (needed to revert a gate updated via apply_error_set_gpu)
        """
        from cuquantum.tensornet.experimental import NetworkState, TNConfig

        n = circuit["n_qubits"]
        gates = circuit["gates"]

        config = TNConfig(num_hyper_samples=num_hypersamples)
        state = NetworkState(tuple([2] * n), dtype=dtype, config=config)

        tensor_ids: Dict[int, int] = {}
        coherent_operands: Dict[int, np.ndarray] = {}

        for gate_idx, gate in enumerate(gates):
            coherent_tensor = TensorNetworkBuilder._gate_tensor(gate)
            coherent_operands[gate_idx] = coherent_tensor

            error_op = None
            if mode == "fuse" and error_set and gate_idx in error_set:
                error_op = error_set[gate_idx]

            gate_tensor = (
                TensorNetworkBuilder._gate_tensor(gate, error_op)
                if error_op is not None
                else coherent_tensor
            )
            tid = state.apply_tensor_operator(gate["qubits"], gate_tensor, unitary=False)
            tensor_ids[gate_idx] = tid

        return state, tensor_ids, coherent_operands

    @staticmethod
    def from_qiskit_circuit(
        qc,
        error_set: Optional[ErrorSet] = None,
        mode: str = "noiseless",
    ) -> TNNetwork:
        """Convert a Qiskit QuantumCircuit to the internal dict format then build."""
        from qiskit.quantum_info import Operator

        n = qc.num_qubits
        gates = []
        for instruction in qc.data:
            op = instruction.operation
            qubits = tuple(qc.find_bit(q).index for q in instruction.qubits)
            U = Operator(op).data
            gates.append({"qubits": qubits, "unitary": U})

        circuit_dict = {"n_qubits": n, "gates": gates}
        return TensorNetworkBuilder.build(circuit_dict, error_set=error_set, mode=mode)
