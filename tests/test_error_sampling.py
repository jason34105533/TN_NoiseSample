import numpy as np
import pytest
from tn_noise_sim.noise_model import NoiseModel, GateNoiseSpec, ErrorType
from tn_noise_sim.error_sampling import ErrorSampler, _I, _X, _Y, _Z, _PAULI_2Q_NONID


@pytest.fixture
def pauli_noise():
    return NoiseModel.uniform_pauli(num_gates=5, probability=0.1)


@pytest.fixture
def depol_noise():
    return NoiseModel.uniform_depolarizing(num_gates=4, probability=0.1)


# ---- shot count tests ----

def test_proportional_shot_sum(pauli_noise):
    sampler = ErrorSampler(pauli_noise, num_gates=5, rng_seed=0)
    sampler.sample(num_error_sets=10, total_shots=100, mode="proportional")
    assert sum(sampler.shot_counts) == 100


def test_non_proportional_shot_sum(pauli_noise):
    sampler = ErrorSampler(pauli_noise, num_gates=5, rng_seed=0)
    sampler.sample(num_error_sets=7, total_shots=100, mode="non_proportional")
    assert sum(sampler.shot_counts) == 100


def test_non_proportional_equal_distribution(pauli_noise):
    sampler = ErrorSampler(pauli_noise, num_gates=5, rng_seed=0)
    sampler.sample(num_error_sets=10, total_shots=100, mode="non_proportional")
    assert all(c == 10 for c in sampler.shot_counts)


def test_non_proportional_remainder(pauli_noise):
    sampler = ErrorSampler(pauli_noise, num_gates=5, rng_seed=0)
    sampler.sample(num_error_sets=3, total_shots=10, mode="non_proportional")
    counts = sampler.shot_counts
    assert sum(counts) == 10
    assert max(counts) - min(counts) <= 1


# ---- operator shape tests ----

def test_pauli_operator_shapes(pauli_noise):
    sampler = ErrorSampler(pauli_noise, num_gates=5, rng_seed=1)
    sampler.sample(num_error_sets=20, total_shots=20, mode="non_proportional")
    for error_set, _ in sampler.to_list():
        for gate_idx, op in error_set.items():
            assert op.shape == (2, 2), f"gate {gate_idx}: expected (2,2), got {op.shape}"


def test_depolarizing_operator_shapes(depol_noise):
    sampler = ErrorSampler(depol_noise, num_gates=4, rng_seed=2)
    sampler.sample(num_error_sets=20, total_shots=20, mode="non_proportional")
    for error_set, _ in sampler.to_list():
        for gate_idx, op in error_set.items():
            assert op.shape == (4, 4), f"gate {gate_idx}: expected (4,4), got {op.shape}"


# ---- operator validity tests ----

def test_pauli_operators_are_paulis(pauli_noise):
    sampler = ErrorSampler(pauli_noise, num_gates=5, rng_seed=3)
    sampler.sample(num_error_sets=50, total_shots=50, mode="non_proportional")
    valid = [_I, _X, _Y, _Z]
    for error_set, _ in sampler.to_list():
        for op in error_set.values():
            assert any(np.allclose(op, v) for v in valid), "Operator not in {I,X,Y,Z}"


def test_depolarizing_operators_valid(depol_noise):
    sampler = ErrorSampler(depol_noise, num_gates=4, rng_seed=4)
    sampler.sample(num_error_sets=50, total_shots=50, mode="non_proportional")
    id4 = np.eye(4, dtype=complex)
    valid = [id4] + _PAULI_2Q_NONID
    for error_set, _ in sampler.to_list():
        for op in error_set.values():
            assert any(np.allclose(op, v) for v in valid), "Operator not a 2Q Pauli"


# ---- frequency test for Pauli probabilities ----

def test_pauli_error_frequency():
    """With p=0.3, identity should appear ~70% of the time."""
    noise = NoiseModel.uniform_pauli(num_gates=1, probability=0.3)
    sampler = ErrorSampler(noise, num_gates=1, rng_seed=5)
    sampler.sample(num_error_sets=1000, total_shots=1000, mode="non_proportional")
    id_count = sum(
        1 for es, _ in sampler.to_list()
        if 0 in es and np.allclose(es[0], _I)
    )
    ratio = id_count / 1000
    assert 0.60 < ratio < 0.80, f"Identity frequency out of expected range: {ratio:.3f}"


# ---- to_list format ----

def test_to_list_format(pauli_noise):
    sampler = ErrorSampler(pauli_noise, num_gates=5, rng_seed=6)
    sampler.sample(num_error_sets=3, total_shots=30, mode="non_proportional")
    result = sampler.to_list()
    assert len(result) == 3
    for error_set, shot_count in result:
        assert isinstance(error_set, dict)
        assert isinstance(shot_count, int)
        assert shot_count >= 0
        for k, v in error_set.items():
            assert isinstance(k, int)
            assert isinstance(v, np.ndarray)
