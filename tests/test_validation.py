import jax.numpy as jnp
import pytest
from jax_qsim import Circuit, zero_state, apply_gate
import jax_qsim.ops as ops
from jax_qsim.observables import PauliString


def test_zero_state_rejects_non_positive_qubits():
    with pytest.raises(ValueError):
        zero_state(0)
    with pytest.raises(ValueError):
        zero_state(-1)


def test_apply_gate_rejects_out_of_range_target():
    state = zero_state(2)
    with pytest.raises(ValueError):
        apply_gate(state, ops.X, [5])
    with pytest.raises(ValueError):
        apply_gate(state, ops.X, [-1])


def test_apply_gate_rejects_duplicate_targets():
    state = zero_state(2)
    with pytest.raises(ValueError):
        apply_gate(state, ops.CNOT, [0, 0])


def test_apply_gate_rejects_mismatched_matrix_dimension():
    state = zero_state(2)
    with pytest.raises(ValueError):
        # ops.CNOT is a 4x4 matrix but only one target is given.
        apply_gate(state, ops.CNOT, [0])


def test_circuit_rejects_out_of_range_qubit():
    c = Circuit(num_qubits=2)
    with pytest.raises(ValueError):
        c.h(5)
    with pytest.raises(ValueError):
        c.cnot(0, 3)


def test_circuit_rejects_duplicate_targets():
    c = Circuit(num_qubits=2)
    with pytest.raises(ValueError):
        c.cnot(0, 0)
    with pytest.raises(ValueError):
        c.swap(1, 1)


def test_circuit_run_rejects_insufficient_params():
    c = Circuit(num_qubits=1)
    c.ry(0, param_index=0)
    c.rz(0, param_index=1)
    with pytest.raises(ValueError):
        c.run(jnp.array([0.5]))


def test_pauli_string_rejects_out_of_range_qubit():
    state = zero_state(2)
    obs = PauliString({5: 'Z'})
    with pytest.raises(ValueError):
        obs.apply(state)
