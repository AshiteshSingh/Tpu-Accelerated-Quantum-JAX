import jax
import jax.numpy as jnp
import jax_qsim.ops as ops
from jax_qsim.core import apply_gate

class PauliString:

    def __init__(self, term: dict[int, str]):
        self.term = term

    def apply(self, state: jnp.ndarray) -> jnp.ndarray:
        num_qubits = state.ndim
        for qubit in self.term:
            if not 0 <= qubit < num_qubits:
                raise ValueError(
                    f'PauliString references qubit {qubit}, but the state only '
                    f'has {num_qubits} qubit(s) (valid range is [0, {num_qubits})).')
        out_state = state
        for qubit, op_char in sorted(self.term.items()):
            if op_char == 'X':
                out_state = apply_gate(out_state, ops.X, [qubit])
            elif op_char == 'Y':
                out_state = apply_gate(out_state, ops.Y, [qubit])
            elif op_char == 'Z':
                out_state = apply_gate(out_state, ops.Z, [qubit])
            elif op_char == 'I':
                pass
            else:
                raise ValueError(f'Unknown Pauli operator: {op_char}')
        return out_state

    def __repr__(self) -> str:
        ops_str = ' * '.join((f'{op}{i}' for i, op in sorted(self.term.items())))
        return f'PauliString({(ops_str if ops_str else 'Identity')})'

def expectation(state: jnp.ndarray, observable: PauliString) -> float:
    o_state = observable.apply(state)
    val = jnp.vdot(state.reshape(-1), o_state.reshape(-1))
    return jnp.real(val)

class Hamiltonian:

    def __init__(self, coeffs: jnp.ndarray, paulis: list[PauliString]):
        self.coeffs = jnp.array(coeffs, dtype=jnp.float32)
        self.paulis = paulis

    def expectation(self, state: jnp.ndarray) -> float:
        vals = jnp.stack([expectation(state, p) for p in self.paulis])
        return jnp.dot(self.coeffs, vals)

    def __repr__(self) -> str:
        terms = [f'{c:.4f} * {p}' for c, p in zip(self.coeffs, self.paulis)]
        return ' + '.join(terms)

def sample(state: jnp.ndarray, num_samples: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    num_qubits = state.ndim
    probs = jnp.abs(state.reshape(-1)) ** 2
    log_probs = jnp.log(jnp.maximum(probs, 1e-12))
    indices = jax.random.categorical(key, log_probs, shape=(num_samples,))
    powers = 2 ** jnp.arange(num_qubits - 1, -1, -1)
    bitstrings = indices[:, None] // powers % 2
    return bitstrings