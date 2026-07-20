import jax
import jax.numpy as jnp
from jax_qsim.core import zero_state, apply_gate
import jax_qsim.ops as ops

class Circuit:

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates = []
        self.num_params = 0

    def _register_param(self, param_index: int):
        if param_index is not None:
            self.num_params = max(self.num_params, param_index + 1)

    def _validate_targets(self, targets: list[int]):
        for t in targets:
            if not 0 <= t < self.num_qubits:
                raise ValueError(
                    f'Qubit index {t} is out of range for a {self.num_qubits}-qubit '
                    f'circuit (valid range is [0, {self.num_qubits})).')
        if len(set(targets)) != len(targets):
            raise ValueError(f'Gate targets must be distinct qubits, got {targets}')

    def h(self, qubit: int):
        self._validate_targets([qubit])
        self.gates.append({'type': 'static', 'matrix': ops.H, 'targets': [qubit]})
        return self

    def x(self, qubit: int):
        self._validate_targets([qubit])
        self.gates.append({'type': 'static', 'matrix': ops.X, 'targets': [qubit]})
        return self

    def y(self, qubit: int):
        self._validate_targets([qubit])
        self.gates.append({'type': 'static', 'matrix': ops.Y, 'targets': [qubit]})
        return self

    def z(self, qubit: int):
        self._validate_targets([qubit])
        self.gates.append({'type': 'static', 'matrix': ops.Z, 'targets': [qubit]})
        return self

    def s(self, qubit: int):
        self._validate_targets([qubit])
        self.gates.append({'type': 'static', 'matrix': ops.S, 'targets': [qubit]})
        return self

    def t(self, qubit: int):
        self._validate_targets([qubit])
        self.gates.append({'type': 'static', 'matrix': ops.T, 'targets': [qubit]})
        return self

    def cnot(self, control: int, target: int):
        self._validate_targets([control, target])
        self.gates.append({'type': 'static', 'matrix': ops.CNOT, 'targets': [control, target]})
        return self

    def cz(self, control: int, target: int):
        self._validate_targets([control, target])
        self.gates.append({'type': 'static', 'matrix': ops.CZ, 'targets': [control, target]})
        return self

    def swap(self, qubit1: int, qubit2: int):
        self._validate_targets([qubit1, qubit2])
        self.gates.append({'type': 'static', 'matrix': ops.SWAP, 'targets': [qubit1, qubit2]})
        return self

    def rx(self, qubit: int, param_index: int):
        self._validate_targets([qubit])
        self._register_param(param_index)
        self.gates.append({'type': 'parameterized', 'gate_func': ops.rx, 'targets': [qubit], 'param_index': param_index})
        return self

    def ry(self, qubit: int, param_index: int):
        self._validate_targets([qubit])
        self._register_param(param_index)
        self.gates.append({'type': 'parameterized', 'gate_func': ops.ry, 'targets': [qubit], 'param_index': param_index})
        return self

    def rz(self, qubit: int, param_index: int):
        self._validate_targets([qubit])
        self._register_param(param_index)
        self.gates.append({'type': 'parameterized', 'gate_func': ops.rz, 'targets': [qubit], 'param_index': param_index})
        return self

    def phase_shift(self, qubit: int, param_index: int):
        self._validate_targets([qubit])
        self._register_param(param_index)
        self.gates.append({'type': 'parameterized', 'gate_func': ops.phase_shift, 'targets': [qubit], 'param_index': param_index})
        return self

    def crx(self, control: int, target: int, param_index: int):
        self._validate_targets([control, target])
        self._register_param(param_index)
        self.gates.append({'type': 'parameterized', 'gate_func': ops.crx, 'targets': [control, target], 'param_index': param_index})
        return self

    def cry(self, control: int, target: int, param_index: int):
        self._validate_targets([control, target])
        self._register_param(param_index)
        self.gates.append({'type': 'parameterized', 'gate_func': ops.cry, 'targets': [control, target], 'param_index': param_index})
        return self

    def crz(self, control: int, target: int, param_index: int):
        self._validate_targets([control, target])
        self._register_param(param_index)
        self.gates.append({'type': 'parameterized', 'gate_func': ops.crz, 'targets': [control, target], 'param_index': param_index})
        return self

    def cphase(self, control: int, target: int, param_index: int):
        self._validate_targets([control, target])
        self._register_param(param_index)
        self.gates.append({'type': 'parameterized', 'gate_func': ops.cphase, 'targets': [control, target], 'param_index': param_index})
        return self

    def run(self, params: jnp.ndarray, initial_state: jnp.ndarray=None) -> jnp.ndarray:
        if params.shape[0] < self.num_params:
            raise ValueError(
                f'Circuit requires {self.num_params} parameter(s), but only '
                f'{params.shape[0]} were provided.')
        if initial_state is None:
            state = zero_state(self.num_qubits)
        else:
            state = initial_state
        for gate in self.gates:
            if gate['type'] == 'static':
                state = apply_gate(state, gate['matrix'], gate['targets'])
            elif gate['type'] == 'parameterized':
                val = params[gate['param_index']]
                matrix = gate['gate_func'](val)
                state = apply_gate(state, matrix, gate['targets'])
        return state

    def compile(self):
        return jax.jit(self.run)

    def __repr__(self) -> str:
        return f'Circuit(num_qubits={self.num_qubits}, gates={len(self.gates)}, params={self.num_params})'