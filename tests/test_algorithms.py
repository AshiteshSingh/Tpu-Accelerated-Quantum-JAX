import jax
import jax.numpy as jnp
from jax_qsim import Circuit
import math

def test_parameterized_circuit():
    """Test a simple parameterized circuit for VQE/QAOA use cases."""
    c = Circuit(num_qubits=2)
    c.rx(0, param="theta")
    c.ry(1, param="phi")
    c.cnot(0, 1)
    
    run_compiled = c.compile()
    
    # Test with theta=pi, phi=pi/2
    params = jnp.array([jnp.pi, jnp.pi/2])
    state = run_compiled(params)
    
    expected = jnp.zeros((2, 2), dtype=jnp.complex64)
    expected = expected.at[1, 0].set(-1.0j / math.sqrt(2.0))
    expected = expected.at[1, 1].set(-1.0j / math.sqrt(2.0))
    
    assert jnp.allclose(state, expected, atol=1e-6)

def test_ghz_state_scaling():
    """Test GHZ state preparation for larger qubit numbers."""
    num_qubits = 6
    c = Circuit(num_qubits=num_qubits)
    
    c.h(0)
    for i in range(num_qubits - 1):
        c.cnot(i, i + 1)
        
    state = c.run(jnp.array([]))
    
    expected = jnp.zeros((2,) * num_qubits, dtype=jnp.complex64)
    expected = expected.at[(0,) * num_qubits].set(1.0 / math.sqrt(2.0))
    expected = expected.at[(1,) * num_qubits].set(1.0 / math.sqrt(2.0))
    
    assert jnp.allclose(state, expected, atol=1e-6)
    
def test_measurements_probability():
    """Test measurement probabilities of a superposition state."""
    c = Circuit(num_qubits=1)
    c.h(0)
    state = c.run(jnp.array([]))
    
    probabilities = jnp.abs(state) ** 2
    expected = jnp.array([0.5, 0.5])
    
    assert jnp.allclose(probabilities, expected, atol=1e-6)
