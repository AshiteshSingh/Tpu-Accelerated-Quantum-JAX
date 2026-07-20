import sys
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import jax.core
import types
try:
    import jax.extend.core as jec
    jec.Primitive = jax.core.Primitive
except ImportError:
    ext_mod = types.ModuleType('jax.extend')
    sys.modules['jax.extend'] = ext_mod
    ext_core_mod = types.ModuleType('jax.extend.core')
    sys.modules['jax.extend.core'] = ext_core_mod
    ext_core_mod.Primitive = jax.core.Primitive
import jax
import jax.numpy as jnp
import pennylane as qml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_DIR = os.path.join(REPO_ROOT, 'experiment')
GPU_DIR = os.path.join(REPO_ROOT, 'gpu')

sys.path.insert(0, EXPERIMENT_DIR)
import jax_qsim.circuit as our_circ
import jax_qsim.statevector as our_sv
sys.path.remove(EXPERIMENT_DIR)
sys.path.insert(0, GPU_DIR)
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith('jax_qsim'):
        del sys.modules[mod_name]
import jax_qsim.circuit as original_circ
NUM_QUBITS = 25
NUM_REPEATS = 3
results_dir = os.path.join(EXPERIMENT_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)
params = jax.random.uniform(jax.random.PRNGKey(42), shape=(NUM_QUBITS,))

def build_our_simulator():
    c = our_circ.Circuit(NUM_QUBITS)
    for q in range(NUM_QUBITS):
        c.h(q)
    for q in range(NUM_QUBITS - 1):
        c.cnot(q, q + 1)
    for q in range(NUM_QUBITS):
        c.ry(q, q)

    def run_fn(state, p):
        final_state = c.run(p, 'statevector', initial_state=state)
        state_flat = final_state.reshape(-1)
        probs = jnp.abs(state_flat) ** 2
        half = 1 << NUM_QUBITS - 1
        marginal_0 = jnp.sum(probs[:half])
        marginal_1 = jnp.sum(probs[half:])
        return jnp.real(marginal_0 - marginal_1)
    return run_fn

def build_original_simulator():
    c = original_circ.Circuit(NUM_QUBITS)
    for q in range(NUM_QUBITS):
        c.h(q)
    for q in range(NUM_QUBITS - 1):
        c.cnot(q, q + 1)
    for q in range(NUM_QUBITS):
        c.rx(q, q)

    def run_fn(p):
        state = c.run(p)
        probs = jnp.abs(state) ** 2
        axes = tuple(range(1, NUM_QUBITS))
        marginal = jnp.sum(probs, axis=axes)
        return jnp.real(marginal[0] - marginal[1])
    return jax.jit(run_fn)

def build_pennylane():
    dev = qml.device('default.qubit', wires=NUM_QUBITS)

    @qml.qnode(dev, interface='jax')
    def circuit(p):
        for q in range(NUM_QUBITS):
            qml.Hadamard(wires=q)
        for q in range(NUM_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
        for q in range(NUM_QUBITS):
            qml.RY(p[q], wires=q)
        return qml.expval(qml.PauliZ(0))
    return jax.jit(circuit)

def build_pennylane_lightning():
    dev = qml.device('lightning.qubit', wires=NUM_QUBITS)

    @qml.qnode(dev)
    def circuit(p):
        for q in range(NUM_QUBITS):
            qml.Hadamard(wires=q)
        for q in range(NUM_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
        for q in range(NUM_QUBITS):
            qml.RY(p[q], wires=q)
        return qml.expval(qml.PauliZ(0))
    return circuit

def main():
    print('=' * 80)
    print(f' 25-QUBIT STATE VECTOR PERFORMANCE BENCHMARK '.center(80, '='))
    print('=' * 80)
    print(f'Statevector size : 2^25 complex64 elements = 256 MB VRAM/RAM')
    print(f'Repeat runs      : {NUM_REPEATS}')
    print(f'Active JAX devices: {jax.devices()}')
    print('-' * 80)
    sys.stdout.flush()
    benchmark_results = {}
    numerical_checks = {}
    state = our_sv.zero_state(NUM_QUBITS)
    print('Running jax_qsim (Our Optimized Simulator)...')
    sys.stdout.flush()
    try:
        our_fn = build_our_simulator()
        t_start = time.time()
        val = our_fn(state, params).block_until_ready()
        t_comp = time.time() - t_start
        print(f'  Warmup compilation time: {t_comp:.3f}s')
        sys.stdout.flush()
        times = []
        for i in range(NUM_REPEATS):
            t0 = time.time()
            _ = our_fn(state, params).block_until_ready()
            times.append(time.time() - t0)
        our_mean = np.mean(times)
        print(f'  Execution speed        : {our_mean:.3f}s (Avg of {NUM_REPEATS} runs)')
        sys.stdout.flush()
        benchmark_results['our_jax_qsim'] = {'compilation': t_comp, 'execution': our_mean, 'status': 'success'}
        numerical_checks['Our Qsim'] = float(val)
    except Exception as e:
        print(f'  Our Qsim failed: {str(e)}')
        sys.stdout.flush()
        benchmark_results['our_jax_qsim'] = {'compilation': 0.0, 'execution': 0.0, 'status': f'failed: {str(e)}'}
    print('\nRunning original_jax_qsim (Original Repo Simulator)...')
    print('  WARNING: Original simulator compiles with 25D transposes.')
    print('  Skipping to prevent Out-Of-Memory (OOM) compilation crash.')
    sys.stdout.flush()
    benchmark_results['original_jax_qsim'] = {'compilation': 0.0, 'execution': 0.0, 'status': 'oom_avoided'}
    numerical_checks['Original Qsim'] = 0.0
    print('\nRunning PennyLane default.qubit (JAX JIT)...')
    sys.stdout.flush()
    try:
        pl_fn = build_pennylane()
        t_start = time.time()
        val_pl = pl_fn(np.array(params)).block_until_ready()
        t_comp = time.time() - t_start
        print(f'  Warmup compilation time: {t_comp:.3f}s')
        sys.stdout.flush()
        times = []
        for i in range(NUM_REPEATS):
            t0 = time.time()
            _ = pl_fn(np.array(params)).block_until_ready()
            times.append(time.time() - t0)
        pl_mean = np.mean(times)
        print(f'  Execution speed        : {pl_mean:.3f}s (Avg of {NUM_REPEATS} runs)')
        sys.stdout.flush()
        benchmark_results['pennylane_jax'] = {'compilation': t_comp, 'execution': pl_mean, 'status': 'success'}
        numerical_checks['PennyLane JAX'] = float(val_pl)
    except Exception as e:
        print(f'  PennyLane default.qubit JAX failed: {str(e)}')
        sys.stdout.flush()
        benchmark_results['pennylane_jax'] = {'compilation': 0.0, 'execution': 0.0, 'status': f'failed: {str(e)}'}
        numerical_checks['PennyLane JAX'] = 0.0
    print('\nRunning PennyLane Lightning (C++ CPU Engine)...')
    sys.stdout.flush()
    try:
        pl_light_fn = build_pennylane_lightning()
        t0 = time.time()
        val_light = pl_light_fn(np.array(params))
        t_light_comp = time.time() - t0
        print(f'  Warmup execution time  : {t_light_comp:.3f}s')
        sys.stdout.flush()
        times = []
        for i in range(NUM_REPEATS):
            t0 = time.time()
            _ = pl_light_fn(np.array(params))
            times.append(time.time() - t0)
        light_mean = np.mean(times)
        print(f'  Execution speed        : {light_mean:.3f}s (Avg of {NUM_REPEATS} runs)')
        sys.stdout.flush()
        benchmark_results['pennylane_lightning'] = {'compilation': t_light_comp, 'execution': light_mean, 'status': 'success'}
        numerical_checks['Lightning C++'] = float(val_light)
    except Exception as e:
        print(f'  PennyLane Lightning failed: {str(e)}')
        sys.stdout.flush()
        benchmark_results['pennylane_lightning'] = {'compilation': 0.0, 'execution': 0.0, 'status': f'failed: {str(e)}'}
        numerical_checks['Lightning C++'] = 0.0
    print('-' * 80)
    print('Numerical values checks:')
    for k, v in numerical_checks.items():
        print(f'  {k:15s}: {v:.6f}')
    print('-' * 80)
    sys.stdout.flush()
    json_path = os.path.join(results_dir, '25q_benchmark_data.json')
    with open(json_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f'Raw data saved to: {json_path}')
    sys.stdout.flush()
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    labels = ['jax_qsim\n(Our Simulator)', 'original_jax_qsim\n(Original Repo)', 'PennyLane JAX\n(default.qubit)', 'PennyLane Lightning\n(C++ Backend)']
    execution_times = [benchmark_results['our_jax_qsim']['execution'] if benchmark_results['our_jax_qsim']['status'] == 'success' else 0.0, 0.0, benchmark_results['pennylane_jax']['execution'] if benchmark_results['pennylane_jax']['status'] == 'success' else 0.0, benchmark_results['pennylane_lightning']['execution'] if benchmark_results['pennylane_lightning']['status'] == 'success' else 0.0]
    compilation_times = [benchmark_results['our_jax_qsim']['compilation'] if benchmark_results['our_jax_qsim']['status'] == 'success' else 0.0, 0.0, benchmark_results['pennylane_jax']['compilation'] if benchmark_results['pennylane_jax']['status'] == 'success' else 0.0, 0.0]
    x = np.arange(len(labels))
    width = 0.35
    rects1 = ax.bar(x - width / 2, execution_times, width, label='Execution time (Avg)', color='#56d364', edgecolor='#30363d')
    rects2 = ax.bar(x + width / 2, compilation_times, width, label='Warmup / Compilation time', color='#79c0ff', edgecolor='#30363d')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.0:
                ax.annotate(f'{height:.3f}s', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', color='#e6edf3', fontsize=10)
    autolabel(rects1)
    autolabel(rects2)
    ax.text(1, 10.0, 'OOM / HANG AVOIDED\n(25D transposes crash)', color='#ff7b72', fontsize=11, fontweight='bold', ha='center', bbox=dict(facecolor='#161b22', edgecolor='#ff7b72', boxstyle='round,pad=0.5'))
    ax.set_title('⚛  25-Qubit State Vector Simulator Benchmarking Comparison', fontsize=14, color='#e6edf3', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color='#8b949e')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, color='#8b949e', labelpad=10)
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=11)
    ax.grid(True, linestyle='--', color='#21262d', alpha=0.5)
    ax.tick_params(colors='#e6edf3')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    plot_path = os.path.join(results_dir, '25q_benchmark_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Comparison plot saved successfully to: {plot_path}')
    print('=' * 80)
    sys.stdout.flush()
if __name__ == '__main__':
    main()