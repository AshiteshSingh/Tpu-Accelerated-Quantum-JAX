import re

with open('research_paper.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Extract 5.4 and 5.5
start_54 = content.find('### 5.4 25-Qubit State-Vector Benchmark')
start_56 = content.find('### 5.6 Qubit Scaling Sweep')
appendix_c_content = content[start_54:start_56].strip()
content = content[:start_54] + content[start_56:]

# Update the headings inside appendix_c_content
appendix_c_content = appendix_c_content.replace('### 5.4', '### C.1')
appendix_c_content = appendix_c_content.replace('### 5.5', '### C.2')
appendix_c_content = appendix_c_content.replace('Figure 11:', 'Figure C.1:')
appendix_c_content = appendix_c_content.replace('Figure 12:', 'Figure C.2:')
# Comment out figure 11 image
appendix_c_content = re.sub(r'!\[Figure C\.1.*?\]\(.*?\)', r'<!-- \g<0> (Plot missing in repository) -->', appendix_c_content)

# 2. Renumber sections 5.6-5.9
content = content.replace('### 5.6 Qubit Scaling Sweep', '### 5.4 Qubit Scaling Sweep')
content = content.replace('### 5.7 JIT Compilation Overhead', '### 5.5 JIT Compilation Overhead')
content = content.replace('### 5.8 Maximum Qubit Threshold by Hardware', '### 5.6 Maximum Qubit Threshold by Hardware')
content = content.replace('### 5.9 Negative Result / Scope Boundary', '### 5.7 Negative Result / Scope Boundary')

# 3. Fix Abstract
old_abstract = """On a consumer NVIDIA GeForce RTX 2050 (4 GB VRAM) — selected specifically to test the framework's baseline performance on an entry-level consumer-grade GPU — our JAX statevector simulator is **bandwidth-limited at 25 qubits** (15.6s vs. 9.9s for PennyLane Lightning CPU). At this scale, JAX is slower due to the RTX 2050's 192 GB/s memory bandwidth ceiling on a 256 MB state-vector; a single run on a higher-bandwidth GPU (like an RTX 4090 or A100) would substantially flip this result in favor of the GPU. The GPU advantage becomes decisive at **27 qubits**, where our simulator achieves a **1.3× speedup**† over PennyLane Lightning GPU (4.61s vs. 6.12s, †N=3 preliminary) as the 1 GB state-vector saturates CPU cache and the GPU's parallel memory system takes over. Gradient computation scales favorably: on a **120-parameter CPU circuit** (N=10 rigorous runs, 9 stable runs post-JIT warm-up), `jax.grad` computes all gradients in a single reverse-mode backward pass at **37.5ms vs. 1,826ms** for PennyLane's parameter-shift rule — a **48.7× improvement** (Section 5.3a). On a smaller 50-parameter GPU circuit (N=3 preliminary), the ratio is ~75×, consistent with PSR overhead scaling linearly with parameter count. Against PennyLane's own JAX reverse-mode backend, the advantage is ~4× (2ms vs ~8ms). On the Cloud TPU v5e-16 mesh (256 GB aggregate HBM2e), full state-vector simulation scales to **33 qubits** (64 GB), and on the TPU v6e-64 cluster, Grover's algorithm is evaluated at **36 qubits** (549 GB). The 37-qubit RCS result (Section 5.9) uses tensor-network amplitude sampling via TensorCircuit and yields **F_XEB ≈ 0** (preliminary, N=5 runs) — indicating the sampled distribution is near-uniform, a null result expected for deep chaotic circuits evaluated under finite bond-dimension approximation."""

new_abstract = """On a consumer NVIDIA GeForce RTX 2050 (4 GB VRAM) — selected specifically to test the framework's baseline performance on an entry-level consumer-grade GPU — our JAX statevector simulator establishes a baseline for single-GPU execution. Gradient computation scales favorably: on a **120-parameter CPU circuit** (N=10 rigorous runs, 9 stable runs post-JIT warm-up), `jax.grad` computes all gradients in a single reverse-mode backward pass at **37.5ms vs. 1,826ms** for PennyLane's parameter-shift rule — a **48.7× improvement** (Section 5.3a). On a smaller 50-parameter GPU circuit (N=3 preliminary), the ratio is ~75×, consistent with PSR overhead scaling linearly with parameter count. Against PennyLane's own JAX reverse-mode backend, the advantage is ~4× (2ms vs ~8ms). On the Cloud TPU v5e-16 mesh (256 GB aggregate HBM2e), full state-vector simulation scales to **33 qubits** (64 GB), and on the TPU v6e-64 cluster, Grover's algorithm is evaluated at **36 qubits** (549 GB). The 37-qubit RCS result (Section 5.7) uses tensor-network amplitude sampling via TensorCircuit and yields **F_XEB ≈ 0** (preliminary, N=5 runs) — indicating the sampled distribution is near-uniform, a null result expected for deep chaotic circuits evaluated under finite bond-dimension approximation."""

content = content.replace(old_abstract, new_abstract)

# 4. Comment out figure 14
content = re.sub(r'!\[Figure 14: GPU qubit scaling benchmark.*?\]\(.*?\)', r'<!-- \g<0> (Plot missing in repository) -->', content)

# 5. Fix Discussion and Conclusion
content = content.replace("1. **Competitive GPU performance**: At 27 qubits, our JAX simulator matches or outperforms PennyLane Lightning GPU and Qiskit-Aer GPU on the same hardware (RTX 2050)", "1. **Competitive GPU performance**: Preliminary observations (see Appendix C) suggest our JAX simulator matches or outperforms PennyLane Lightning GPU and Qiskit-Aer GPU on the same hardware (RTX 2050) at 27 qubits.")

old_bandwidth_text = "At 25 qubits (256 MB state-vector), the RTX 2050 (192 GB/s) is bandwidth-limited, not compute-limited. PennyLane Lightning CPU (using multi-core CPU DRAM at ~50–80 GB/s per-thread but higher effective bandwidth) wins at this scale on this specific hardware. The GPU advantage becomes decisive at 27+ qubits where the state-vector exceeds L3 cache."
new_bandwidth_text = "Preliminary tests at 25 qubits (256 MB state-vector) indicate the RTX 2050 (192 GB/s) is bandwidth-limited, not compute-limited. PennyLane Lightning CPU (using multi-core CPU DRAM at ~50–80 GB/s per-thread but higher effective bandwidth) wins at this scale on this specific hardware. However, a single run on a higher-bandwidth GPU (like an RTX 4090 or A100) would substantially flip this result in favor of the GPU."
content = content.replace(old_bandwidth_text, new_bandwidth_text)

old_statistical = "Sections 5.3 (gradient timing) and 5.6 (scaling sweep 4–20 qubits) now report **N=10 timed runs** with mean ± σ from raw JSON logs (`n10_benchmark_20260530_214024.json`, timestamp 2026-05-30 21:40:24). The 25q/27q GPU data (Sections 5.4–5.5) retain †N=3 from the original RTX 2050 hardware sessions, as re-running 25q/27q XLA compilation on CPU-only hardware is impractical (requires 256 MB–1 GB VRAM)."
new_statistical = "Sections 5.3 (gradient timing) and 5.4 (scaling sweep 4–20 qubits) now report **N=10 timed runs** with mean ± σ from raw JSON logs (`n10_benchmark_20260530_214024.json`, timestamp 2026-05-30 21:40:24). The 25q/27q GPU data have been moved to Appendix C as preliminary observations since they retain †N=3 from the original RTX 2050 hardware sessions, and re-running 25q/27q XLA compilation on CPU-only hardware to achieve N=10 is impractical (requires 256 MB–1 GB VRAM)."
content = content.replace(old_statistical, new_statistical)

content = content.replace("- **1.3× GPU speedup** vs PennyLane Lightning GPU at 27 qubits; GPU advantage at 25q limited by RTX 2050 memory bandwidth\n", "")

old_ack = "Because of this research and framework, researchers everywhere can now use our code to perform massive distributed quantum simulations directly on TPUs without having to write any complex C++ or CUDA distributed systems code."
content = content.replace(" " + old_ack, "")

old_conclusion_end = "The pure-JAX design makes any circuit composable with `jax.grad`, `jax.vmap`, and `jax.pmap` without code modification, providing a productive research tool for the NISQ algorithm development cycle."
new_conclusion_end = old_conclusion_end + " " + old_ack
content = content.replace(old_conclusion_end, new_conclusion_end)

# 6. Append Appendix C
appendix_c_full = f"""

## Appendix C: Preliminary Observations (25 & 27 Qubit GPU Benchmarks)

> [!NOTE]
> The data in this appendix is derived from preliminary hardware sessions on an NVIDIA RTX 2050 (†N=3 runs) and is included as preliminary observations rather than rigorous N=10 benchmark claims.

{appendix_c_content}
"""

content = content + appendix_c_full

with open('research_paper.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Applied fixes via script.")
