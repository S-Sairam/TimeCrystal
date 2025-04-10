# ⏳ Continuous Time Crystal Simulation — Guided Learning & Replication

This project is an **educational replication** of the paper:

> **"Continuous time crystal in an electron-nuclear spin system: stability and melting of periodic auto-oscillations"**  
> *A. Greilich et al., TU Dortmund, Ioffe Institute (2023)*

---

## 🧠 Objective

To **learn the physics behind continuous time crystals** step-by-step, while **gradually building a simulation** in Python (using QuTiP), based on the experimental and theoretical setup described in the paper.

This is not a one-shot simulation. Instead, this repo documents:

- The **theory** of each concept needed
- How it's **applied in the Greilich paper**
- The corresponding **Python code**, with hints and minimal scaffolding

---

## 🧰 Libraries Used

- [`QuTiP`](https://qutip.org) – Quantum Toolbox in Python (for simulating open quantum systems)
- `NumPy`, `Matplotlib` – numerical computation and plotting
- Optionally: `SciPy` for Fourier analysis (FFT)

---

## 🧱 Learning Flow

Each module in this repo follows this structure:

1. **Concept** — concise explanation of a key quantum topic
2. **Relevance** — where this appears in the Greilich paper
3. **Your Turn** — coding prompt with hints (you do the coding)
4. **Optional Discussion** — deeper insight or curiosity links

---

## ✅ Topics Covered / Roadmap

- 🔜 Qubits and Spin-1/2 systems
- 🔜 Time evolution under a Hamiltonian
- 🔜 Constructing Hamiltonians for driven spin systems
- 🔜 Interaction with external fields (Bx, Bz, angle α)
- 🔜 Optical pumping and periodic driving
- 🔜 Emergence of time-crystalline order
- 🔜 Simulating spin dynamics and magnetization

---

## 📍 Goal

To replicate the spin oscillations and **ω/2 periodicity** observed in Greilich et al., and build a simulation framework that captures the essence of **continuous time crystals** — all while learning the theory behind it.

---

## 📚 Reference

Greilich, A., Kopteva, N. E., Kamenskii, A. N., Sokolov, P. S., Korenev, V. L., & Bayer, M. (2023).  
**Continuous time crystal in an electron-nuclear spin system: stability and melting of periodic auto-oscillations**  
['arXiv'](https://arxiv.org/pdf/2303.15989)

---
