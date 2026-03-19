# Distributed Optimization in Multilayer Networks via Tensor-Laplacian Dynamics

<p align="center">
  <a href="https://doi.org/10.1109/JSYST.2025.3640067">
    <img src="https://img.shields.io/badge/IEEE%20Systems%20Journal-10.1109%2FJSYST.2025.3640067-blue?style=flat-square&logo=ieee" alt="IEEE Paper"/>
  </a>
  <img src="https://img.shields.io/badge/Julia-1.8%2B-9558B2?style=flat-square&logo=julia" alt="Julia"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Published-brightgreen?style=flat-square" alt="Status"/>
</p>

---

## 📄 Paper

**Title:** Distributed Optimization in Multilayer Networks via Tensor-Laplacian Dynamics

**Authors:** Christian D. Rodríguez-Camargo, Andrés F. Urquijo-Rodríguez, Eduardo Mojica-Nava

**Journal:** IEEE Systems Journal (2025)

**DOI:** [10.1109/JSYST.2025.3640067](https://doi.org/10.1109/JSYST.2025.3640067)

> **Abstract:** Multilayer networks provide a more advanced and comprehensive framework for modeling real-world systems compared to traditional single-layer and multiplex networks. In this article, we generalize previously developed results for distributed optimization in multiplex networks to the more general case of multilayer networks by employing a tensor formalism to represent multilayer networks and their tensor-Laplacian diffusion dynamics. This generalized framework removes the need for replica nodes, allowing variability in both topology and number of nodes across layers. We derive the multilayer combinatorial Laplacian tensor and extend the distributed gradient descent algorithm. Numerical examples validate our approach, and we explore the impact of heterogeneous layer topologies and complex interlayer dynamics on consensus time, underscoring their implications for real-world multilayer systems.

> **Note:** This work is a direct generalization of our previous results for multiplex networks [[IEEE Trans. Control Netw. Syst., 2025](https://doi.org/10.1109/TCNS.2024.3467018)]. The companion repository for the multiplex case is available [here](https://github.com/<your-username>/multiplex-distributed-optimization).

---

## 🗂️ Repository Structure

```
.
├── README.md
├── LICENSE
├── Project.toml                          # Julia package dependencies
├── Manifest.toml
├── Multilayers.jl                        # Core multilayer saddle-point dynamics (Figs. 3, 5, 9)
├── Multilayers_consensus_time_map.jl     # 2D consensus time surface sweep (Fig. 7)
└── Switching_multilayer.jl               # Time-varying (switching) topology dynamics (Fig. 10)
```

---

## 🔬 Overview

This repository contains the Julia code reproducing all numerical experiments in the paper. The implementation generalizes distributed consensus-based optimization from multiplex to fully general **multilayer networks** using a tensor-Laplacian formalism, where layers may have different numbers of nodes and non-one-to-one interlayer connections.

### Key innovations over the multiplex case

- **No replica nodes required:** layers may have heterogeneous node sets and asymmetric interlayer connections.
- **Tensor-Laplacian** $\mathcal{L}^{\alpha\tilde{\mu}}_{\beta\tilde{\nu}}$ replaces the supra-Laplacian, encoding both intra- and interlayer diffusion in a fourth-order structure.
- **NetworkX-driven graph assembly:** the multilayer graph is built declaratively via edge lists, and its Laplacian is extracted through the Python `networkx` library via `PyCall`.

### Algorithms implemented

**Distributed saddle-point dynamics** (`Multilayers.jl` → `RHS!`, `solve_dynamics`, `solve_dynamics_ss`)

The second-order ODE system solved is identical in structure to the multiplex case, but now the operator $\mathcal{L}$ is the full tensor-Laplacian of the multilayer graph:

$$\dot{y} = v, \qquad \dot{v} = -(\mathcal{L} + \mathbb{I}) v - \mathcal{L}^2 y$$

Two integration modes are available:

- `solve_dynamics(L; tf)` — fixed-horizon integration with **RK4**
- `solve_dynamics_ss(L)` — integration terminated when $\sum_{i<j}|y_i - y_j| \leq 10^{-2}$ (consensus detection via `DiscreteCallback`), using **DP5** (Dormand-Prince)

**2D consensus time surface** (`Multilayers_consensus_time_map.jl` → `main`)

Produces the 3D landscapes of consensus time $t_C( D^{\alpha,\tilde{\mu}}_{\beta,\tilde{\nu}},\, D^{\alpha',\tilde{\mu}'}_{\beta',\tilde{\nu}'})$ by sweeping two selected interlayer diffusion constants over $[0.1, 1.0]^2$ on a $\text{num} \times \text{num}$ grid, with all others fixed. Results are exported to `.dat` files for plotting.

**Time-varying (switching) topology** (`Switching_multilayer.jl`)

Simulates consensus dynamics on a 2-layer multilayer network ($N_1 = 3$ ring, $N_2 = 5$ random Erdős–Rényi graph) whose interlayer Laplacian $L_{\text{inter}}(t)$ is redrawn every $\Delta t = 2.5$ time units. The total Laplacian at time $t$ is:

$$L(t) = \begin{bmatrix} L_1 & 0 \\ 0 & L_2 \end{bmatrix} + D \cdot L_{\text{inter}}(t), \qquad D = 0.4$$

Solved with **DP5** over $t \in [0, 20]$. Three independent runs (panels a, b, c of Fig. 10) are produced and exported to `switching_multilayer.pdf` via `pyx`.

---

## ⚙️ Installation

### Prerequisites

- [Julia](https://julialang.org/downloads/) ≥ 1.8
- Python environment with the following packages:
  ```
  networkx matplotlib pyx
  ```
  Install via: `pip install networkx matplotlib pyx`
- (Optional) A LaTeX distribution for `LaTeXStrings`-rendered axis labels

### Setup

```bash
git clone https://github.com/alfagalileo/Consensus-Multilayers.git
cd multilayer-tensor-optimization
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

After instantiation, configure `PyCall` to point to your Python environment:

```julia
julia --project=.
using PyCall
# Verify networkx is accessible
pyimport("networkx")
```

---

## 🚀 Usage

### Multilayer saddle-point dynamics

`Multilayers.jl` exposes a declarative API. A multilayer network is defined by three dictionaries and solved in three steps:

```julia
include("Multilayers.jl")

# 1. Define the network
nodes_per_layer = [3, 5]

Edges_Intra = Dict()
Edges_Intra[1,1] = [[1,2], [2,3], [3,1]]           # ring in layer 1
Edges_Intra[2,2] = [[i, i%5+1] for i in 1:5]       # ring in layer 2

Edges_Inter = Dict()
Edges_Inter[1,2] = [[1,2], [3,4]]                   # asymmetric interlayer links

Dx = Dict((1,2) => 0.5)                             # interlayer diffusion constant

# 2. Build the tensor-Laplacian matrix
L = make_structure(nodes_per_layer, Edges_Intra, Edges_Inter, Dx; verbose=false)

# 3. Solve
sol = solve_dynamics(L; tf=300.0)       # fixed-horizon (RK4)
# sol = solve_dynamics_ss(L)            # run until consensus (DP5 + callback)

# 4. Plot / export
params = Dict(:nodes_per_layer => nodes_per_layer)
plot_solution(params, sol)   # exports plot.svg and plot.eps via pyx
export_data(sol)             # exports data.csv
```

**Preconfigured scenarios in `main()`:**

| Scenario (uncomment in `main()`) | Layers | Nodes | Reproduces |
|---|---|---|---|
| Multiplex 2-layer (ring + complete) | 2 | 5+5 | Fig. 9 |
| Heterogeneous 2-layer | 2 | 3+5 | Fig. 3 |
| 4-layer heterogeneous | 4 | 3+4+5+6 | Fig. 5 |

Run:
```bash
julia --project=. Multilayers.jl
```

### 2D consensus time surface

Reproduces the 3D surface plots of Fig. 7 by sweeping two selected interlayer diffusion constants:

```julia
include("Multilayers_consensus_time_map.jl")

# Sweep D^{1→2} and D^{3→5} (indices 1 and 2 in pairs_inter), others fixed at 0.5
Tc = main(which=[1,2], fixed_val=0.5, num=20)
# Result exported to: Dx[1, 2]_fixed=0.5.dat
```

Function signature:

```julia
main(; which=[1,2], fixed_val=0.2, num=20)
# which     – indices (1-6) of the two interlayer pairs to sweep
# fixed_val – value of all other D^{α→β} constants
# num       – grid resolution (num × num points)
```

Run all 8 panels of Fig. 7 by varying `which` over the 8 pairs listed in the paper caption.

```bash
julia --project=. --threads=auto Multilayers_consensus_time_map.jl
```

### Time-varying (switching) topology

Reproduces the three panels of Fig. 10:

```bash
julia --project=. Switching_multilayer.jl
```

Or from the REPL:

```julia
include("Switching_multilayer.jl")
plot_solution()   # runs 3 independent ODE solutions, exports switching_multilayer.pdf
```

Key constants (defined at top of file):

```julia
const N1 = 3          # nodes in layer 1 (ring graph)
const N2 = 5          # nodes in layer 2 (random graph, p=0.4)
const T_max = 20.0    # total integration time
const Δt_switch = 2.5 # topology switching period
const D = 0.4         # interlayer diffusion strength
Random.seed!(1234)    # fixed seed for reproducibility of random graph draws
```

---

## 📊 Reproducing Paper Figures

| Figure | Description | File | Entry point |
|--------|-------------|------|-------------|
| Fig. 3 | Consensus dynamics, 2-layer heterogeneous ($N_{L_1}=3$, $N_{L_2}=5$, $D_x^{1\to2}=0.5$) | `Multilayers.jl` | `main()` — 2nd scenario |
| Fig. 5 | Consensus dynamics, 4-layer ($N=3,4,5,6$) | `Multilayers.jl` | `main()` — 3rd scenario (uncomment) |
| Fig. 6 | Consensus time $t_C$ vs. single $D_x$ (1D sweep) | `Multilayers.jl` | `main()` — sweep block (uncomment) |
| Fig. 7 | Consensus time surfaces (2D sweeps, 8 panels) | `Multilayers_consensus_time_map.jl` | `main(which=..., fixed_val=0.5, num=20)` |
| Fig. 9 | Multiplex validation (2 layers, 5 replica nodes, $D_x^{1\to2}=0.4$) | `Multilayers.jl` | `main()` — 1st scenario (uncomment) |
| Fig. 10 | Time-varying topology consensus (3 panels) | `Switching_multilayer.jl` | `plot_solution()` |

---

## 📐 Key Parameters

| Symbol | Variable in code | Description |
|--------|-----------------|-------------|
| $N^{(M)}$ | `nodes_per_layer` | Vector of node counts per layer |
| $w_{ij}(\tilde{h}\tilde{k})$ | `Edges_Intra`, `Edges_Inter` | Intra- and interlayer edge lists |
| $D^{[\alpha,\beta]}$ | `Dx` | Dict of interlayer diffusion constants |
| $D$ | `D` | Uniform interlayer strength (switching case) |
| $\Delta t$ | `Δt_switch` | Topology switching period |
| $\epsilon$ | convergence threshold | $\sum_{i<j}\|y_i-y_j\|\leq 10^{-2}$ |
| `num` | grid resolution | Points per axis in 2D $t_C$ surface |

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{rodriguez2025distributed,
  title   = {Distributed Optimization in Multilayer Networks via
             Tensor-{Laplacian} Dynamics},
  author  = {Rodr{\'i}guez-Camargo, Christian D. and
             Urquijo-Rodr{\'i}guez, Andr{\'e}s F. and
             Mojica-Nava, Eduardo},
  journal = {IEEE Systems Journal},
  year    = {2025},
  doi     = {10.1109/JSYST.2025.3640067}
}
```

If you also use the companion multiplex code, please additionally cite:

```bibtex
@article{rodriguez2025consensus,
  title   = {Consensus-based Distributed Optimization for Multi-agent Systems
             over Multiplex Networks},
  author  = {Rodr{\'i}guez-Camargo, Christian D. and
             Urquijo-Rodr{\'i}guez, Andres F. and
             Mojica-Nava, Eduardo},
  journal = {IEEE Transactions on Control of Network Systems},
  volume  = {12},
  number  = {1},
  pages   = {1040--1051},
  year    = {2025},
  doi     = {10.1109/TCNS.2024.3467018}
}
```

---

## 🤝 Acknowledgments

This work was supported in part by:

- **EPSRC** (Grants No. EP/R513143/1 and No. EP/T517793/1)
- **Uniminuto** VIII Convocatoria para el Desarrollo y Fortalecimiento de los Grupos de Investigación (code C119-173)
- **Uniminuto** Convocatoria de investigación para prototipado de tecnologías que promueven el cuidado o la restauración del medioambiente (code CPT123-200-5220)
- Industrial Engineering Program, Corporación Universitaria Minuto de Dios (Uniminuto), Colombia

---

## 👥 Authors

| Name | Affiliation | Contact |
|------|-------------|---------|
| **Christian D. Rodríguez-Camargo** | AMOPP Group, University College London & PAAS-UN, Universidad Nacional de Colombia | christian.rodriguez-camargo.21@ucl.ac.uk |
| **Andrés F. Urquijo-Rodríguez** | GIII-ECCI, Universidad ECCI & Universidad Nacional de Colombia | afurquijor@unal.edu.co |
| **Eduardo Mojica-Nava** *(corresponding)* | Dept. of Electrical and Electronics Engineering & PAAS-UN, Universidad Nacional de Colombia | eamojican@unal.edu.co |

---

## 📬 Contact

For questions about the code or the paper, please open an [issue](../../issues) or contact the corresponding author above.

---

## 📃 License

This project is released under the [MIT License](LICENSE).
