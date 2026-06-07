# graph-spectral

Spectral graph theory library for Rust. Pure `std` — no external dependencies.

## Features

- **Spectrum analysis** — Dominant eigenvalue/eigenvector via power iteration, top-k eigenvalues with deflation, spectral radius
- **Laplacian matrices** — Combinatorial (`L = D - A`), normalized (`L_sym = I - D^{-1/2} A D^{-1/2}`), random-walk (`L_rw = I - D^{-1} A`)
- **Spectral clustering** — Bisection and k-way clustering via Fiedler vector, normalized cut, ratio cut, modularity
- **Cheeger constant** — Approximation via Fiedler sweep, exact computation for small graphs, edge expansion, conductance
- **Fiedler vector** — Algebraic connectivity, connectivity testing, spectral gap, sign-based partitioning

## Usage

```rust
use graph_spectral::{Graph, fiedler, clustering, cheeger, spectrum};

// Create a graph
let mut g = Graph::new(5);
g.add_edge(0, 1, 1.0);
g.add_edge(1, 2, 1.0);
g.add_edge(2, 3, 1.0);
g.add_edge(3, 4, 1.0);

// Algebraic connectivity
let ac = fiedler::algebraic_connectivity(&g, 1000, 1e-10);
println!("Algebraic connectivity: {ac}");

// Spectral bisection
let result = clustering::spectral_bisection(&g, 1000, 1e-10);
println!("Cut size: {}", result.cut_size);

// Cheeger constant approximation
let (lower, upper) = cheeger::cheeger_from_fiedler(&g, 1000, 1e-10);
println!("Cheeger bounds: [{lower}, {upper}]");

// Dominant eigenvalue
let (eigval, eigvec) = spectrum::dominant_eigenvalue(&g, 1000, 1e-10);
println!("Dominant eigenvalue: {eigval}");
```

## License

MIT
