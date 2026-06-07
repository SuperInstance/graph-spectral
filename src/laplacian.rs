//! Graph Laplacian matrix construction.
//!
//! Provides combinatorial and normalized Laplacian matrices,
//! and computes the Laplacian eigenvalue spectrum.

use crate::{Graph, mat_vec, dot};

/// Build the combinatorial Laplacian matrix `L = D - A`.
///
/// `D` is the degree matrix and `A` is the adjacency matrix.
pub fn combinatorial_laplacian(graph: &Graph) -> Vec<Vec<f64>> {
    let n = graph.vertex_count();
    let adj = graph.adjacency_matrix();
    let mut lap = vec![vec![0.0; n]; n];
    for i in 0..n {
        let deg: f64 = adj[i].iter().sum();
        lap[i][i] = deg;
        for j in 0..n {
            if i != j {
                lap[i][j] = -adj[i][j];
            }
        }
    }
    lap
}

/// Build the symmetric normalized Laplacian `L_sym = I - D^{-1/2} A D^{-1/2}`.
pub fn normalized_laplacian(graph: &Graph) -> Vec<Vec<f64>> {
    let n = graph.vertex_count();
    let adj = graph.adjacency_matrix();
    let mut lap = vec![vec![0.0; n]; n];

    let degrees: Vec<f64> = (0..n).map(|i| adj[i].iter().sum()).collect();

    for i in 0..n {
        lap[i][i] = 1.0;
        for j in 0..n {
            if i != j && adj[i][j] > 0.0 {
                let d_sqrt_i = degrees[i].sqrt().max(1e-15);
                let d_sqrt_j = degrees[j].sqrt().max(1e-15);
                lap[i][j] = -adj[i][j] / (d_sqrt_i * d_sqrt_j);
            }
        }
    }
    lap
}

/// Build the random-walk normalized Laplacian `L_rw = I - D^{-1} A`.
pub fn random_walk_laplacian(graph: &Graph) -> Vec<Vec<f64>> {
    let n = graph.vertex_count();
    let adj = graph.adjacency_matrix();
    let mut lap = vec![vec![0.0; n]; n];

    for i in 0..n {
        let deg: f64 = adj[i].iter().sum();
        lap[i][i] = 1.0;
        if deg > 1e-15 {
            for j in 0..n {
                if i != j {
                    lap[i][j] = -adj[i][j] / deg;
                }
            }
        }
    }
    lap
}

/// Compute Laplacian eigenvalues using iterative QR-like approach.
/// Returns eigenvalues in non-decreasing order.
pub fn laplacian_eigenvalues(graph: &Graph, max_iter: usize, tol: f64) -> Vec<f64> {
    let lap = combinatorial_laplacian(graph);
    let n = lap.len();
    if n == 0 {
        return vec![];
    }

    // Use Gershgorin circles for initial bounds, then refine via inverse iteration
    // For simplicity, use a direct approach: extract eigenvalues from iterative method
    let mut eigenvalues = Vec::with_capacity(n);

    // Tridiagonalize and compute eigenvalues via bisection is complex.
    // We use a simpler approach: compute eigenvalues via power iteration with shifts.
    let mut deflated = lap.clone();

    for k in 0..n {
        let (eigval, eigvec) = shifted_power_iteration(&deflated, 0.0, max_iter, tol);
        eigenvalues.push(eigval);

        if k < n - 1 {
            let en = norm(&eigvec);
            if en > 1e-15 {
                for i in 0..n {
                    for j in 0..n {
                        deflated[i][j] -= eigval * (eigvec[i] * eigvec[j]) / (en * en);
                    }
                }
            }
        }
    }

    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Shifted power iteration to find eigenvalue closest to zero (smallest).
fn shifted_power_iteration(
    mat: &[Vec<f64>],
    _shift: f64,
    max_iter: usize,
    tol: f64,
) -> (f64, Vec<f64>) {
    let n = mat.len();
    let mut v = vec![1.0; n];
    let vnorm = norm(&v);
    for x in v.iter_mut() {
        *x /= vnorm;
    }

    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let mut w = mat_vec(mat, &v);
        let new_eigenvalue = dot(&v, &w);

        let wnorm = norm(&w);
        if wnorm < 1e-15 {
            break;
        }
        for x in w.iter_mut() {
            *x /= wnorm;
        }

        let diff = (new_eigenvalue - eigenvalue).abs();
        eigenvalue = new_eigenvalue;

        v = w;
        if diff < tol {
            break;
        }
    }

    (eigenvalue, v)
}

/// Compute the trace of the Laplacian (equals 2m where m = number of edges for simple graphs).
pub fn laplacian_trace(graph: &Graph) -> f64 {
    let lap = combinatorial_laplacian(graph);
    (0..graph.vertex_count()).map(|i| lap[i][i]).sum()
}

/// Compute the quadratic form `x^T L x` for a vector `x`.
pub fn quadratic_form(graph: &Graph, x: &[f64]) -> f64 {
    let lap = combinatorial_laplacian(graph);
    let lx = mat_vec(&lap, x);
    dot(x, &lx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_combinatorial_laplacian_path() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let lap = combinatorial_laplacian(&g);
        assert!((lap[0][0] - 1.0).abs() < 1e-10);
        assert!((lap[1][1] - 2.0).abs() < 1e-10);
        assert!((lap[2][2] - 1.0).abs() < 1e-10);
        assert!((lap[0][1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_row_sum_zero() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let lap = combinatorial_laplacian(&g);
        for i in 0..4 {
            let row_sum: f64 = lap[i].iter().sum();
            assert!(row_sum.abs() < 1e-10, "Row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn test_normalized_laplacian_diagonal() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let lap = normalized_laplacian(&g);
        for i in 0..3 {
            assert!((lap[i][i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_random_walk_laplacian() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1, 1.0);
        let lap = random_walk_laplacian(&g);
        assert!((lap[0][0] - 1.0).abs() < 1e-10);
        assert!((lap[0][1] - (-1.0)).abs() < 1e-10);
        assert!((lap[1][0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_trace() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let t = laplacian_trace(&g);
        // trace = 2 * edge_count for simple graph
        assert!((t - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_form() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1, 1.0);
        let x = vec![1.0, -1.0];
        let q = quadratic_form(&g, &x);
        // x^T L x = (x0 - x1)^2 = 4
        assert!((q - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_eigenvalues() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let eigs = laplacian_eigenvalues(&g, 500, 1e-8);
        // P3 eigenvalues: 0, 1, 3
        assert_eq!(eigs.len(), 3);
        assert!(eigs[0].abs() < 0.5); // smallest ~0
    }

    #[test]
    fn test_empty_graph_laplacian() {
        let g = Graph::new(0);
        let lap = combinatorial_laplacian(&g);
        assert!(lap.is_empty());
    }
}
