//! Adjacency matrix eigenvalue computation via power iteration.
//!
//! Provides spectral decomposition of the adjacency matrix including
//! dominant eigenvalue and eigenvector computation.

use crate::{Graph, dot, mat_vec, normalize};

/// Result of spectral analysis of a graph's adjacency matrix.
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// Eigenvalues (approximated via power iteration).
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors.
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Compute the dominant eigenvalue and eigenvector using power iteration.
///
/// Returns `(eigenvalue, eigenvector)`.
pub fn dominant_eigenvalue(graph: &Graph, max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
    let n = graph.vertex_count();
    if n == 0 {
        return (0.0, vec![]);
    }
    let mat = graph.adjacency_matrix();
    power_iteration(&mat, max_iter, tol)
}

/// Power iteration for dominant eigenvalue of a matrix.
///
/// Returns `(eigenvalue, eigenvector)`.
pub fn power_iteration(mat: &[Vec<f64>], max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
    let n = mat.len();
    if n == 0 {
        return (0.0, vec![]);
    }
    let mut v = vec![1.0; n];
    normalize(&mut v);

    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let w = mat_vec(mat, &v);
        let new_eigenvalue = dot(&v, &w);
        let norm_w = normalize(&mut {
            let mut w = w;
            // Handle sign
            w
        });

        let diff = (new_eigenvalue - eigenvalue).abs();
        eigenvalue = new_eigenvalue;

        // Update v from the normalized w
        let mut new_v = mat_vec(mat, &v);
        let nv = normalize(&mut new_v);
        if nv < 1e-15 {
            break;
        }

        if diff < tol && norm_w > 0.0 {
            v = new_v;
            break;
        }
        v = new_v;
    }

    // Final eigenvalue computation
    let w = mat_vec(mat, &v);
    eigenvalue = dot(&v, &w);

    (eigenvalue, v)
}

/// Compute the top `k` eigenvalues and eigenvectors using deflation.
pub fn top_k_eigenvalues(graph: &Graph, k: usize, max_iter: usize, tol: f64) -> SpectralResult {
    let n = graph.vertex_count();
    let actual_k = k.min(n);
    let mat = graph.adjacency_matrix();
    let mut deflated = mat.clone();
    let mut eigenvalues = Vec::with_capacity(actual_k);
    let mut eigenvectors = Vec::with_capacity(actual_k);

    for _ in 0..actual_k {
        let (eigval, eigvec) = power_iteration(&deflated, max_iter, tol);
        eigenvalues.push(eigval);
        eigenvectors.push(eigvec.clone());

        // Deflate: remove component of this eigenvector
        let en = normalize(&mut eigvec.clone());
        if en > 1e-15 {
            for i in 0..n {
                for j in 0..n {
                    deflated[i][j] -= eigval * eigvec[i] * eigvec[j];
                }
            }
        }
    }

    SpectralResult {
        eigenvalues,
        eigenvectors,
    }
}

/// Compute the spectral radius (largest absolute eigenvalue) of the adjacency matrix.
pub fn spectral_radius(graph: &Graph) -> f64 {
    let (eigval, _) = dominant_eigenvalue(graph, 1000, 1e-10);
    eigval.abs()
}

/// Compute the trace of the adjacency matrix (sum of diagonal, equals 2×#self-loops for simple graphs).
pub fn adjacency_trace(graph: &Graph) -> f64 {
    let mat = graph.adjacency_matrix();
    (0..graph.vertex_count()).map(|i| mat[i][i]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_empty_graph() {
        let g = Graph::new(0);
        let (eigval, eigvec) = dominant_eigenvalue(&g, 100, 1e-10);
        assert_eq!(eigval, 0.0);
        assert!(eigvec.is_empty());
    }

    #[test]
    fn test_single_vertex() {
        let g = Graph::new(1);
        let (eigval, _) = dominant_eigenvalue(&g, 100, 1e-10);
        assert!((eigval).abs() < 1e-10);
    }

    #[test]
    fn test_path_graph_2() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1, 1.0);
        let (eigval, _) = dominant_eigenvalue(&g, 1000, 1e-10);
        assert!((eigval - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_complete_graph_3() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        let (eigval, _) = dominant_eigenvalue(&g, 1000, 1e-10);
        // Dominant eigenvalue of K3 is 2
        assert!((eigval - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_top_k_eigenvalues() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let result = top_k_eigenvalues(&g, 2, 1000, 1e-10);
        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.eigenvectors.len(), 2);
        // Largest eigenvalue of P3 is sqrt(2)
        assert!((result.eigenvalues[0] - std::f64::consts::SQRT_2).abs() < 0.5, "eigval = {}", result.eigenvalues[0]);
    }

    #[test]
    fn test_spectral_radius() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        g.add_edge(3, 0, 1.0);
        let sr = spectral_radius(&g);
        // Cycle C4 has spectral radius 2
        assert!((sr - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_adjacency_trace() {
        let g = Graph::new(3);
        let t = adjacency_trace(&g);
        assert!((t).abs() < 1e-10);
    }

    #[test]
    fn test_power_iteration_identity() {
        let mat = vec![vec![2.0, 0.0], vec![0.0, 1.0]];
        let (eigval, eigvec) = power_iteration(&mat, 100, 1e-10);
        assert!((eigval - 2.0).abs() < 1e-6);
        assert!((eigvec[0].abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_more_than_n() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1, 1.0);
        let result = top_k_eigenvalues(&g, 5, 100, 1e-10);
        assert_eq!(result.eigenvalues.len(), 2);
    }

    #[test]
    fn test_star_graph() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(0, 3, 1.0);
        let (eigval, _) = dominant_eigenvalue(&g, 1000, 1e-10);
        // Star graph S4: dominant eigenvalue is sqrt(3) ≈ 1.732
        assert!((eigval - 3.0_f64.sqrt()).abs() < 0.5, "eigval = {eigval}");
    }
}
