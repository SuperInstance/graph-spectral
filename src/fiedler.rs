//! Fiedler vector and algebraic connectivity.
//!
//! Provides computation of the Fiedler vector (eigenvector corresponding
//! to the second-smallest Laplacian eigenvalue) and algebraic connectivity.

use crate::{Graph, laplacian, mat_vec, dot, normalize};

/// Result of Fiedler vector computation.
#[derive(Debug, Clone)]
pub struct FiedlerResult {
    /// Algebraic connectivity (second-smallest Laplacian eigenvalue).
    pub algebraic_connectivity: f64,
    /// Fiedler vector (eigenvector for the second-smallest eigenvalue).
    pub fiedler_vector: Vec<f64>,
}

/// Compute the Fiedler vector and algebraic connectivity.
///
/// The Fiedler vector is the eigenvector corresponding to the second-smallest
/// eigenvalue of the graph Laplacian. It provides a spectral embedding for
/// graph partitioning.
pub fn fiedler(graph: &Graph, max_iter: usize, tol: f64) -> FiedlerResult {
    let n = graph.vertex_count();
    if n <= 1 {
        return FiedlerResult {
            algebraic_connectivity: 0.0,
            fiedler_vector: vec![0.0; n],
        };
    }

    let lap = laplacian::combinatorial_laplacian(graph);
    let ones = vec![1.0 / (n as f64).sqrt(); n];

    // Initialize with a vector orthogonal to ones
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64) - (n as f64 - 1.0) / 2.0).collect();
    normalize(&mut v);
    let d = dot(&v, &ones);
    for (i, vi) in v.iter_mut().enumerate() {
        *vi -= d * ones[i];
    }
    normalize(&mut v);

    // Power iteration on Laplacian (projecting away from ones each step)
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let mut w = mat_vec(&lap, &v);

        // Re-orthogonalize against ones
        let d = dot(&w, &ones);
        for i in 0..n {
            w[i] -= d * ones[i];
        }

        let new_eigenvalue = dot(&v, &w);
        let wn = normalize(&mut w);
        if wn < 1e-15 {
            break;
        }

        let diff = (new_eigenvalue - eigenvalue).abs();
        eigenvalue = new_eigenvalue;
        v = w;

        if diff < tol {
            break;
        }
    }

    FiedlerResult {
        algebraic_connectivity: eigenvalue.max(0.0),
        fiedler_vector: v,
    }
}

/// Compute only the algebraic connectivity (second-smallest Laplacian eigenvalue).
pub fn algebraic_connectivity(graph: &Graph, max_iter: usize, tol: f64) -> f64 {
    fiedler(graph, max_iter, tol).algebraic_connectivity
}

/// Compute the Fiedler vector normalized to unit length.
pub fn normalized_fiedler(graph: &Graph, max_iter: usize, tol: f64) -> Vec<f64> {
    let result = fiedler(graph, max_iter, tol);
    result.fiedler_vector
}

/// Check if a graph is connected using the Fiedler value.
///
/// A graph is connected if and only if the algebraic connectivity > 0.
pub fn is_connected(graph: &Graph) -> bool {
    algebraic_connectivity(graph, 500, 1e-8) > 1e-6
}

/// Compute the number of zero eigenvalues of the Laplacian (= number of connected components).
pub fn connected_component_count(graph: &Graph, max_iter: usize, tol: f64) -> usize {
    let eigs = laplacian::laplacian_eigenvalues(graph, max_iter, tol);
    eigs.iter().filter(|&&e| e.abs() < tol.max(0.1)).count().max(1)
}

/// Compute the spectral gap (difference between two largest eigenvalues of adjacency matrix).
pub fn spectral_gap(graph: &Graph, max_iter: usize, tol: f64) -> f64 {
    let eigs = laplacian::laplacian_eigenvalues(graph, max_iter, tol);
    if eigs.len() < 2 {
        return 0.0;
    }
    // Spectral gap of Laplacian: λ₂ - λ₁ (but λ₁ = 0 for connected)
    eigs[1] - eigs[0]
}

/// Partition the graph using the Fiedler vector (sign-based bisection).
///
/// Returns two vertex sets based on the sign of the Fiedler vector values.
pub fn fiedler_partition(graph: &Graph, max_iter: usize, tol: f64) -> (Vec<usize>, Vec<usize>) {
    let result = fiedler(graph, max_iter, tol);
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    for (i, &v) in result.fiedler_vector.iter().enumerate() {
        if v >= 0.0 {
            positive.push(i);
        } else {
            negative.push(i);
        }
    }
    (positive, negative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_fiedler_path_graph() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let result = fiedler(&g, 1000, 1e-10);
        // P3: algebraic connectivity = 1
        assert!((result.algebraic_connectivity - 1.0).abs() < 0.2);
        assert_eq!(result.fiedler_vector.len(), 3);
    }

    #[test]
    fn test_algebraic_connectivity_complete() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                g.add_edge(i, j, 1.0);
            }
        }
        let ac = algebraic_connectivity(&g, 1000, 1e-10);
        // K4: algebraic connectivity = 4
        assert!(ac > 2.0);
    }

    #[test]
    fn test_fiedler_single_vertex() {
        let g = Graph::new(1);
        let result = fiedler(&g, 100, 1e-10);
        assert!((result.algebraic_connectivity).abs() < 1e-10);
    }

    #[test]
    fn test_is_connected_path() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert!(is_connected(&g));
    }

    #[test]
    fn test_is_not_connected() {
        let g = Graph::new(4); // no edges at all
        assert!(!is_connected(&g), "should be disconnected");
    }

    #[test]
    fn test_fiedler_partition() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let (pos, neg) = fiedler_partition(&g, 1000, 1e-10);
        assert!(!pos.is_empty());
        assert!(!neg.is_empty());
        assert_eq!(pos.len() + neg.len(), 4);
    }

    #[test]
    fn test_spectral_gap() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let gap = spectral_gap(&g, 500, 1e-8);
        assert!(gap >= 0.0);
    }

    #[test]
    fn test_normalized_fiedler() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let v = normalized_fiedler(&g, 1000, 1e-10);
        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((nrm - 1.0).abs() < 1e-6);
    }
}
