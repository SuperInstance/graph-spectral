//! Cheeger constant approximation and edge expansion.
//!
//! Provides tools for estimating the Cheeger constant (isoperimetric number)
//! and edge expansion of a graph.

use crate::{Graph, laplacian};

/// Result of Cheeger constant computation.
#[derive(Debug, Clone)]
pub struct CheegerResult {
    /// Approximate Cheeger constant.
    pub cheeger_constant: f64,
    /// The partition achieving the Cheeger constant: (set S, complement).
    pub partition: (Vec<usize>, Vec<usize>),
}

/// Approximate the Cheeger constant using the Fiedler value.
///
/// Uses the relation: `λ₂/2 ≤ h(G) ≤ √(2λ₂)` where `λ₂` is the algebraic connectivity.
pub fn cheeger_from_fiedler(graph: &Graph, max_iter: usize, tol: f64) -> (f64, f64) {
    let eigs = laplacian::laplacian_eigenvalues(graph, max_iter, tol);
    if eigs.len() < 2 {
        return (0.0, f64::INFINITY);
    }
    let lambda2 = eigs[1].max(0.0);
    (lambda2 / 2.0, (2.0 * lambda2).sqrt())
}

/// Compute the exact edge expansion for a given subset S.
///
/// `h(S) = |∂S| / min(|S|, |V\S|)` where `∂S` is the edge boundary.
pub fn edge_expansion(graph: &Graph, subset: &[usize]) -> f64 {
    let n = graph.vertex_count();
    if subset.is_empty() || subset.len() >= n {
        return 0.0;
    }

    let in_set: Vec<bool> = {
        let mut s = vec![false; n];
        for &v in subset {
            s[v] = true;
        }
        s
    };

    let mut boundary = 0;
    for &u in subset {
        for &(v, _) in graph.neighbors(u) {
            if !in_set[v] {
                boundary += 1;
            }
        }
    }

    let vol = subset.len().min(n - subset.len());
    (boundary as f64) / (vol as f64)
}

/// Approximate the Cheeger constant by trying all Fiedler-vector-based sweeps.
///
/// Sorts vertices by Fiedler vector value and tries each prefix as a cut.
pub fn approximate_cheeger(graph: &Graph, max_iter: usize, tol: f64) -> CheegerResult {
    let n = graph.vertex_count();
    if n <= 1 {
        return CheegerResult {
            cheeger_constant: 0.0,
            partition: ((0..n).collect(), vec![]),
        };
    }

    // Get Fiedler vector via Laplacian eigenvector computation
    let fiedler_vec = compute_fiedler_for_cheeger(graph, max_iter, tol);

    // Sort vertices by Fiedler vector value
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        fiedler_vec[a]
            .partial_cmp(&fiedler_vec[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut best_h = f64::INFINITY;
    let mut best_k = 1;

    for k in 1..n {
        let subset: Vec<usize> = order[..k].to_vec();
        let h = edge_expansion(graph, &subset);
        if h < best_h {
            best_h = h;
            best_k = k;
        }
    }

    let set_s: Vec<usize> = order[..best_k].to_vec();
    let complement: Vec<usize> = order[best_k..].to_vec();

    CheegerResult {
        cheeger_constant: best_h,
        partition: (set_s, complement),
    }
}

/// Compute the isoperimetric number `h(G)` by brute force over all subsets (exponential, small graphs only).
pub fn exact_cheeger(graph: &Graph) -> f64 {
    let n = graph.vertex_count();
    if n <= 1 || n > 20 {
        return 0.0;
    }

    let mut best_h = f64::INFINITY;
    let total_subsets = 1usize << n;

    for mask in 1..total_subsets {
        let subset: Vec<usize> = (0..n).filter(|&i| (mask >> i) & 1 == 1).collect();
        if subset.is_empty() || subset.len() >= n {
            continue;
        }
        let h = edge_expansion(graph, &subset);
        best_h = best_h.min(h);
    }

    if best_h.is_infinite() { 0.0 } else { best_h }
}

/// Compute the conductance of a subset S.
///
/// `φ(S) = |∂S| / min(vol(S), vol(V\S))` where vol is the sum of degrees.
pub fn conductance(graph: &Graph, subset: &[usize]) -> f64 {
    let n = graph.vertex_count();
    if subset.is_empty() || subset.len() >= n {
        return 0.0;
    }

    let in_set: Vec<bool> = {
        let mut s = vec![false; n];
        for &v in subset {
            s[v] = true;
        }
        s
    };

    let mut boundary = 0;
    let mut vol_s = 0.0;
    let mut vol_comp = 0.0;

    for u in 0..n {
        let deg = graph.degree(u);
        if in_set[u] {
            vol_s += deg;
        } else {
            vol_comp += deg;
        }
        for &(v, _) in graph.neighbors(u) {
            if in_set[u] && !in_set[v] {
                boundary += 1;
            }
        }
    }

    let min_vol = vol_s.min(vol_comp);
    if min_vol < 1e-15 {
        return 0.0;
    }
    (boundary as f64) / min_vol
}

fn compute_fiedler_for_cheeger(graph: &Graph, max_iter: usize, tol: f64) -> Vec<f64> {
    let n = graph.vertex_count();
    let lap = laplacian::combinatorial_laplacian(graph);
    let ones = vec![1.0 / (n as f64).sqrt(); n];
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64) - (n as f64 - 1.0) / 2.0).collect();
    crate::normalize(&mut v);
    let dot_ones = crate::dot(&v, &ones);
    for (i, vi) in v.iter_mut().enumerate() {
        *vi -= dot_ones * ones[i];
    }
    crate::normalize(&mut v);

    for _ in 0..max_iter {
        let mut w = crate::mat_vec(&lap, &v);
        let d = crate::dot(&w, &ones);
        for i in 0..n {
            w[i] -= d * ones[i];
        }
        let wn = crate::normalize(&mut w);
        if wn < 1e-15 {
            break;
        }
        let diff: f64 = v.iter().zip(w.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        v = w;
        if diff < tol {
            break;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_edge_expansion_single_vertex() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let h = edge_expansion(&g, &[0]);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_edge_expansion_middle() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let h = edge_expansion(&g, &[1]);
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_edge_expansion_empty() {
        let g = Graph::new(3);
        let h = edge_expansion(&g, &[]);
        assert!((h).abs() < 1e-10);
    }

    #[test]
    fn test_cheeger_bounds() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let (lower, upper) = cheeger_from_fiedler(&g, 500, 1e-8);
        assert!(lower >= 0.0);
        assert!(upper >= lower);
    }

    #[test]
    fn test_approximate_cheeger_path() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let result = approximate_cheeger(&g, 500, 1e-8);
        assert!(result.cheeger_constant > 0.0);
        assert!(!result.partition.0.is_empty());
        assert!(!result.partition.1.is_empty());
    }

    #[test]
    fn test_exact_cheeger_path3() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let h = exact_cheeger(&g);
        // P3: best cut is {0,1} vs {2} or {0} vs {1,2}, expansion = 1/1 = 1
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exact_cheeger_complete() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        let h = exact_cheeger(&g);
        // K3: any single vertex has 2 boundary edges, expansion = 2/1 = 2
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_conductance() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let phi = conductance(&g, &[0, 1]);
        assert!(phi > 0.0);
    }
}
