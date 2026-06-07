//! Spectral clustering using the Fiedler vector.
//!
//! Partitions a graph into clusters based on spectral properties
//! of the Laplacian matrix.

use crate::{Graph, laplacian, mat_vec, dot, normalize};

/// Result of spectral clustering.
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Cluster assignment for each vertex.
    pub assignments: Vec<usize>,
    /// Number of clusters.
    pub num_clusters: usize,
    /// Cut size (number of edges between clusters).
    pub cut_size: usize,
}

/// Perform spectral bisection into 2 clusters using the Fiedler vector.
///
/// Vertices with positive Fiedler vector values go to cluster 0,
/// negative values to cluster 1.
pub fn spectral_bisection(graph: &Graph, max_iter: usize, tol: f64) -> ClusteringResult {
    let n = graph.vertex_count();
    if n == 0 {
        return ClusteringResult {
            assignments: vec![],
            num_clusters: 0,
            cut_size: 0,
        };
    }

    let fiedler = compute_fiedler_vector(graph, max_iter, tol);
    let median = median_value(&fiedler);

    let assignments: Vec<usize> = fiedler
        .iter()
        .map(|v| if *v >= median { 0 } else { 1 })
        .collect();

    let cut_size = count_cut_edges(graph, &assignments);

    ClusteringResult {
        assignments,
        num_clusters: 2,
        cut_size,
    }
}

/// Perform spectral clustering into `k` clusters using recursive bisection.
pub fn spectral_k_clustering(graph: &Graph, k: usize, max_iter: usize, tol: f64) -> ClusteringResult {
    let n = graph.vertex_count();
    if n == 0 || k == 0 {
        return ClusteringResult {
            assignments: vec![],
            num_clusters: 0,
            cut_size: 0,
        };
    }

    if k == 1 {
        return ClusteringResult {
            assignments: vec![0; n],
            num_clusters: 1,
            cut_size: 0,
        };
    }

    let mut assignments = vec![0usize; n];
    let mut num_clusters = 1;

    // Recursive bisection
    for _ in 1..k {
        if num_clusters >= k {
            break;
        }
        // Find the largest cluster to split
        let cluster_sizes: Vec<usize> = (0..num_clusters)
            .map(|c| assignments.iter().filter(|&&a| a == c).count())
            .collect();
        let split_cluster = cluster_sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, &s)| s)
            .map(|(i, _)| i)
            .unwrap_or(0);

        if cluster_sizes[split_cluster] <= 1 {
            break;
        }

        // Build subgraph of the cluster to split
        let vertices: Vec<usize> = (0..n).filter(|&i| assignments[i] == split_cluster).collect();
        let _idx_map = vertices.to_vec();
        let sub_n = vertices.len();

        let mut subgraph = Graph::new(sub_n);
        let mut reverse_map = vec![0usize; n];
        for (si, &vi) in vertices.iter().enumerate() {
            reverse_map[vi] = si;
        }

        for (si, &vi) in vertices.iter().enumerate() {
            for &(vj, w) in graph.neighbors(vi) {
                if vertices.contains(&vj) {
                    let sj = reverse_map[vj];
                    subgraph.add_edge(si, sj, w);
                }
            }
        }

        let result = spectral_bisection(&subgraph, max_iter, tol);

        let new_cluster_id = num_clusters;
        for (si, &vi) in vertices.iter().enumerate() {
            if result.assignments[si] == 1 {
                assignments[vi] = new_cluster_id;
            }
        }
        num_clusters += 1;
    }

    let cut_size = count_cut_edges(graph, &assignments);

    ClusteringResult {
        assignments,
        num_clusters,
        cut_size,
    }
}

/// Compute the normalized cut size for a given partition.
pub fn normalized_cut(graph: &Graph, assignments: &[usize]) -> f64 {
    let n = graph.vertex_count();
    let num_clusters = assignments.iter().copied().max().unwrap_or(0) + 1;

    let mut cut = 0.0_f64;
    let mut vol = vec![0.0_f64; num_clusters];

    for u in 0..n {
        vol[assignments[u]] += graph.degree(u);
        for &(v, _) in graph.neighbors(u) {
            if u < v && assignments[u] != assignments[v] {
                cut += 1.0;
            }
        }
    }

    let total_vol: f64 = vol.iter().sum();
    if total_vol < 1e-15 {
        return 0.0;
    }

    let mut ncut = 0.0;
    for vol_c in vol.iter().take(num_clusters) {
        if *vol_c > 1e-15 {
            ncut += cut / *vol_c;
        }
    }
    ncut
}

/// Compute the ratio cut for a partition.
pub fn ratio_cut(graph: &Graph, assignments: &[usize]) -> f64 {
    let n = graph.vertex_count();
    let num_clusters = assignments.iter().copied().max().unwrap_or(0) + 1;
    let mut cut = 0;
    let mut sizes = vec![0usize; num_clusters];

    for u in 0..n {
        sizes[assignments[u]] += 1;
        for &(v, _) in graph.neighbors(u) {
            if u < v && assignments[u] != assignments[v] {
                cut += 1;
            }
        }
    }

    let mut rc = 0.0;
    for size in sizes.iter().take(num_clusters) {
        if *size > 0 {
            rc += (cut as f64) / (*size as f64);
        }
    }
    rc
}

/// Compute modularity of a partition.
pub fn modularity(graph: &Graph, assignments: &[usize]) -> f64 {
    let n = graph.vertex_count();
    let m = graph.edge_count() as f64;
    if m < 1e-15 {
        return 0.0;
    }

    let degrees: Vec<f64> = (0..n).map(|v| graph.degree(v)).collect();
    let mut q = 0.0;

    for u in 0..n {
        for &(v, _w) in graph.neighbors(u) {
            if assignments[u] == assignments[v] {
                q += 1.0 - degrees[u] * degrees[v] / (2.0 * m);
            }
        }
    }

    q / (2.0 * m)
}

// --- Internal helpers ---

fn compute_fiedler_vector(graph: &Graph, max_iter: usize, tol: f64) -> Vec<f64> {
    let lap = laplacian::combinatorial_laplacian(graph);
    let n = graph.vertex_count();
    if n <= 1 {
        return vec![0.0; n];
    }

    // Find eigenvector for second-smallest eigenvalue via inverse iteration with deflation
    let ones = vec![1.0 / (n as f64).sqrt(); n];
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64) - (n as f64 - 1.0) / 2.0).collect();
    normalize(&mut v);

    // Remove component along ones vector
    let dot_ones = dot(&v, &ones);
    for (i, vi) in v.iter_mut().enumerate() {
        *vi -= dot_ones * ones[i];
    }
    normalize(&mut v);

    // Iterate: apply Laplacian, keep orthogonal to ones
    for _ in 0..max_iter {
        let mut w = mat_vec(&lap, &v);

        // Re-orthogonalize against ones
        let dot_ones = dot(&w, &ones);
        for i in 0..n {
            w[i] -= dot_ones * ones[i];
        }

        let wn = normalize(&mut w);
        if wn < 1e-15 {
            break;
        }

        // Check convergence
        let diff: f64 = v.iter().zip(w.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        v = w;
        if diff < tol {
            break;
        }
    }

    v
}

fn median_value(v: &[f64]) -> f64 {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    sorted[mid]
}

fn count_cut_edges(graph: &Graph, assignments: &[usize]) -> usize {
    let n = graph.vertex_count();
    let mut count = 0;
    for u in 0..n {
        for &(v, _) in graph.neighbors(u) {
            if u < v && assignments[u] != assignments[v] {
                count += 1;
            }
        }
    }
    count
}
