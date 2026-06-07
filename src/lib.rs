//! # graph-spectral
//!
//! Spectral graph theory library providing eigenvalue computation, spectral clustering,
//! Cheeger constant approximation, algebraic connectivity, Fiedler vector computation,
//! and graph partitioning algorithms.
//!
//! ## Modules
//!
//! - [`spectrum`] — Adjacency and Laplacian eigenvalue computation via power iteration
//! - [`laplacian`] — Graph Laplacian matrix construction (combinatorial and normalized)
//! - [`clustering`] — Spectral clustering using Fiedler vector
//! - [`cheeger`] — Cheeger constant approximation and edge expansion
//! - [`fiedler`] — Fiedler vector and algebraic connectivity

pub mod spectrum;
pub mod laplacian;
pub mod clustering;
pub mod cheeger;
pub mod fiedler;

/// A simple graph represented as an adjacency list.
#[derive(Clone, Debug)]
pub struct Graph {
    /// Number of vertices.
    n: usize,
    /// Adjacency list: `adj[i]` contains `(neighbor, weight)`.
    adj: Vec<Vec<(usize, f64)>>,
    /// Whether the graph is directed.
    directed: bool,
}

impl Graph {
    /// Create a new undirected graph with `n` vertices and no edges.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![vec![]; n],
            directed: false,
        }
    }

    /// Create a new directed graph with `n` vertices and no edges.
    pub fn new_directed(n: usize) -> Self {
        Self {
            n,
            adj: vec![vec![]; n],
            directed: true,
        }
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.n
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        let count: usize = self.adj.iter().map(|v| v.len()).sum();
        if self.directed { count } else { count / 2 }
    }

    /// Add an undirected edge `(u, v)` with optional weight.
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        assert!(u < self.n && v < self.n, "Vertex index out of bounds");
        self.adj[u].push((v, weight));
        if !self.directed && u != v {
            self.adj[v].push((u, weight));
        }
    }

    /// Get the adjacency list.
    pub fn adjacency(&self) -> &[Vec<(usize, f64)>] {
        &self.adj
    }

    /// Get the degree of vertex `v`.
    pub fn degree(&self, v: usize) -> f64 {
        self.adj[v].iter().map(|(_, w)| w).sum()
    }

    /// Get the neighbors of vertex `v`.
    pub fn neighbors(&self, v: usize) -> &[(usize, f64)] {
        &self.adj[v]
    }

    /// Compute the adjacency matrix as a flat row-major `n × n` matrix.
    pub fn adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let mut mat = vec![vec![0.0; self.n]; self.n];
        for (u, row) in mat.iter_mut().enumerate() {
            for &(v, w) in &self.adj[u] {
                row[v] += w;
            }
        }
        mat
    }
}

/// Internal: matrix-vector multiplication.
pub(crate) fn mat_vec(mat: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Internal: vector dot product.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Internal: vector norm.
pub(crate) fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Internal: normalize a vector in place. Returns the norm before normalization.
pub(crate) fn normalize(v: &mut [f64]) -> f64 {
    let n = norm(v);
    if n > 1e-15 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    n
}
