// src/utils/mod.rs

pub mod linear_algebra;

// Re-export specific functions for easier access
pub use linear_algebra::{
    build_laplacian_operator,
    build_advection_operator,
    build_parameter_identity,
};
