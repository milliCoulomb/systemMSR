// src/utils/linear_algebra.rs

use nalgebra::{DMatrix, DVector};

/// Builds the Laplacian operator matrix for a 1D diffusion problem on a non-uniform grid.
///
/// The Laplacian operator is constructed using finite difference approximations
/// tailored for non-uniform cell spacings. Each diagonal element accounts for the
/// diffusion coefficients and cell spacings of adjacent cells.
///
/// # Arguments
///
/// * `diffusion` - A vector of diffusion coefficients for each cell.
/// * `delta_x` - A vector of cell spacings (\(\Delta x_i\)) for each cell.
///
/// # Returns
///
/// * A tridiagonal Laplacian operator matrix of size \(N \times N\),
///   where \(N\) is the number of cells.
pub fn build_laplacian_operator(diffusion: &DVector<f64>, delta_x: &DVector<f64>) -> DMatrix<f64> {
    let n = diffusion.len();
    assert_eq!(delta_x.len(), n, "diffusion and delta_x must be the same length");
    let mut laplacian = DMatrix::zeros(n, n);

    for i in 0..n {
        if i == 0 {
            // Forward difference for the first cell (Neumann or Dirichlet boundary)
            let dx_forward = delta_x[i];
            let D_forward = diffusion[i];
            laplacian[(i, i)] = -D_forward / dx_forward;
            laplacian[(i, i + 1)] = D_forward / dx_forward;
        } else if i == n - 1 {
            // Backward difference for the last cell (Neumann or Dirichlet boundary)
            let dx_backward = delta_x[i - 1];
            let D_backward = diffusion[i - 1];
            laplacian[(i, i - 1)] = D_backward / dx_backward;
            laplacian[(i, i)] = -D_backward / dx_backward;
        } else {
            // Central difference for interior cells
            let dx_backward = delta_x[i - 1];
            let dx_forward = delta_x[i];
            let D_backward = diffusion[i - 1];
            let D_forward = diffusion[i];

            // Coefficients based on harmonic mean
            let a = D_backward / (dx_backward * (dx_backward + dx_forward));
            let c = D_forward / (dx_forward * (dx_backward + dx_forward));
            let b = -(a + c);

            laplacian[(i, i - 1)] = a;
            laplacian[(i, i)] = b;
            laplacian[(i, i + 1)] = c;
        }
    }

    laplacian
}

/// Builds the Advection operator matrix for a 1D advection problem using an upwind scheme on a non-uniform grid.
///
/// The Advection operator accounts for varying cell spacings and velocities,
/// ensuring stability and accuracy in advection-dominated scenarios.
///
/// # Arguments
///
/// * `velocity` - A vector of velocities for each cell.
/// * `delta_x` - A vector of cell spacings (\(\Delta x_i\)) for each cell.
///
/// # Returns
///
/// * A bidiagonal Advection operator matrix of size \(N \times N\),
///   where \(N\) is the number of cells.
pub fn build_advection_operator(velocity: &DVector<f64>, delta_x: &DVector<f64>) -> DMatrix<f64> {
    let n = velocity.len();
    assert_eq!(delta_x.len(), n, "velocity and delta_x must be the same length");
    let mut advection = DMatrix::zeros(n, n);

    for i in 0..n {
        if i == 0 {
            // Forward difference for the first cell (no upwind cell)
            let dx_forward = delta_x[i];
            let v_forward = velocity[i];
            advection[(i, i)] = v_forward / dx_forward;
            advection[(i, i + 1)] = -v_forward / dx_forward;
        } else {
            // Upwind scheme: use the velocity of the current cell
            let dx = delta_x[i];
            let v = velocity[i];
            advection[(i, i - 1)] = v / dx;
            advection[(i, i)] = -v / dx;
        }
    }

    advection
}

/// Builds a Parameterized Identity matrix where each diagonal element is a specified parameter.
///
/// This matrix is useful for scaling equations or representing coefficients in diagonal form.
///
/// # Arguments
///
/// * `parameters` - A vector of parameters to place on the diagonal.
///
/// # Returns
///
/// * An Identity matrix of size \(N \times N\) with the given parameters on the diagonal,
///   where \(N\) is the number of parameters.
pub fn build_parameter_identity(parameters: &DVector<f64>) -> DMatrix<f64> {
    let n = parameters.len();
    let mut identity = DMatrix::zeros(n, n);

    for i in 0..n {
        identity[(i, i)] = parameters[i];
    }

    identity
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_build_laplacian_operator_uniform_grid() {
        let diffusion = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let delta_x = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let laplacian = build_laplacian_operator(&diffusion, &delta_x);

        let expected = DMatrix::from_row_slice(3, 3, &[
            -1.0, 1.0, 0.0,
             1.0, -2.0, 1.0,
             0.0, 1.0, -1.0,
        ]);

        assert_eq!(laplacian, expected);
    }

    #[test]
    fn test_build_laplacian_operator_non_uniform_grid() {
        let diffusion = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let delta_x = DVector::from_vec(vec![1.0, 2.0, 1.0]); // Δx0=1, Δx1=2, Δx2=1
        let laplacian = build_laplacian_operator(&diffusion, &delta_x);

        let expected = DMatrix::from_row_slice(3, 3, &[
            -1.0, 1.0, 0.0,
             1.0, -1.5, 0.75,
             0.0, 1.5, -1.5,
        ]);

        // Allow some tolerance due to floating point arithmetic
        for i in 0..3 {
            for j in 0..3 {
                assert!((laplacian[(i, j)] - expected[(i, j)]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_build_advection_operator_uniform_grid() {
        let velocity = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let delta_x = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let advection = build_advection_operator(&velocity, &delta_x);

        let expected = DMatrix::from_row_slice(3, 3, &[
            1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
        ]);

        assert_eq!(advection, expected);
    }

    #[test]
    fn test_build_advection_operator_non_uniform_grid() {
        let velocity = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let delta_x = DVector::from_vec(vec![1.0, 2.0, 1.0]); // Δx0=1, Δx1=2, Δx2=1
        let advection = build_advection_operator(&velocity, &delta_x);

        let expected = DMatrix::from_row_slice(3, 3, &[
            1.0 / 1.0, -1.0 / 1.0, 0.0,
            2.0 / 2.0, -2.0 / 2.0, 0.0,
            3.0 / 1.0, -3.0 / 1.0, 0.0,
        ]);

        // Allow some tolerance due to floating point arithmetic
        for i in 0..3 {
            for j in 0..3 {
                assert!((advection[(i, j)] - expected[(i, j)]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_build_parameter_identity() {
        let parameters = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        let identity = build_parameter_identity(&parameters);

        let expected = DMatrix::from_row_slice(3, 3, &[
            4.0, 0.0, 0.0,
            0.0, 5.0, 0.0,
            0.0, 0.0, 6.0,
        ]);

        assert_eq!(identity, expected);
    }
}