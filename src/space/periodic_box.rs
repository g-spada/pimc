use crate::path_state::traits::{WorldLineDimensions, WorldLinePositionAccess};
use ndarray::{Array1, ArrayView1};

pub struct PeriodicBox {
    lengths: Vec<f64>,
}

impl PeriodicBox {
    pub fn new(lengths: Vec<f64>) -> Self {
        Self { lengths }
    }

    pub fn difference<'a, A, B>(self, r1: A, r2: B) -> Array1<f64>
    where
        A: Into<ArrayView1<'a, f64>>,
        B: Into<ArrayView1<'a, f64>>,
    {
        let r1_view = r1.into();
        let r2_view = r2.into();

        // Ensure the shapes are the same
        assert_eq!(
            r1_view.len(),
            r2_view.len(),
            "Arrays must have the same shape"
        );
        assert_eq!(
            r1_view.len(),
            self.lengths.len(),
            "Lengths vector must have the same length as input arrays"
        );

        // Compute the element-wise difference and apply the nearest-image transformation
        (&r1_view - &r2_view)
            .iter()
            .zip(self.lengths)
            .map(|(&x, l)| x - l * (0.5 * (2.0 * x / l).floor()).ceil())
            .collect()
    }

    pub fn fundamental_image<A>(self, r: A) -> Array1<f64>
    where
        A: Into<Array1<f64>>,
    {
        r.into()
            .iter()
            .zip(self.lengths)
            .map(|(&x, l)| x - l * (x / l).floor())
            .collect()
    }

    pub fn reseat_polymer<W>(self, worldlines: &mut W, particle: usize)
    where
        W: WorldLineDimensions + WorldLinePositionAccess,
    {
        let image0 = self.fundamental_image(worldlines.position(particle, 0).to_owned());
        let shift = &worldlines.position(particle, 0) - &image0;
        let t = worldlines.time_slices();
        let raw_dim = worldlines.positions(particle, 0, t).raw_dim();
        worldlines.set_positions(
            particle,
            0,
            t,
            &(&shift.broadcast(raw_dim).unwrap() + &worldlines.positions(particle, 0, t)),
        );
    }
}

#[test]
fn test_periodic_box_difference() {
    use ndarray::array;
    let pbc = PeriodicBox::new([1.0, 1.0, 2.0].to_vec());
    let diff = pbc.difference(&array![0.4, 1.1, 1.8], &array![0.0, 0.0, 0.0]);
    let expected = array![0.4, 0.1, -0.2];
    for i in 0..3 {
        assert!((diff[i] - expected[i]).abs() < 1e-12);
    }
}
