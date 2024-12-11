#[cfg(test)]
mod tests {
    use crate::path_state::sector::Sector;
    use crate::path_state::traits::{
        WorldLineDimensions, WorldLinePermutationAccess, WorldLinePositionAccess,
        WorldLineWormAccess,
    };
    use ndarray::{s, array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

    /// Mock struct for a 2D polymer worldline.
    pub struct MockPolymer {
        positions: Array3<f64>, // Shape: (particles, time slices, spatial dimensions)
        preceding: [Option<usize>; 2], // Fixed size: 2 particles
        following: [Option<usize>; 2], // Fixed size: 2 particles
        worm_head: Option<usize>,
        worm_tail: Option<usize>,
        sector: Sector,
    }

    impl MockPolymer {
        /// Create a new mock polymer.
        pub fn new() -> Self {
            Self {
                positions: Array3::zeros((2, 10, 2)), // 2 particles, 10 time slices, 2D
                preceding: [None; 2],
                following: [None; 2],
                worm_head: None,
                worm_tail: None,
                sector: Sector::Z, // Default sector
            }
        }
    }

    impl WorldLineDimensions for MockPolymer {
        const TIME_SLICES: usize = 10;
        const SPATIAL_DIMENSIONS: usize = 2;
    }

    impl WorldLinePositionAccess for MockPolymer {
        fn particles(&self) -> usize {
            2
        }

        fn position(&self, particle: usize, time_slice: usize) -> ArrayView1<f64> {
            self.positions.slice(s![particle, time_slice, ..])
        }

        fn position_mut(&mut self, particle: usize, time_slice: usize) -> ArrayViewMut1<f64> {
            self.positions.slice_mut(s![particle, time_slice, ..])
        }

        fn set_position(&mut self, particle: usize, time_slice: usize, bead_position: &[f64]) {
            assert_eq!(bead_position.len(), 2, "Position must have 2 dimensions");
            self.positions
                .slice_mut(s![particle, time_slice, ..])
                .assign(&Array1::from(bead_position.to_vec()));
        }

        fn positions(
            &self,
            particle: usize,
            start_slice: usize,
            end_slice: usize,
        ) -> ArrayView2<f64> {
            self.positions
                .slice(s![particle, start_slice..end_slice, ..])
        }

        fn positions_mut(
            &mut self,
            particle: usize,
            start_slice: usize,
            end_slice: usize,
        ) -> ArrayViewMut2<f64> {
            self.positions
                .slice_mut(s![particle, start_slice..end_slice, ..])
        }

        fn set_positions(
            &mut self,
            particle: usize,
            start_slice: usize,
            end_slice: usize,
            positions: &Array2<f64>,
        ) {
            assert_eq!(
                positions.shape(),
                &[end_slice - start_slice, 2],
                "Invalid positions shape"
            );
            self.positions
                .slice_mut(s![particle, start_slice..end_slice, ..])
                .assign(positions);
        }
    }

    impl WorldLinePermutationAccess for MockPolymer {
        fn preceding(&self, particle: usize) -> Option<usize> {
            self.preceding[particle]
        }

        fn following(&self, particle: usize) -> Option<usize> {
            self.following[particle]
        }
    }

    impl WorldLineWormAccess for MockPolymer {
        fn worm_head(&self) -> Option<usize> {
            self.worm_head
        }

        fn worm_tail(&self) -> Option<usize> {
            self.worm_tail
        }

        fn sector(&self) -> Sector {
            self.sector.clone()
        }
    }

    #[test]
    fn test_mock_polymer_positions() {
        let mut polymer = MockPolymer::new();

        // Set and retrieve a position
        polymer.set_position(0, 0, &[1.0, 2.0]);
        let pos = polymer.position(0, 0);
        assert_eq!(pos.to_vec(), vec![1.0, 2.0]);

        // Set and retrieve multiple positions
        let new_positions = array![[3.0, 4.0], [5.0, 6.0]];
        polymer.set_positions(0, 1, 3, &new_positions);
        let positions = polymer.positions(0, 1, 3);
        assert_eq!(positions, new_positions);
    }

    #[test]
    fn test_mock_polymer_permutations() {
        let mut polymer = MockPolymer::new();
        polymer.preceding[0] = Some(1);
        polymer.following[1] = Some(0);

        assert_eq!(polymer.preceding(0), Some(1));
        assert_eq!(polymer.following(1), Some(0));
    }

    #[test]
    fn test_mock_polymer_worm() {
        let mut polymer = MockPolymer::new();
        polymer.worm_head = Some(0);
        polymer.worm_tail = Some(1);
        polymer.sector = Sector::G;

        assert_eq!(polymer.worm_head(), Some(0));
        assert_eq!(polymer.worm_tail(), Some(1));
        assert_eq!(polymer.sector(), Sector::G);
    }
}
