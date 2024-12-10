use super::accepted_update::AcceptedUpdate;
use ndarray::Array2;
use std::collections::{BTreeSet, HashMap};
use std::ops::Range;

/// Represents a modification to a range of timeslices for a single particle.
type Modification<T> = (Range<usize>, Array2<T>);

/// A collection of modifications for a single particle.
type ParticleModifications<T> = Vec<Modification<T>>;

/// A map of particle indices to their respective modifications.
type ModificationsMap<T> = HashMap<usize, ParticleModifications<T>>;

/// A data structure to represent and manage proposed modifications to
/// particle worldlines in a Monte Carlo simulation.
///
/// This structure allows storing and managing modifications to particle positions
/// over ranges of timeslices. Each modification is associated with a specific particle
/// and includes the range of timeslices it affects along with the corresponding
/// positions in an `Array2<T>`.
///
/// The structure tracks:
/// - The list of all modified particles.
/// - The union of all modified timeslices.
/// - Modifications for each particle as a collection of `(Range<usize>, Array2<T>)`.
///
/// # Type Parameters
/// - `T`: The type of the properties being modified (e.g., `f64` for positions).
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use std::ops::Range;
/// use pimc_rs::updates::proposed_update::ProposedUpdate;
///
/// let mut updates = ProposedUpdate::new();
///
/// // Add position modifications for particle 1
/// let positions = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// updates.add_position_modification(1, 0..4, positions);
///
/// // Check the modified particles and timeslices
/// assert_eq!(updates.get_modified_particles(), vec![1]);
/// assert_eq!(updates.get_modified_timeslices(), vec![0, 1, 2, 3]);
///
/// // Retrieve modifications for particle 1
/// let modifications = updates.get_modifications(1).unwrap();
/// assert_eq!(modifications.len(), 1);
/// ```
#[derive(Debug)]
pub struct ProposedUpdate<T> {
    /// Maps particle indices to their respective modifications.
    modifications: ModificationsMap<T>,

    /// Cached set of all modified timeslices, stored in sorted order.
    modified_timeslices: BTreeSet<usize>,

    /// Cached set of all modified particle indices, stored in sorted order.
    modified_particles: BTreeSet<usize>,
}

#[allow(clippy::new_without_default)]
impl<T> ProposedUpdate<T> {
    /// Creates a new, empty `ProposedUpdate` instance.
    ///
    /// # Examples
    /// ```
    /// use pimc_rs::updates::proposed_update::ProposedUpdate;
    ///
    /// let updates: ProposedUpdate<f64> = ProposedUpdate::new();
    /// assert!(updates.get_modified_particles().is_empty());
    /// assert!(updates.get_modified_timeslices().is_empty());
    /// ```
    pub fn new() -> Self {
        ProposedUpdate {
            modifications: HashMap::new(),
            modified_timeslices: BTreeSet::new(),
            modified_particles: BTreeSet::new(),
        }
    }

    /// Adds a position modification for a specific particle and timeslice range.
    ///
    /// This method updates the proposed positions for a particle over a specified range
    /// of timeslices. Each particle can have multiple non-overlapping modifications.
    /// Modifications with overlapping timeslice ranges for the same particle index are not allowed.
    /// This behavior enforces strict separation of ranges to ensure modifications
    /// are unambiguous and easily retrievable.
    ///
    /// # Parameters
    /// - `particle_index`: The index of the particle being modified.
    /// - `range`: The range of timeslices affected by the modification.
    /// - `positions`: A 2D array where each row corresponds to a timeslice in the range,
    ///   and each column corresponds to a spatial coordinate.
    ///
    /// # Panics
    /// - Panics if the number of rows in `positions` does not match the length of the `range`.
    /// - Panics if the `range` overlaps with any existing range for the same `particle_index`.
    ///
    /// # Examples
    /// ```
    /// use ndarray::array;
    /// use pimc_rs::updates::proposed_update::ProposedUpdate;
    ///
    /// let mut updates = ProposedUpdate::new();
    /// let positions = array![[1.0, 2.0], [3.0, 4.0]];
    ///
    /// // Add a modification for particle 1 over the range 0..2
    /// updates.add_position_modification(1, 0..2, positions);
    /// assert_eq!(updates.get_modified_particles(), vec![1]);
    /// assert_eq!(updates.get_modified_timeslices(), vec![0, 1]);
    /// ```
    pub fn add_position_modification<A>(
        &mut self,
        particle_index: usize,
        range: Range<usize>,
        positions: A,
    ) where
        A: Into<Array2<T>> + Clone,
    {
        let positions: Array2<T> = positions.into();

        assert_eq!(
            positions.nrows(),
            range.len(),
            "Number of rows in positions must match the range length"
        );

        // Check for overlapping ranges
        if let Some(existing_modifications) = self.modifications.get(&particle_index) {
            for (existing_range, _) in existing_modifications {
                let no_overlap =
                    range.start >= existing_range.end || range.end <= existing_range.start;
                assert!(
                    no_overlap,
                    "Overlapping range detected for particle {}: {:?} overlaps with {:?}",
                    particle_index, range, existing_range
                );
            }
        }

        // If no overlap, add the modification
        self.modified_particles.insert(particle_index);
        let entry = self.modifications.entry(particle_index).or_default();
        entry.push((range.clone(), positions));

        for timeslice in range {
            self.modified_timeslices.insert(timeslice);
        }
    }

    /// Retrieves all modifications for a given particle.
    ///
    /// # Parameters
    /// - `particle_index`: The index of the particle.
    ///
    /// # Returns
    /// - `Some(&Vec<Modification<T>>)` if the particle has modifications.
    /// - `None` if the particle has no modifications.
    ///
    /// # Examples
    /// ```
    /// use ndarray::array;
    /// use pimc_rs::updates::proposed_update::ProposedUpdate;
    ///
    /// let mut updates = ProposedUpdate::new();
    /// let positions = array![[1.0, 2.0], [3.0, 4.0]];
    /// updates.add_position_modification(1, 0..2, positions);
    ///
    /// let modifications = updates.get_modifications(1).unwrap();
    /// assert_eq!(modifications.len(), 1);
    /// ```
    pub fn get_modifications(&self, particle_index: usize) -> Option<&Vec<Modification<T>>> {
        self.modifications.get(&particle_index)
    }

    /// Retrieves the list of all modified timeslices.
    ///
    /// # Returns
    /// A vector containing all modified timeslices.
    ///
    /// # Examples
    /// ```
    /// use ndarray::array;
    /// use pimc_rs::updates::proposed_update::ProposedUpdate;
    ///
    /// let mut updates = ProposedUpdate::new();
    /// let positions = array![[1.0, 2.0], [3.0, 4.0]];
    /// updates.add_position_modification(1, 0..2, positions);
    ///
    /// assert_eq!(updates.get_modified_timeslices(), vec![0, 1]);
    /// ```
    pub fn get_modified_timeslices(&self) -> Vec<usize> {
        self.modified_timeslices.iter().cloned().collect()
    }

    /// Retrieves the list of all modified particle indices.
    ///
    /// # Returns
    /// A vector containing all modified particle indices.
    ///
    /// # Examples
    /// ```
    /// use ndarray::array;
    /// use pimc_rs::updates::proposed_update::ProposedUpdate;
    ///
    /// let mut updates = ProposedUpdate::new();
    /// let positions = array![[1.0, 2.0], [3.0, 4.0]];
    /// updates.add_position_modification(1, 0..2, positions);
    ///
    /// assert_eq!(updates.get_modified_particles(), vec![1]);
    /// ```
    pub fn get_modified_particles(&self) -> Vec<usize> {
        self.modified_particles.iter().cloned().collect()
    }

    /// Converts a `ProposedUpdate<T>` into a `SimplifiedUpdate` by extracting
    /// only the `Range<usize>` from each modification, along with the
    /// `modified_timeslices` and `modified_particles`.
    pub fn to_accepted_update(&self) -> AcceptedUpdate {
        let mut accepted_modifications = HashMap::new();

        // Extract modifications for each particle
        for (particle_index, particle_modifications) in &self.modifications {
            let ranges: Vec<Range<usize>> = particle_modifications
                .iter()
                .map(|(range, _)| range.clone()) // Extract only the range
                .collect();
            accepted_modifications.insert(*particle_index, ranges);
        }

        // Create and return the simplified update
        AcceptedUpdate {
            modifications: accepted_modifications,
            modified_timeslices: self.modified_timeslices.clone(), // Copy the timeslices
            modified_particles: self.modified_particles.clone(),   // Copy the particles
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::collections::{BTreeSet, HashMap};
    use std::ops::Range;

    #[test]
    fn test_new() {
        let updates: ProposedUpdate<f64> = ProposedUpdate::new();
        assert!(updates.get_modified_particles().is_empty());
        assert!(updates.get_modified_timeslices().is_empty());
    }

    #[test]
    fn test_add_position_modification() {
        let mut updates = ProposedUpdate::new();
        let positions = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        updates.add_position_modification(1, 0..3, positions);

        let particles = updates.get_modified_particles();
        assert_eq!(particles, vec![1]);

        let timeslices = updates.get_modified_timeslices();
        assert_eq!(timeslices, vec![0, 1, 2]);

        let modifications = updates.get_modifications(1).unwrap();
        assert_eq!(modifications.len(), 1);
        assert_eq!(modifications[0].0, 0..3);
    }

    #[test]
    fn test_multiple_modifications() {
        let mut updates = ProposedUpdate::new();
        let positions1 = array![[1.0, 2.0], [3.0, 4.0]];
        let positions2 = array![[5.0, 6.0], [7.0, 8.0]];
        updates.add_position_modification(1, 0..2, positions1);
        updates.add_position_modification(2, 2..4, positions2);

        let particles = updates.get_modified_particles();
        assert_eq!(particles, vec![1, 2]);

        let timeslices = updates.get_modified_timeslices();
        assert_eq!(timeslices, vec![0, 1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "Number of rows in positions must match the range length")]
    fn test_invalid_position_modification() {
        let mut updates = ProposedUpdate::new();
        let positions = array![[1.0, 2.0]]; // Only 1 row
        updates.add_position_modification(1, 0..2, positions); // Range length is 2
    }

    #[test]
    #[should_panic(expected = "Overlapping range detected for particle 1: 4..8 overlaps with 2..5")]
    fn test_overlapping_ranges_panic() {
        let mut updates = ProposedUpdate::new();
        let positions1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let positions2 = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]];

        // Add a modification for range 2..5
        updates.add_position_modification(1, 2..5, positions1);

        // Adding an overlapping modification for range 4..8 should panic
        updates.add_position_modification(1, 4..8, positions2);
    }

    #[test]
    fn test_to_accepted_update() {
        // Step 1: Create a sample ProposedUpdate
        let mut modifications = HashMap::new();
        modifications.insert(
            0,
            vec![
                (0..3, Array2::zeros((3, 3))),
                (8..10, Array2::zeros((2, 3))),
            ],
        );
        modifications.insert(
            1,
            vec![
                (2..5, Array2::zeros((3, 3))),
                (10..12, Array2::zeros((2, 3))),
            ],
        );

        let modified_timeslices = BTreeSet::from([0, 1, 2, 3, 4, 5, 8, 9, 10, 11]);
        let modified_particles = BTreeSet::from([0, 1]);

        let proposed_update = ProposedUpdate::<f64> {
            modifications,
            modified_timeslices: modified_timeslices.clone(),
            modified_particles: modified_particles.clone(),
        };

        // Step 2: Convert to SimplifiedUpdate
        let accepted_update = proposed_update.to_accepted_update();

        // Step 3: Validate the SimplifiedUpdate
        // Validate modifications
        let expected_modifications: HashMap<usize, Vec<Range<usize>>> =
            HashMap::from([(0, vec![0..3, 8..10]), (1, vec![2..5, 10..12])]);
        assert_eq!(accepted_update.modifications, expected_modifications);

        // Validate modified_timeslices
        assert_eq!(accepted_update.modified_timeslices, modified_timeslices);

        // Validate modified_particles
        assert_eq!(accepted_update.modified_particles, modified_particles);
    }
}
