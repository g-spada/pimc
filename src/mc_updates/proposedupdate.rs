use std::collections::{BTreeSet, HashMap};
use std::ops::Range;

/// A data structure to represent and manage proposed modifications
/// to particle worldlines in a Monte Carlo simulation.
///
/// This structure stores modifications for a set of particles over
/// ranges of timeslices. Each modification includes the indices of
/// affected particles and their respective new properties (e.g., positions or states).
/// The structure allows for efficient access to these modifications
/// and provides precomputed lists of all modified particles and timeslices.
///
/// # Type Parameters
/// - `T`: The type of the properties being modified (e.g., `Vec<f64>` for positions).
///
/// # Efficiency
/// - **Particle Modifications:** Stored in a `HashMap` for efficient lookup.
/// - **Timeslices and Particles Union:** Cached in `BTreeSet` for sorted retrieval.
///
/// # Examples
/// ```
/// use pimc_rs::mc_updates::proposedupdate::ProposedUpdate;
/// let mut modification = ProposedUpdate::new();
///
/// // Add a modification for particle 0 over timeslices 1 to 4
/// modification.add_modification(
///     0,
///     1..4,
///     vec![[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]],
/// );
///
/// // Retrieve all modified timeslices
/// let timeslices = modification.get_modified_timeslices();
/// assert_eq!(timeslices, vec![1, 2, 3]);
///
/// // Retrieve modifications for a specific particle and timeslice
/// if let Some(value) = modification.get_modification(0, 2) {
///     println!("Modification for particle 0 at timeslice 2: {:?}", value);
/// }
/// ```
pub struct ProposedUpdate<T> {
    /// Map of particle indices to their respective modifications.
    ///
    /// Each entry maps a particle index to a vector of:
    /// - `Range<usize>`: The range of modified timeslices for the particle.
    /// - `Vec<T>`: The new properties corresponding to the modified timeslices.
    modifications: HashMap<usize, Vec<(Range<usize>, Vec<T>)>>,
    /// Cached set of all modified timeslices, stored in sorted order.
    modified_timeslices: BTreeSet<usize>,
    /// Cached set of all modified particle indices, stored in sorted order.
    modified_particles: BTreeSet<usize>,
}

impl<T> ProposedUpdate<T> {
    /// Creates a new, empty `ProposedUpdate`.
    ///
    /// # Examples
    /// ```
    /// use pimc_rs::mc_updates::proposedupdate::ProposedUpdate;
    /// let modification = ProposedUpdate::<Vec<f64>>::new();
    /// assert!(modification.get_modified_particles().is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            modifications: HashMap::new(),
            modified_timeslices: BTreeSet::new(),
            modified_particles: BTreeSet::new(),
        }
    }

    /// Adds a modification for a specific particle over a range of timeslices.
    ///
    /// # Parameters
    /// - `particle`: The index of the particle being modified.
    /// - `range`: The range of timeslices being modified (`start..end`).
    /// - `properties`: A vector of new properties for the specified timeslices.
    ///
    /// # Panics
    /// Panics if the length of `properties` does not match the length of `range`.
    pub fn add_modification(&mut self, particle: usize, range: Range<usize>, properties: Vec<T>) {
        // Ensure the range and properties length match
        assert_eq!(range.end - range.start, properties.len());

        // Update the modifications map
        self.modifications
            .entry(particle)
            .or_insert_with(Vec::new)
            .push((range.clone(), properties));

        // Update the cached modified particles and timeslices
        self.modified_particles.insert(particle);
        self.modified_timeslices.extend(range);
    }

    /// Retrieves all modified timeslices as a sorted list.
    ///
    /// # Returns
    /// A vector containing all modified timeslices in ascending order.
    pub fn get_modified_timeslices(&self) -> Vec<usize> {
        self.modified_timeslices.iter().copied().collect()
    }

    /// Retrieves all modified particle indices as a sorted list.
    ///
    /// # Returns
    /// A vector containing all modified particle indices in ascending order.
    pub fn get_modified_particles(&self) -> Vec<usize> {
        self.modified_particles.iter().copied().collect()
    }

    /// Retrieves the modification for a specific particle and timeslice.
    ///
    /// # Parameters
    /// - `particle`: The index of the particle being queried.
    /// - `timeslice`: The timeslice being queried.
    ///
    /// # Returns
    /// An `Option` containing a reference to the modified property, or `None` if no modification exists.
    pub fn get_modification(&self, particle: usize, timeslice: usize) -> Option<&T> {
        self.modifications.get(&particle).and_then(|ranges| {
            for (range, properties) in ranges {
                if range.contains(&timeslice) {
                    return Some(&properties[timeslice - range.start]);
                }
            }
            None
        })
    }
}
