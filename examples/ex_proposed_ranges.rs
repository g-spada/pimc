use ndarray::{array, s};
use pimc::updates::proposed_update::ProposedUpdate;

fn main() {
    let mut updates = ProposedUpdate::new();

    // Example: Add position modifications for particle 1
    let positions = array![
        [1.0, 2.0], // Position at timeslice 0
        [2.0, 3.0], // Position at timeslice 1
        [3.0, 4.0], // Position at timeslice 2
    ];
    updates.add_position_modification(1, 0..3, positions);

    // Retrieve modified particles and timeslices
    let particles = updates.get_modified_particles();
    let timeslices = updates.get_modified_timeslices();

    println!("Modified particles: {:?}", particles);
    println!("Modified timeslices: {:?}", timeslices);

    // Get position modifications for particle 1
    if let Some(mods) = updates.get_modifications(1) {
        println!("Position modifications for particle 1: {:?}", mods);
    }

    // Create an 8x2 array of positions
    let positions2 = array![
        [1.0, 2.0], // Row 0
        [2.0, 3.0], // Row 1
        [3.0, 4.0], // Row 2
        [4.0, 5.0], // Row 3
        [5.0, 6.0], // Row 4
        [6.0, 7.0], // Row 5
        [7.0, 8.0], // Row 6
        [8.0, 9.0], // Row 7
    ];

    // Add the first half (rows 0 to 3 included) for particle 1, range 3..7
    let first_half = positions2.slice(s![0..4, ..]).to_owned(); // Slice rows 0 to 3, all columns
    updates.add_position_modification(1, 3..7, first_half);

    // Add the second half (rows 4 to 7) for particle 2, range 4..8
    let second_half = positions2.slice(s![4..8, ..]).to_owned(); // Slice rows 4 to 7, all columns
    updates.add_position_modification(2, 4..8, second_half);

    // Debug the stored modifications
    println!(
        "Particle 1 modifications: {:?}",
        updates.get_modifications(1)
    );
    println!(
        "Particle 2 modifications: {:?}",
        updates.get_modifications(2)
    );

    // Create another instance of ProposedUpdate instance
    let mut updates2 = ProposedUpdate::new();

    // Create a 4x2 array of positions
    let positions2 = array![
        [1.0, 2.0, -2.0], // Row 0
        [3.0, 4.0, -4.0], // Row 1
        [5.0, 6.0, -6.0], // Row 2
        [7.0, 8.0, -6.0], // Row 3
    ];

    // Extract the first column
    let first_column = positions2.slice(s![.., 0..1]).to_owned(); // Convert to an owned Array2

    // Add the column as a modification
    updates2.add_position_modification(1, 0..4, first_column);

    // Verify the modification
    if let Some(modifications) = updates2.get_modifications(1) {
        for (range, pos) in modifications {
            println!("Range: {:?}, Positions:\n{:?}", range, pos);
        }
    }
}
