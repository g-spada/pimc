use pimc_rs::mc_updates::proposedupdate::ProposedUpdate;
use pimc_rs::path_state::particlestate::ParticleState;
fn main() {
    //////////////////////////////////////////////////////
    // Test the ProposedUpdate Struct for position updates
    //////////////////////////////////////////////////////
    let mut pos_update = ProposedUpdate::new();

    // Add a modification for particle 0 over timeslices 1 to 4
    pos_update.add_modification(
        0,
        1..4,
        vec![[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]],
    );

    // Add another modification for particle 0 at timeslice 8
    pos_update.add_modification(0, 8..9, vec![[8.0, -8.0, 8.0]]);

    // Add another modification for particle 1
    pos_update.add_modification(1, 3..5, vec![[2.0, 3.0, 4.0], [2.1, 3.1, 4.1]]);

    // Add another modification for particle 5 at timeslice 3
    pos_update.add_modification(5, 3..4, vec![[3.0, -4.0, 3.0]]);

    // Get all modified timeslices
    let timeslices = pos_update.get_modified_timeslices();
    println!("Modified timeslices: {:?}", timeslices);

    // Get all modified particles
    let particles = pos_update.get_modified_particles();
    println!("Modified particles: {:?}", particles);

    // Access a specific modification
    if let Some(value) = pos_update.get_modification(0, 2) {
        println!("Modification for particle 0 at timeslice 2: {:?}", value);
    }

    //////////////////////////////////////////////////////
    // Test the ProposedUpdate Struct for state updates
    //////////////////////////////////////////////////////
    let mut state_update = ProposedUpdate::new();

    // Add a modification for particle 0 over timeslices 1 to 4
    state_update.add_modification(
        0,
        1..4,
        vec![ParticleState::Up, ParticleState::Up, ParticleState::Dn],
    );

    // Add modifications for particle states
    state_update.add_modification(
        0,    // Particle index
        1..4, // Timeslices
        vec![
            // Proposed states
            ParticleState::Up,
            ParticleState::Dn,
            ParticleState::Up,
        ],
    );

    state_update.add_modification(
        1,    // Another particle index
        2..5, // Timeslices
        vec![ParticleState::Dn, ParticleState::Up, ParticleState::Dn],
    );

    // Retrieve and print all modified timeslices
    let timeslices = state_update.get_modified_timeslices();
    println!("Modified timeslices: {:?}", timeslices);

    // Retrieve and print all modified particles
    let particles = state_update.get_modified_particles();
    println!("Modified particles: {:?}", particles);

    // Access a specific modification
    if let Some(state) = state_update.get_modification(0, 2) {
        println!("Particle 0 at timeslice 2: {:?}", state);
    }
}
