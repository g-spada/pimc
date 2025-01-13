use env_logger::Builder;
use log::info;
//use ndarray::{Array1, ArrayView1};
use pimc_rs::action::traits::PotentialDensityMatrix;
use pimc_rs::path_state::worm::Worm;
use pimc_rs::space::free_space::FreeSpace;
use pimc_rs::system::traits::SystemAccess;
use pimc_rs::updates::accepted_update::AcceptedUpdate;
use pimc_rs::updates::monte_carlo_update::MonteCarloUpdate;
use pimc_rs::updates::open_close::OpenClose;
use pimc_rs::updates::proposed_update::ProposedUpdate;
use pimc_rs::updates::redraw::Redraw;
use pimc_rs::updates::redraw_head::RedrawHead;
use pimc_rs::updates::redraw_tail::RedrawTail;
use pimc_rs::updates::swap::Swap;
use pimc_rs::updates::worm_translate::WormTranslate;
use rand::distributions::{Distribution, WeightedIndex};
//use rand::Rng;

const N: usize = 3;
const M: usize = 8;
const D: usize = 2;

const MP1: usize = M + 1;

pub struct MySystem {
    pub space: FreeSpace<D>,
    pub path: Worm<N, MP1, D>,
}

impl SystemAccess for MySystem {
    type Space = FreeSpace<D>;
    type WorldLine = Worm<N, MP1, D>;

    fn space(&self) -> &Self::Space {
        &self.space
    }

    fn path(&self) -> &Self::WorldLine {
        &self.path
    }

    fn path_mut(&mut self) -> &mut Self::WorldLine {
        &mut self.path
    }

    fn two_lambda_tau(&self, _: usize) -> f64 {
        0.1
    }
}

pub struct FakeDensityMatrix {}

impl PotentialDensityMatrix for FakeDensityMatrix {
    fn potential_density_matrix<S: SystemAccess>(&self, _system: &S) -> f64 {
        1.0
    }

    fn potential_density_matrix_update<S: SystemAccess>(
        &self,
        _system: &S,
        _update: &ProposedUpdate<f64>,
    ) -> f64 {
        0.5
    }
}

fn main() {
    // Programmatically set the logging level to TRACE
    Builder::new().filter_level(log::LevelFilter::Trace).init();

    info!("Starting Example");
    // Create path object
    let mut path = Worm::<N, MP1, D>::new();
    // Create a worm of length two: (T) - 0 - 2 - (H)
    path.set_preceding(0, None);
    path.set_following(0, Some(2));
    path.set_preceding(2, Some(0));
    path.set_following(2, None);

    // Space object
    let flatlandia = FreeSpace::<D>;

    // Combine path and space into system
    let mut system = MySystem {
        space: flatlandia,
        path: path,
    };

    // Print starting configuration
    info!("Starting configuration:\n{:?}", system.path());

    // Instantiate Action
    let action = FakeDensityMatrix {};

    // RNG
    let mut rng = rand::thread_rng();

    // MONTE CARLO UPDATES
    // Translate update
    let mut mc_transl = WormTranslate {
        max_displacement: 1.0,
        accept_count: 0,
        reject_count: 0,
    };

    // Redraw update
    let mut mc_redraw = Redraw {
        min_delta_t: M / 3,
        max_delta_t: M - 1,
        accept_count: 0,
        reject_count: 0,
    };

    // RedrawHead update
    let mut mc_redrawhead = RedrawHead {
        min_delta_t: M / 3,
        max_delta_t: M - 1,
        accept_count: 0,
        reject_count: 0,
    };

    // RedrawTail update
    let mut mc_redrawtail = RedrawTail {
        min_delta_t: M / 3,
        max_delta_t: M - 1,
        accept_count: 0,
        reject_count: 0,
    };

    // Open/Close update
    let mut mc_openclose = OpenClose {
        min_delta_t: M / 3,
        max_delta_t: M - 1,
        open_close_constant: 0.5,
        max_head_displacement: 1.0,
        accept_count: 0,
        reject_count: 0,
    };

    // Swap update
    let mut mc_swap = Swap {
        min_delta_t: M / 3,
        max_delta_t: M - 1,
        accept_count: 0,
        reject_count: 0,
    };

    //let updates: [&dyn MonteCarloUpdate; 6] = [
    //&mc_transl,
    //&mc_redraw,
    //&mc_redrawhead,
    //&mc_redrawtail,
    //&mc_openclose,
    //&mc_swap,
    //];

    // Define relative frequencies
    let weights = [20, 30, 10, 15, 15, 10];

    // Create a weighted index for random selection
    let dist = WeightedIndex::new(&weights).unwrap();

    for mc_it in 0..10 {
        info!("######################################");
        info!("# ITERATION {}", mc_it);

        let success: Option<AcceptedUpdate>;

        // Select an update randomly based on weights
        match dist.sample(&mut rng) {
            0 => success = mc_transl.monte_carlo_update(&mut system, &action, &mut rng),
            1 => success = mc_redraw.monte_carlo_update(&mut system, &action, &mut rng),
            2 => success = mc_redrawhead.monte_carlo_update(&mut system, &action, &mut rng),
            3 => success = mc_redrawtail.monte_carlo_update(&mut system, &action, &mut rng),
            4 => success = mc_openclose.monte_carlo_update(&mut system, &action, &mut rng),
            5 => success = mc_swap.monte_carlo_update(&mut system, &action, &mut rng),
            _ => unreachable!(),
        };

        if let Some(update) = success {
            for part in update.modified_particles {
                info!(
                    "New path for p = {}\n{:?}",
                    part,
                    system
                        .path()
                        .positions(part, 0, system.path().time_slices())
                );
            }
        }

        // Print configuration
        info!("Current configuration:\n{:?}", system.path());
        info!("Head is {:?}", system.path().worm_head());
        info!("Tail is {:?}", system.path().worm_tail());
        info!("Sector is {:?}", system.path().sector());

        // Check permutations
        for particle in 0..N {
            if let Some(next) = system.path().following(particle) {
                let diff = system.path().position(particle, M).to_owned()
                    - system.path().position(next, 0);
                assert!(
                    diff.iter().any(|&l| l.abs() < 1e-10),
                    "Particles {} and {} not correctly glued together: difference is {:}",
                    particle,
                    next,
                    diff
                )
            }
        }
    }
}
