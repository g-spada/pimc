//use ndarray::{Array1, ArrayView1};
//use pimc::space::free_space::FreeSpace;
use env_logger::Builder;
use log::{debug, info};
use ndarray::Zip;
use pimc::action::traits::PotentialDensityMatrix;
use pimc::path_state::sector::Sector;
use pimc::path_state::worm::Worm;
use pimc::space::periodic_box::PeriodicBox;
use pimc::system::homonuclear_system::HomonuclearSystem;
use pimc::system::traits::{ReseatPolymer, SystemAccess};
use pimc::updates::accepted_update::AcceptedUpdate;
use pimc::updates::monte_carlo_update::MonteCarloUpdate;
use pimc::updates::open_close::OpenClose;
use pimc::updates::proposed_update::ProposedUpdate;
use pimc::updates::redraw::Redraw;
use pimc::updates::redraw_head::RedrawHead;
use pimc::updates::redraw_tail::RedrawTail;
use pimc::updates::swap::Swap;
use pimc::updates::worm_translate::WormTranslate;
use pimc::utils::accumulator::Accumulator;
use pimc::utils::consts::ZETA_3_2;
use rand::distributions::{Distribution, WeightedIndex};
use std::f64::consts::PI;

const N: usize = 8;
const M: usize = 16;
const D: usize = 3;

const MP1: usize = M + 1;
const T_OVER_TC0: f64 = 1.0;
const DENSITY: f64 = 1e-4;

const SWEEPS: u64 = 2_u64.pow(24);
const WARMUP: u64 = 2_u64.pow(23);
const MEASURE_EVERY: u64 = 16;

const TARGET_NZ_NG_RATIO: f64 = 1.0;

pub struct DensityMatrix {}

impl PotentialDensityMatrix for DensityMatrix {
    fn potential_density_matrix<S: SystemAccess>(&self, _system: &S) -> f64 {
        1.0
    }

    fn potential_density_matrix_update<S: SystemAccess>(
        &self,
        _system: &S,
        _update: &ProposedUpdate<f64>,
    ) -> f64 {
        1.0
    }
}

fn main() {
    // Programmatically set the logging level
    Builder::new().filter_level(log::LevelFilter::Info).init();
    info!("T_OVER_TC0 = {:.2}", T_OVER_TC0);
    info!("N = {:.2}", N);
    info!("M = {:.2}", M);
    //// Create path object
    //let mut path = Worm::<N, MP1, D>::new();

    let box_side: f64 = (N as f64 / DENSITY).powf(1.0 / D as f64);
    let tc0: f64 = 4.0 * PI * (DENSITY / ZETA_3_2).powf(2.0 / 3.0);
    info!("tc0 = {}", tc0);
    let beta: f64 = 1.0 / (T_OVER_TC0 * tc0);
    info!("beta = {}", beta);
    let tau: f64 = beta / M as f64;

    // Space object
    let periodic_box = PeriodicBox::<D> {
        length: [box_side; D],
    };

    // Combine path and space into system
    let mut system = HomonuclearSystem {
        //space: flatlandia,
        space: periodic_box,
        path: Worm::<N, MP1, D>::new(),
        //two_lambda_tau: (ZETA_3_2 / DENSITY).powf(2.0 / 3.0) / (2.0 * PI * T_OVER_TC0 * M as f64),
        two_lambda_tau: 2.0 * tau,
    };

    // Print starting configuration
    //info!("Starting configuration:\n{:#?}", system);

    // Instantiate Action
    let action = DensityMatrix {};

    // RNG
    let mut rng = rand::thread_rng();

    // MONTE CARLO UPDATES
    // Translate update
    let upd = (2 * M / 3).max(1);
    let mut mc_transl = WormTranslate {
        max_displacement: 1.0,
        accept_count: 0,
        reject_count: 0,
    };

    // Redraw update
    let mut mc_redraw = Redraw {
        min_delta_t: upd,
        max_delta_t: upd,
        accept_count: 0,
        reject_count: 0,
    };

    // RedrawHead update
    let mut mc_redrawhead = RedrawHead {
        min_delta_t: upd,
        max_delta_t: upd,
        accept_count: 0,
        reject_count: 0,
    };

    // RedrawTail update
    let mut mc_redrawtail = RedrawTail {
        min_delta_t: upd,
        max_delta_t: upd,
        accept_count: 0,
        reject_count: 0,
    };

    // Open/Close update
    let mut mc_openclose = OpenClose {
        min_delta_t: upd,
        max_delta_t: upd,
        open_close_constant: DENSITY / N as f64,
        accept_count: 0,
        reject_count: 0,
    };

    // Swap update
    let mut mc_swap = Swap {
        min_delta_t: upd,
        max_delta_t: upd,
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
    let weights = [2, 10, 2, 2, 6, 2];

    // Create a weighted index for random selection
    let dist = WeightedIndex::new(&weights).unwrap();

    // Accumulator for the energy
    type DefaultAccumulator = Accumulator<1024, 16>;
    let mut energy_acc = DefaultAccumulator::new();
    let mut virial_acc = DefaultAccumulator::new();
    let mut sector_acc = DefaultAccumulator::new();

    for mc_it in 0..SWEEPS {
        debug!("######################################");
        debug!("# ITERATION {}", mc_it);

        let success: Option<AcceptedUpdate>;

        // Select an update randomly based on weights
        //match dist.sample(&mut rng) {
        //0 => success = mc_transl.monte_carlo_update(&mut system, &action, &mut rng),
        //1 => success = mc_redraw.monte_carlo_update(&mut system, &action, &mut rng),
        //2 => success = mc_redrawhead.monte_carlo_update(&mut system, &action, &mut rng),
        //3 => success = mc_redrawtail.monte_carlo_update(&mut system, &action, &mut rng),
        //4 => success = mc_openclose.monte_carlo_update(&mut system, &action, &mut rng),
        //5 => success = mc_swap.monte_carlo_update(&mut system, &action, &mut rng),
        //_ => unreachable!(),
        //};

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
                system.reseat_polymer(part);
                debug!(
                    "New path for p = {}\n{:?}",
                    part,
                    system
                        .path()
                        .positions(part, 0, system.path().time_slices())
                );
            }
        }

        // Print configuration
        debug!("Current configuration:\n{:?}", system.path());
        debug!("Head is {:?}", system.path().worm_head());
        debug!("Tail is {:?}", system.path().worm_tail());
        debug!("Sector is {:?}", system.path().sector());

        // Check permutations
        for particle in 0..N {
            if let Some(next) = system.path().following(particle) {
                let path = system.path();
                let space = system.space();
                let diff = space.difference(path.position(particle, M), path.position(next, 0));
                assert!(
                    diff.iter().any(|&l| l.abs() < 1e-10),
                    "Particles {} and {} not correctly glued together: difference is {:#?}",
                    particle,
                    next,
                    diff
                )
            }
        }

        if system.path.sector() == Sector::Z {
            sector_acc.add(1.0);
        } else {
            sector_acc.add(0.0);
        }

        if (mc_it < WARMUP) && ((mc_it + 1) % (WARMUP / 4) == 0) {
            info!("after mc_it = {} iterations", mc_it);
            let (mean, mean_error, _autocorr_time, _std_dev) = sector_acc.statistics().unwrap();
            info!("average sector Z frequency: {} +- {:.2e}", mean, mean_error);
            let factor = mean / (1.0 - mean);
            mc_openclose.open_close_constant *= factor / TARGET_NZ_NG_RATIO;
            info!(
                "new open/close constant {}",
                mc_openclose.open_close_constant
            );
            mc_openclose.accept_count = 0;
            mc_openclose.reject_count = 0;
            sector_acc.clear();
            //info!("Cleared sector_acc. New size is {}", sector_acc.size());
        }
        if mc_it >= WARMUP {
            if (mc_it - WARMUP) % MEASURE_EVERY == 0 {
                let path = system.path();
                if path.sector() == Sector::Z {
                    // COMPUTE ENERGY
                    let mut gradient_pow2_sum: f64 = 0.0;
                    let mut gradm_winding_sum: f64 = 0.0;
                    for particle in 0..N {
                        for slice in 0..M {
                            //let r_slice = path.position(particle, slice);
                            //let r_next  = path.position(particle, slice + 1);
                            //let mut grad_sq = 0.0;
                            //for d in 0..D {
                            //let diff = r_next[d] - r_slice[d];
                            //grad_sq += diff * diff;
                            //}
                            //gradient_pow2_sum += grad_sq;

                            gradient_pow2_sum += Zip::from(&path.position(particle, slice))
                                .and(&path.position(particle, slice + 1))
                                .map_collect(|&a, &b| (a - b).powi(2))
                                .sum()
                        }
                        let diff_last =
                            &path.position(particle, M - 1) - &path.position(particle, M);
                        let diff_jump = &path.position(particle, M) - &path.position(particle, 0);
                        gradm_winding_sum += diff_last.dot(&diff_jump);
                    }
                    // energy per particle (in units of kB*Tc0)
                    let d_over_two_tau = (D as f64) / (2.0 * tau);
                    let energy_per_particle = d_over_two_tau
                        - gradient_pow2_sum / (4.0 * tau * tau * (M as f64) * (N as f64));
                    energy_acc.add(energy_per_particle / tc0);
                    // energy per particle - virial estimator
                    let d_over_two_beta = (D as f64) / (2.0 * beta);
                    let virial_energy = d_over_two_beta
                        + gradm_winding_sum / (4.0 * tau * tau * (M as f64) * (N as f64));
                    virial_acc.add(virial_energy / tc0);

                    //assert_eq!(energy_per_particle, virial_energy, "TH {energy_per_particle}, VI {virial_energy}");

                    //println!("{}",energy_per_particle);
                }
            }
        }
    }
    println!("{:#?}", system.path());

    let (mean, mean_error, _autocorr_time, _std_dev) = sector_acc.statistics().unwrap();
    info!("average sector Z frequency: {} +- {:.2e}", mean, mean_error);

    println!(
        "Energy thermodynamic estimator (mean, err, autocorr, std_dev)\n{:#?}",
        energy_acc.statistics()
    );

    println!(
        "Energy virial estimator (mean, err, autocorr, std_dev)\n{:#?}",
        virial_acc.statistics()
    );
}
