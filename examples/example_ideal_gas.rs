use env_logger::Builder;
use log::{debug, info};
use ndarray::Zip;
use pimc::action::traits::PotentialDensityMatrix;
use pimc::define_pimc;
use pimc::monte_carlo::proposed_update::ProposedUpdate;
use pimc::path::path_configuration::PathConfiguration;
use pimc::path::sector::Sector;
use pimc::space::periodic_box::PeriodicBox;
use pimc::system::homonuclear_system::HomonuclearSystem;
use pimc::system::traits::SystemAccess;
use pimc::updates::{OpenClose, Redraw, RedrawHead, RedrawTail, Swap, Translate};
use pimc::utils::accumulator::Accumulator;
use pimc::utils::consts::ZETA_3_2;
use pimc::utils::ideal_bosons::ideal_gas_energy;
use rand_pcg::Pcg64;
use std::f64::consts::PI;
//use rand::distributions::{Distribution, WeightedIndex};

const N: usize = 8;
const M: usize = 8;
const D: usize = 3;

const MP1: usize = M + 1;
const T_OVER_TC0: f64 = 1.0;
const DENSITY: f64 = 1e-4;

const SWEEPS: u64 = 2_u64.pow(24);
const WARMUP: u64 = 2_u64.pow(23);
const MEASURE_EVERY: u64 = 16;

const TARGET_NZ_NG_RATIO: f64 = 1.0;

define_pimc!(
    Pimc,
    [OpenClose, Translate, Redraw, RedrawHead, RedrawTail, Swap]
);

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
    info!("////////////////////////////////////////////////");
    info!("T_OVER_TC0 = {:.2}", T_OVER_TC0);
    info!("N = {:.2}", N);
    info!("M = {:.2}", M);

    let box_side: f64 = (N as f64 / DENSITY).powf(1.0 / D as f64);
    let tc0: f64 = 4.0 * PI * (DENSITY / ZETA_3_2).powf(2.0 / 3.0);
    info!("tc0 = {}", tc0);
    let beta: f64 = 1.0 / (T_OVER_TC0 * tc0);
    info!("beta = {}", beta);
    let tau: f64 = beta / M as f64;
    info!("////////////////////////////////////////////////");

    // Space object
    let periodic_box = PeriodicBox::<D> {
        length: [box_side; D],
    };

    // Combine path and space into system
    let system = HomonuclearSystem {
        //space: flatlandia,
        space: periodic_box,
        path: PathConfiguration::<N, MP1, D>::new(),
        //two_lambda_tau: (ZETA_3_2 / DENSITY).powf(2.0 / 3.0) / (2.0 * PI * T_OVER_TC0 * M as f64),
        two_lambda_tau: 2.0 * tau,
    };

    // Print starting configuration
    //info!("Starting configuration:\n{:#?}", system);

    // Action
    let action = DensityMatrix {};

    let mut pimc = Pimc::new(system, action, Pcg64::new(42_u128, 0))
        .with_update(OpenClose::new(M / 2, M - 2, DENSITY))
        .with_update(Translate::new(box_side / 100.0))
        .with_update(Redraw::new(M / 2, M - 2))
        .with_update(RedrawHead::new(M / 2, M - 2))
        .with_update(RedrawTail::new(M / 2, M - 2))
        .with_update(Swap::new(M / 2, M - 2));

    // Test the step method
    pimc.step::<OpenClose>();
    pimc.step::<Redraw>();
    info!("{}", pimc.stats());

    //// Define relative frequencies
    //let weights = [2, 10, 2, 2, 6, 2];

    //// Create a weighted index for random selection
    //let dist = WeightedIndex::new(&weights).unwrap();

    // Accumulator for the energy
    type DefaultAccumulator = Accumulator<1024, 16>;
    let mut energy_acc = DefaultAccumulator::new();
    let mut virial_acc = DefaultAccumulator::new();
    let mut sector_acc = DefaultAccumulator::new();

    for mc_it in 0..SWEEPS {
        debug!("######################################");
        debug!("# ITERATION {}", mc_it);
        pimc.sweep();

        // Print configuration
        debug!("Current configuration:\n{:?}", pimc.system.path());
        debug!("Head is {:?}", pimc.system.path().worm_head());
        debug!("Tail is {:?}", pimc.system.path().worm_tail());
        debug!("Sector is {:?}", pimc.system.path().sector());
        if pimc.system.path.sector() == Sector::Z {
            sector_acc.add(1.0);
        } else {
            sector_acc.add(0.0);
        }

        if (mc_it < WARMUP) && ((mc_it + 1) % (WARMUP / 4) == 0) {
            info!("after mc_it = {} iterations", mc_it);
            let (mean, mean_error, _autocorr_time, _std_dev) = sector_acc.statistics().unwrap();
            info!("average sector Z frequency: {} +- {:.2e}", mean, mean_error);
            let factor = mean / (1.0 - mean);
            let old_open_close_constant = pimc
                .get_update_parameter::<OpenClose>("open_close_constant")
                .unwrap();
            pimc.set_update_parameter::<OpenClose>(
                "open_close_constant",
                old_open_close_constant * factor / TARGET_NZ_NG_RATIO,
            );
            info!(
                "new open/close constant {}",
                pimc.get_update_parameter::<OpenClose>("open_close_constant")
                    .unwrap()
            );
            sector_acc.clear();
            debug!("Cleared sector_acc. New size is {}", sector_acc.size());
        }
        if mc_it >= WARMUP {
            if (mc_it - WARMUP) % MEASURE_EVERY == 0 {
                let path = pimc.system.path();
                if path.sector() == Sector::Z {
                    // COMPUTE ENERGY
                    let mut gradient_pow2_sum: f64 = 0.0;
                    let mut gradm_winding_sum: f64 = 0.0;
                    for particle in 0..N {
                        for slice in 0..M {
                            let pos1 = path.position(particle, slice);
                            let pos2 = path.position(particle, slice + 1);
                            let squared_diffs = Zip::from(&pos1)
                                .and(&pos2)
                                .map_collect(|&a, &b| (a - b).powi(2));
                            gradient_pow2_sum += squared_diffs.sum();
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
                }
            }
        }
    }
    println!("{:#?}", pimc.system.path());

    info!("{}", pimc.stats());

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

    println!(
        "COMPARE WITH EXACT VALUE: {}",
        ideal_gas_energy(N, T_OVER_TC0)
    );
}
