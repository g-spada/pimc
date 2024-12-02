use pimc_rs::path_state::worm::Worm;
use pimc_rs::updates::monte_carlo_update::MonteCarloUpdate;
use pimc_rs::updates::worm_translate::WormTranslate;

fn main() {
    let mut path = Worm::<3, 2, 3>::new();
    path.set_preceding(0, None);
    path.set_following(0, Some(2));
    path.set_preceding(2, Some(0));
    path.set_following(2, None);
    println!("Head is {:?}", path.worm_head());
    println!("Tail is {:?}", path.worm_tail());
    println!("Sector is {:?}", path.sector());

    let mut mc_transl = WormTranslate::new((&[1.0]).to_vec());
    let mut rng = rand::thread_rng();

    let success = mc_transl.try_update(&mut path, |_, _| 0.5, &mut rng);
    println!("Move accepted: {}", success);
    for part in 0..path.particles() {
        println!("{:?}", path.positions(part, 0, path.time_slices()));
    }
}
