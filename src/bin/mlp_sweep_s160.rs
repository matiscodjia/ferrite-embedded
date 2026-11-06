//! Entraine un MLP (que du Linear + activations, aucune conv) sur cible,
//! sur un jeu de donnees synthetique fixe (10 classes gaussiennes en
//! dimension 64, genere une fois par gen_dataset.rs, embarque en Flash
//! via `include!` -- cout RAM nul pour le dataset, voir mlp_dataset.rs).
//!
//! Architecture : 64 -> 64 -> 32 -> 10 -> Softmax, ReLU/ReLU/Softmax+cross_entropy, SGD.
//! (cross_entropy attend des probabilites en entree, pas des logits bruts --
//! le Softmax final est necessaire, son omission donne un loss NaN des la
//! premiere epoque : log() d'une valeur negative.)
//! Budget theorique poids+gradients : 64*64+64*32+32*10 = 6464 cases
//! * 8 octets (f32 poids + f32 gradient) = ~50 Ko, sur 128 Ko de RAM totale
//! -- marge volontaire tant que le vrai cout pile n'est pas mesure (cf.
//! d88645a dans l'historique : le reel est deja mesure ~2x le theorique
//! pour les tenseurs de conv, meme prudence de mise ici).
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_rt::{exception, ExceptionFrame};
use defmt_rtt as _;
use frugal_ml::autodiff::core::module::Module;
use frugal_ml::autodiff::layers::activations::{ReLU, Softmax};
use frugal_ml::autodiff::layers::linear::Linear;
use frugal_ml::autodiff::loss::cross_entropy;
use frugal_ml::autodiff::optim::sgd::Sgd;
use frugal_ml::autodiff::optim::train::train_step;
use frugal_ml::linalg::tensor::Vector;
use frugal_ml::seq;
use frugal_ml::Scalar;
use panic_probe as _;
use stm32f4xx_hal::{pac, prelude::*};

include!("../mlp_dataset.rs");

mod bench {
    use cortex_m::peripheral::{DCB, DWT};

    pub fn init(dcb: &mut DCB, dwt: &mut DWT) {
        dcb.enable_trace();
        dwt.enable_cycle_counter();
    }

    #[inline(always)]
    pub fn cycles() -> u32 {
        DWT::cycle_count()
    }
}

#[exception]
unsafe fn HardFault(ef: &ExceptionFrame) -> ! {
    defmt::error!("HardFault ! PC={:x} SP={:x}", ef.pc(), ef.lr());
    loop {}
}

fn one_hot(label: usize) -> Vector<K> {
    let mut data = [0.0 as Scalar; K];
    data[label] = 1.0;
    Vector::from_data(data)
}

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let mut cp = cortex_m::Peripherals::take().unwrap();
    bench::init(&mut cp.DCB, &mut cp.DWT);

    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();
    defmt::info!("sysclk reel = {} Hz", clocks.sysclk().raw());
    defmt::info!(
        "dataset : {} echantillons, {} classes, dimension {}",
        N,
        K,
        IN
    );

    let mut net = seq!(
        Linear::<64, 160, 10240>::from_seed(1),
        ReLU::<160> {},
        Linear::<160, 80, 12800>::from_seed(2),
        ReLU::<80> {},
        Linear::<80, 10, 800>::from_seed(3),
        Softmax::<10> {}
    );
    let mut opt = Sgd::new(0.05);

    const EPOCHS: u32 = 3;
    for epoch in 0..EPOCHS {
        let t0 = bench::cycles();
        let mut total_loss: Scalar = 0.0;
        for &(x, label) in DATASET.iter() {
            let input = Vector::from_data(x);
            let target = one_hot(label);
            total_loss += train_step(&mut net, &mut opt, input, target, cross_entropy);
        }
        let elapsed = bench::cycles().wrapping_sub(t0);
        let mean_loss = total_loss / N as Scalar;

        if epoch % 10 == 0 || epoch == EPOCHS - 1 {
            defmt::info!(
                "epoch {} | loss moyenne = {} | {} cycles/epoque ({} us) | {} us/echantillon",
                epoch,
                mean_loss,
                elapsed,
                elapsed as f32 / 168.0,
                elapsed as f32 / 168.0 / N as f32
            );
        }
    }

    // Precision finale, calculee sur le meme jeu (pas de split train/test --
    // le but ici est la preuve de convergence, pas une mesure de generalisation).
    let mut correct: u32 = 0;
    for &(x, label) in DATASET.iter() {
        let input = Vector::from_data(x);
        let (output, _) = net.forward(input);
        let mut best_idx = 0;
        let mut best_val = output[0];
        for i in 1..K {
            if output[i] > best_val {
                best_val = output[i];
                best_idx = i;
            }
        }
        if best_idx == label {
            correct += 1;
        }
    }
    defmt::info!("precision finale (train) : {}/{}", correct, N as u32);

    loop {}
}
