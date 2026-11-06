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
use frugal_ml::autodiff::optim::optimizer::Optimizer;
use frugal_ml::autodiff::optim::sgd::Sgd;
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

    macro_rules! sp {
        ($label:literal) => {
            defmt::info!("SP apres {} = {:x}", $label, cortex_m::register::msp::read());
        };
    }
    sp!("entree main, avant construction du reseau");

    let mut net = seq!(
        Linear::<64, 96, 6144>::from_seed(1),
        ReLU::<96> {},
        Linear::<96, 48, 4608>::from_seed(2),
        ReLU::<48> {},
        Linear::<48, 10, 480>::from_seed(3),
        Softmax::<10> {}
    );
    let mut opt = Sgd::new(0.05);
    sp!("reseau + optimiseur construits");

    for (i, &(x, label)) in DATASET.iter().take(3).enumerate() {
        defmt::info!("--- echantillon {} ---", i);
        let input = Vector::from_data(x);
        let target = one_hot(label);
        sp!("input/target construits");

        let (output, ctx) = net.forward(input);
        sp!("forward termine");

        let (loss, loss_grad) = cross_entropy(output, target);
        defmt::info!("loss = {}", loss);
        sp!("loss calculee");

        let (_, grads) = net.backward(loss_grad, &ctx);
        sp!("backward termine");

        opt.step(&mut net, &grads);
        sp!("update termine");
    }
    defmt::info!("fin des 3 echantillons sans crash");

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
