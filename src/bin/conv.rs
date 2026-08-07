#![no_std]
#![no_main]

use core::hint::black_box;
use cortex_m_rt::entry;
use cortex_m_rt::{ExceptionFrame, exception};
use defmt_rtt as _;
use ferrite::linalg::{Tensor3D, Tensor4D, tensordot_3};
use ferrite::sp::{Gaussian3D, filter_bank};
use panic_probe as _;
use stm32f4xx_hal::{pac, prelude::*};
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

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let mut cp = cortex_m::Peripherals::take().unwrap();
    bench::init(&mut cp.DCB, &mut cp.DWT);

    // Sans ceci, le cœur tourne sur l'horloge de reset par défaut (HSI, 16 MHz)
    // et pas les 168 MHz que ce fichier suppose plus bas pour convertir les
    // cycles DWT en µs — un compteur de cycles reste correct à n'importe
    // quelle fréquence, mais la conversion en temps réel ne l'est pas sans ça.
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();
    defmt::info!("sysclk reel = {} Hz", clocks.sysclk().raw());

    const N: u32 = 10;

    let tensor = Tensor4D::<1, 1, 96, 96, 9216>::new([1.0; 9216]); // 72ko
    //Simulation d'une image monochrome de 64 par 64 pixels
    let filter: Tensor3D<1, 3, 3, 9> = Gaussian3D::kernel(); //filtre couleur 3 par 3 --> 9 octets
    let filters: Tensor4D<1, 1, 3, 3, 9> = filter_bank([&filter; 1]); //extension en dim 4 pour
    //-> 9 octets
    //la compatibilité de l'opération de contraction tensoriel avec la sequence video

    loop {
        let t0 = bench::cycles();
        for _ in 0..N {
            let _: Tensor4D<1, 94, 94, 1, 8836> = tensordot_3( //17ko
                black_box(&tensor.im2col_view::<94, 94, 3, 3>(1)),
                black_box(&filters),
            );
        }
        let per_iter = bench::cycles().wrapping_sub(t0) / N;

        defmt::info!(
            "mul_vec : {} cycles ({} us)",
            per_iter,
            per_iter as f32 / 168.0
        );
    }
}
