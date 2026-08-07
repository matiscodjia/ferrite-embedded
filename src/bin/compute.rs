#![no_std]
#![no_main]

use core::hint::black_box;
use cortex_m_rt::entry;
use defmt_rtt as _;
use ferrite::Tensor;
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

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let mut cp = cortex_m::Peripherals::take().unwrap();
    bench::init(&mut cp.DCB, &mut cp.DWT);

    // Même correctif que conv.rs : sans ça, DWT compte des cycles à 16 MHz
    // (HSI par défaut) alors que la conversion en µs plus bas suppose 168 MHz.
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();
    defmt::info!("sysclk reel = {} Hz", clocks.sysclk().raw());

    const N: u32 = 500;
    let a = Tensor::<4, 4, 16>::new([1.0; 16]);
    let b = Tensor::<4, 4, 16>::new([1.0; 16]);
    loop {
        let t0 = bench::cycles();
        for _ in 0..N {
            black_box(black_box(&a).multiply::<4, 16, 16>(black_box(&b)));
        }
        let per_iter = bench::cycles().wrapping_sub(t0) / N;

        defmt::info!(
            "mul_vec : {} cycles ({} us)",
            per_iter,
            per_iter as f32 / 168.0
        );
    }
}
