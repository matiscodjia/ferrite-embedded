#![no_std]
#![no_main]
use core::hint::black_box;

use cortex_m_rt::entry;
use defmt_rtt as _;
use panic_probe as _;
use stm32f4xx_hal::{pac, prelude::*};

#[entry]
fn main() -> ! {
    let sp = cortex_m::register::msp::read();
    defmt::info!("SP = {:x}, marge = {} octets", sp, sp - 0x2000_0000);
    //Pattern singleton, acquision des données
    let dp = pac::Peripherals::take().unwrap();

    //Configurer l'horloge
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();

    //Configurer la sortie (les pins)

    defmt::info!("sysclk reel = {} Hz", clocks.sysclk().raw());

    loop {}
}
