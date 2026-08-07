#![no_std]
#![no_main]
use core::hint::black_box;

use cortex_m_rt::entry;
use defmt_rtt as _;
use panic_probe as _;
use stm32f4xx_hal::{pac, prelude::*};

#[entry]
fn main() -> ! {
    //Pattern singleton, acquision des données
    let dp = pac::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();

    //Configurer l'horloge
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();

    //Configurer la sortie (les pins)
    let gpioa = dp.GPIOA.split();
    let mut led = gpioa.pa5.into_push_pull_output();

    let mut delay = cp.SYST.delay(&clocks);
    defmt::info!("sysclk reel = {} Hz", clocks.sysclk().raw());

    loop {
        let tab = black_box([0.0; 102400]);
        led.set_high();
        delay.delay_ms(500u32);
        led.set_low();
        delay.delay_ms(200u32);
    }
}
