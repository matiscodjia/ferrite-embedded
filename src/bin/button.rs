#![no_std]
#![no_main]
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

    let gpioc = dp.GPIOC.split();
    let button = gpioc.pc13.into_pull_up_input();

    let mut delay = cp.SYST.delay(&clocks);
    defmt::info!("sysclk reel = {} Hz", clocks.sysclk().raw());

    loop {
        if button.is_low() {
            led.toggle();
        }
        delay.delay_ms(100u32);
    }
}
