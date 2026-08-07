#![no_std]
#![no_main]

use defmt_rtt as _;
use panic_probe as _;

use core::fmt::Write;
use cortex_m_rt::entry;
use stm32f4xx_hal::{pac, prelude::*, serial::config::Config};

#[entry]
fn main() -> ! {
    let cp = cortex_m::Peripherals::take().unwrap();
    let dp = pac::Peripherals::take().unwrap();

    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.use_hse(8.MHz()).sysclk(168.MHz()).freeze();

    let gpioa = dp.GPIOA.split();

    let tx_pin = gpioa.pa2.into_alternate();
    let rx_pin = gpioa.pa3.into_alternate();
    let gpioc = dp.GPIOC.split();
    let button = gpioc.pc13.into_pull_up_input();
    let mut serial = dp
        .USART2
        .serial(
            (tx_pin, rx_pin),
            Config::default().baudrate(115_200.bps()),
            &clocks,
        )
        .unwrap();

    writeln!(serial, "Hello from Ferrite\r\n").unwrap();
    let pi: f32 = 3.14159;
    writeln!(serial, "pi = {:.4}\r\n", pi).unwrap();
    let mut counter = 0u32;
    let mut delay = cp.SYST.delay(&clocks);
    loop {
        if button.is_low() {
            writeln!(serial, "tick {}\r\n", counter).unwrap();
        }
        counter += 1;
        delay.delay_ms(500u32);
    }
}
