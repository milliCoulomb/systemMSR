// src/thermal_hydraulics/pump.rs

use crate::input::input_deck::PumpSchedule;

/// Represents a pump with a flow rate schedule.
#[derive(Debug, Clone)]
pub struct Pump {
    pub schedule: PumpSchedule,
}

impl Pump {
    /// Creates a new Pump with the given schedule.
    ///
    /// # Arguments
    ///
    /// * `schedule` - PumpSchedule defining flow rates over time.
    pub fn new(schedule: PumpSchedule) -> Self {
        Pump { schedule }
    }

    /// Retrieves the flow rate at a specific time using linear interpolation.
    ///
    /// # Arguments
    ///
    /// * `time` - Current simulation time [s].
    ///
    /// # Returns
    ///
    /// * `f64` representing the flow rate at the given time.
    pub fn get_flow_rate(&self, time: f64) -> f64 {
        interpolate(&self.schedule.time, &self.schedule.flow_rate, time)
    }
}

/// Performs linear interpolation given vectors of x and y values and a target value.
///
/// # Arguments
///
/// * `x` - Slice of x-values.
/// * `y` - Slice of y-values.
/// * `value` - Target x-value.
///
/// # Returns
///
/// * Interpolated y-value.
fn interpolate(x: &[f64], y: &[f64], value: f64) -> f64 {
    if value <= x[0] {
        y[0]
    } else if value >= x[x.len()-1] {
        y[y.len()-1]
    } else {
        for i in 0..x.len()-1 {
            if value >= x[i] && value < x[i+1] {
                let t = (value - x[i]) / (x[i+1] - x[i]);
                return y[i] * (1.0 - t) + y[i+1] * t;
            }
        }
        y[0] // Fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 0.0];
        assert_eq!(interpolate(&x, &y, -1.0), 0.0);
        assert_eq!(interpolate(&x, &y, 0.0), 0.0);
        assert_eq!(interpolate(&x, &y, 0.5), 0.5);
        assert_eq!(interpolate(&x, &y, 1.0), 1.0);
        assert_eq!(interpolate(&x, &y, 1.5), 0.5);
        assert_eq!(interpolate(&x, &y, 2.0), 0.0);
        assert_eq!(interpolate(&x, &y, 3.0), 0.0);
    }

    #[test]
    fn test_get_flow_rate() {
        let schedule = PumpSchedule {
            time: vec![0.0, 50.0, 100.0],
            flow_rate: vec![1.0, 1.5, 1.0],
        };
        let pump = Pump::new(schedule);
        assert_eq!(pump.get_flow_rate(-1.0), 1.0);
        assert_eq!(pump.get_flow_rate(0.0), 1.0);
        assert_eq!(pump.get_flow_rate(25.0), 1.25);
        assert_eq!(pump.get_flow_rate(50.0), 1.5);
        assert_eq!(pump.get_flow_rate(75.0), 1.25);
        assert_eq!(pump.get_flow_rate(100.0), 1.0);
        assert_eq!(pump.get_flow_rate(101.0), 1.0);
    }

    // also test the Pump struct
    #[test]
    fn test_pump() {
        let schedule = PumpSchedule {
            time: vec![0.0, 50.0, 100.0],
            flow_rate: vec![1.0, 1.5, 1.0],
        };
        let pump = Pump::new(schedule.clone());
        assert_eq!(pump.schedule.time, schedule.time);
        assert_eq!(pump.schedule.flow_rate, schedule.flow_rate);
    }

    // test the PumpSchedule struct
    #[test]
    fn test_pump_schedule() {
        let schedule = PumpSchedule {
            time: vec![0.0, 50.0, 100.0],
            flow_rate: vec![1.0, 1.5, 1.0],
        };
        assert_eq!(schedule.time, vec![0.0, 50.0, 100.0]);
        assert_eq!(schedule.flow_rate, vec![1.0, 1.5, 1.0]);
    }
}
