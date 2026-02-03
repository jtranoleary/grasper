
#[cfg(test)]
mod tests {
    use crate::GlassSimulation;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_boundary_stretch_overlap() {
        let mut sim = GlassSimulation::new(10.0, 100.0, 10);
        // Points are 100.0 down to 0.0 (inclusive)
        // [100, 88.8, 77.7, ..., 0]
        
        let initial_points = sim.get_points();
        println!("Initial Ys: {:?}", initial_points.iter().skip(1).step_by(2).collect::<Vec<_>>());

        // Select near top (e.g. 90.0) and push DOWN by a lot (-60.0)
        // Decay is 0.05.
        // Point 0 (100.0) is > 90.0. Moves -60.0 rigid => 40.0.
        // Point at 0.0. Dist = 90.0. Factor = exp(-0.05 * 90) = exp(-4.5) ~ 0.011.
        // Point at 0.0 moves -60 * 0.011 = -0.66. Ends at -0.66.
        // New stats: Top is 40.0. Bottom is -0.66.
        // What about points in between?
        // Point at 44.4 (below 90). Dist = 45.6. Factor = exp(-0.05 * 45.6) = exp(-2.28) ~ 0.10.
        // Moves -60 * 0.10 = -6. Ends at 38.4.
        // So 40.0 > 38.4. Monotonicity seems preserved?
        
        // Let's try to break it.
        // We need the "Rigid" part to overtake the "Decay" part.
        // Rigid move is `y += offset`.
        // Decay move is `y += offset * factor`.
        // If offset is Negative (-D).
        // y_new = y - D * factor.
        // derivative dy_new/dy = 1 + D * decay * exp(-decay * dist). (Since dist = y_origin - y).
        // If derivative < 0, order flips.
        // 1 + (-D) * decay * factor < 0.
        // D * decay * factor > 1.
        // If D=100, decay=0.05. max factor=1.0.
        // 100 * 0.05 * 1.0 = 5.0 > 1.0. 
        // YES. It flips locally near the boundary!
        
        sim.apply_boundary_stretch(50.0, -100.0);
        
        let points = sim.get_points();
        let ys: Vec<f32> = points.iter().skip(1).step_by(2).cloned().collect();
        println!("Post-Stretch Ys: {:?}", ys);

        let mut monotonic = true;
        for i in 0..ys.len()-1 {
            if ys[i+1] > ys[i] { // We expect Descending order (Tip to Base)?
                // Wait, Simulation::new creates 1.0 -> 0.0 vs... 
                // Loop 0..segments: y = half_height - i*seg_height.
                // i=0 -> +y. i=max -> -y.
                // So Ys should be DESCENDING.
                // If ys[i+1] > ys[i], that's an ascent. Violation.
                monotonic = false;
                println!("Violation at {}: {} < {}", i, ys[i], ys[i+1]);
                break;
            }
        }
        
        assert!(monotonic, "Y values are not monotonic descending!");
    }
}
