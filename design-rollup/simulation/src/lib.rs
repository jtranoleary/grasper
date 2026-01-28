use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GlassSimulation {
    // Stored as flattened [x0, y0, x1, y1, ...]
    points: Vec<f32>,
}

#[wasm_bindgen]
impl GlassSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new(radius: f32, height: f32, segments: usize) -> GlassSimulation {
        let mut points = Vec::with_capacity(segments * 2);
        let half_height = height / 2.0;
        let segment_height = height / (segments as f32 - 1.0);

        // Create a simple cylinder profile
        // Ordered from top (+y) to bottom (-y) or vice versa.
        // Let's go Top (+y) to Bottom (-y) to match typical lathed geometry expectation if needed,
        // though LatheGeometry usually takes points. The order matters for the face culling.
        // Three.js Lathe usually expects points with increasing Y if you want simple mapping, 
        // but we can just be consistent.
        
        for i in 0..segments {
            let y = half_height - (i as f32 * segment_height);
            points.push(radius); // x (radius)
            points.push(y);      // y
        }

        GlassSimulation { points }
    }

    pub fn apply_jack(&mut self, y_center: f32, amplitude: f32, sigma: f32) {
        // R_new = R_old * (1.0 - Amplitude * e^(- (y - y_center)^2 / (2 * sigma^2)))
        // Amplitude should be clamped to avoid negative radius (R < 0)
        // Though physical jack won't go through itself usually, but we'll assume valid inputs or simple math.
        
        let two_sigma_sq = 2.0 * sigma * sigma;

        for i in (0..self.points.len()).step_by(2) {
            let r_idx = i;
            let y_idx = i + 1;
            
            let r_old = self.points[r_idx];
            let y = self.points[y_idx];
            
            let diff = y - y_center;
            let gaussian = (- (diff * diff) / two_sigma_sq).exp();
            let factor = 1.0 - amplitude * gaussian;
            
            // Prevent negative radius
            let r_new = (r_old * factor).max(0.1); 
            
            self.points[r_idx] = r_new;
        }
    }

    pub fn apply_stretch(&mut self, factor: f32) {
        // factor > 1.0 means stretching (longer, thinner)
        // factor < 1.0 means compressing (shorter, fatter)
        
        // R_new = R_old / sqrt(factor)
        // Y_new = Y_old * factor
        
        let r_scale = 1.0 / factor.sqrt();
        
        for i in (0..self.points.len()).step_by(2) {
            let r_idx = i;
            let y_idx = i + 1;

            self.points[r_idx] *= r_scale;
            self.points[y_idx] *= factor;
        }
    }

    pub fn apply_boundary_stretch(&mut self, y_origin: f32, offset: f32) {
        // Decay factor controls how far the stretch influence propagates.
        // 0.05 implies ~20 units decay length (1/e).
        let decay = 0.05; 
        let epsilon = 0.001;

        for i in (0..self.points.len()).step_by(2) {
            let y_idx = i + 1;
            let y = self.points[y_idx];
            
            if offset > 0.0 {
                // Moving UP:
                // Points Above (y > y_origin) move rigidly with the selection.
                // Points Below (y <= y_origin) stretch (move less as they get further from selection).
                if y > y_origin + epsilon {
                     self.points[y_idx] += offset;
                } else {
                     let dist = (y_origin - y).max(0.0);
                     let factor = (-decay * dist).exp();
                     self.points[y_idx] += offset * factor;
                }
            } else {
                // Moving DOWN (offset < 0):
                // Points Below (y < y_origin) move rigidly with the selection.
                // Points Above (y >= y_origin) stretch (move less as they get further from selection).
                if y < y_origin - epsilon {
                    self.points[y_idx] += offset;
                } else {
                    let dist = (y - y_origin).max(0.0);
                    let factor = (-decay * dist).exp();
                    self.points[y_idx] += offset * factor;
                }
            }
        }
    }

    pub fn get_points(&self) -> Vec<f32> {
        self.points.clone()
    }
}
