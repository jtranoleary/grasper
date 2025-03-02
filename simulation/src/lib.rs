// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Entry point for the fluid simulation.

use instant::Instant;
use js_sys;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Particle {
    x: f32,
    y: f32,
    z: f32,
    vy: f32
}

#[wasm_bindgen]
pub struct Simulation {
    particles: Vec<Particle>,
    floor_y: f32,
    gravity: f32,
    last_update_time: Option<Instant>
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Simulation {
        let mut simulation = Simulation {
            particles: Vec::new(),
            floor_y: 0.0,
            gravity: -9.8,
            last_update_time: Some(instant::Instant::now())
        };
        simulation.reset_particles();

        simulation
    }

    pub fn update(&mut self) {
        let now = instant::Instant::now();
        let dt = match self.last_update_time {
            Some(last_time) => now.duration_since(last_time)
                                           .as_secs_f32(),
            None => 1.0 / 60.0
        };
        self.last_update_time = Some(now);

        for particle in &mut self.particles {
            particle.vy += self.gravity * dt;
            particle.y += particle.vy * dt;
            if particle.y < self.floor_y {
                particle.y = self.floor_y;
                particle.vy = 0.0;
            }
        }
    }

    pub fn get_particle_positions(&self) -> Vec<f32> {
        let mut positions = Vec::with_capacity(
            self.particles.len() * 3);
        for particle in &self.particles {
            positions.push(particle.x);
            positions.push(particle.y);
            positions.push(particle.z);
        }
        positions
    }

    pub fn reset_particles(&mut self) {
        let num_particles = 10000;
        let bounding_box_dim = 5.0;
        let y_offset = 2.0;
        let mut particles = Vec::with_capacity(num_particles);

        for _ in 0..num_particles {
            particles.push(Particle {
                x: (js_sys::Math::random() as f32 - 0.5) * bounding_box_dim,
                y: (js_sys::Math::random() as f32 - 0.5) * bounding_box_dim +
                    y_offset,
                z: (js_sys::Math::random() as f32 - 0.5) * bounding_box_dim,
                vy: 0.0
            });
        }

        self.particles = particles;
    }
}
