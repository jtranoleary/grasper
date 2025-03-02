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
#[derive(Clone, Copy)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Particle {
    x: f32,
    y: f32,
    z: f32,
    px: f32,
    py: f32,
    pz: f32,
    vx: f32,
    vy: f32,
    vz: f32,
}

#[wasm_bindgen]
pub struct Simulation {
    particles: Vec<Particle>,
    floor_y: f32,
    gravity: f32,
    last_update_time: Option<Instant>,
    grid: UniformGrid,
    rest_density: f32,
    stiffness: f32,
    viscosity: f32,
}

#[wasm_bindgen]
impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 {
            x,
            y,
            z
        }
    }
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Simulation {
        let num_particles = 2000;
        let bounding_box_dim = 5.0;
        let y_offset = 2.0;
        let floor_y = 0.0;
        let mut particles = Vec::with_capacity(num_particles);

        let bounding_box_min = Vec3::new(
            -bounding_box_dim / 2.0,
            floor_y,
            -bounding_box_dim / 2.0
        );
        let bounding_box_max = Vec3::new(
            bounding_box_dim / 2.0,
            bounding_box_dim + floor_y,
            bounding_box_dim / 2.0
        );

        for _ in 0..num_particles {
            let x = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim;
            let y = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim + y_offset;
            let z = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim;
            particles.push(Particle {
                x,
                y,
                z,
                px: x,
                py: y,
                pz: z,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
            });
        }

        let cell_size = 0.5;
        let grid = UniformGrid::new(cell_size, bounding_box_min,
                                                 bounding_box_max);

        Simulation {
            particles,
            floor_y: 0.0,
            gravity: -9.81,
            last_update_time: None,
            grid,
            rest_density: 1000.0,
            stiffness: 1.0,
            viscosity: 0.1,
        }
    }

    pub fn update(&mut self) {
        let now = instant::Instant::now();
        let dt = match self.last_update_time {
            Some(last_time) => now.duration_since(last_time)
                                           .as_secs_f32(),
            None => 1.0 / 60.0,
        };
        self.last_update_time = Some(now);

        for particle in &mut self.particles {
            particle.vy += self.gravity * dt;

            particle.px = particle.x;
            particle.py = particle.y;
            particle.pz = particle.z;

            particle.x += particle.vx * dt;
            particle.y += particle.vy * dt;
            particle.z += particle.vz * dt;
        }

        for particle in &mut self.particles {
            particle.vx = (particle.x - particle.px) / dt;
            particle.vy = (particle.y - particle.py) / dt;
            particle.vz = (particle.z - particle.pz) / dt;

            if particle.y < self.floor_y {
                particle.y = self.floor_y;
                particle.py = 0.0;
            }
        }
    }

    pub fn get_particle_positions(&self) -> Vec<f32> {
        let mut positions = Vec::with_capacity(self.particles.len() * 3);
        for particle in &self.particles {
            positions.push(particle.x);
            positions.push(particle.y);
            positions.push(particle.z);
        }
        positions
    }

    pub fn reset_particles(&mut self) {
        let bounding_box_dim = 5.0;
        let y_offset = 2.0;

        for particle in &mut self.particles {
            let x = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim;
            let y = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim + y_offset;
            let z = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim;
            particle.x = x;
            particle.y = y;
            particle.z = z;
            particle.px = x;
            particle.py = y;
            particle.pz = z;
            particle.vx = 0.0;
            particle.vy = 0.0;
            particle.vz = 0.0;
        }
    }
}