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

// This file is the simulation's main entry point.

import * as THREE from '../node_modules/@types/three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
import { TeapotGeometry } from 'three/addons/geometries/TeapotGeometry.js';

import {
    EffectComposer,
    RenderPass,
    SelectiveBloomEffect,
    BlendFunction,
    EffectPass,
} from 'postprocessing';

const hdrTextureURL = new URL('/hdri/machine_shop.hdr', import.meta.url).href;

class StarterScene {
    private camera!: THREE.PerspectiveCamera;
    private scene!: THREE.Scene;
    private renderer!: THREE.WebGLRenderer;
    private stats!: Stats;
    private gui!: GUI;
    private composer!: EffectComposer;
    private underlay!: THREE.Mesh;
    private controls!: OrbitControls;
    private bloomPass!: SelectiveBloomEffect;

    constructor() {
        this.init();
    }

    private init() {
        this.createScene();
        this.createCamera();
        this.createRenderer();
        this.createComposer();

        // Create objects before adding postprocessing
        this.createObjects();
        this.setupPostProcessing();

        this.setupHDR();
        this.setupControls();
        this.setupGUI();
        this.setupEventListeners();
        this.startAnimationLoop();
    }

    private createScene() {
        this.scene = new THREE.Scene();
    }

    private createCamera() {
        this.camera = new THREE.PerspectiveCamera(
            50, window.innerWidth / window.innerHeight, 0.1, 100);
        this.camera.position.set(0.5, 1, 2);
        this.camera.layers.enable(0);
        this.camera.layers.enable(1);
    }

    private createRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            powerPreference: "high-performance",
            antialias: false,
            stencil: false,
            depth: false
        });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(this.renderer.domElement);
    }

    private createComposer() {
        this.composer = new EffectComposer(this.renderer);
        const mainRenderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(mainRenderPass);
    }

    private createObjects() {
        this.underlay = this.makeUnderlay();
        this.scene.add(this.underlay);

        const teapot = this.makeTeapot();
        this.scene.add(teapot);
    }

    private makeUnderlay(): THREE.Mesh {
        // TODO: replace with a loaded mesh. Use hard-coded points and generate
        // a mesh programatically for now.
        const points = [
            // ...
        ];
        const vectors = points.map((point) => new THREE.Vector2(point[0] / 50,
                                                                point[1] / 50));
        const geometry = new THREE.LatheGeometry(vectors);
        const material = new THREE.MeshPhysicalMaterial({
            color: 0xffffff,
            emissive: 0xEC6F06,
            emissiveIntensity: 1,
            transmission: 0.95,
            opacity: 1,
            metalness: 0,
            roughness: 0.05,
            ior: 1.75,
            thickness: 0.01,
            specularIntensity: 1,
            specularColor: 0xffffff,
            envMapIntensity: 1,
            side: THREE.DoubleSide,
            transparent: true,
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = 'underlayMesh';
        mesh.layers.set(1);
        mesh.rotateX(Math.PI / 2);
        mesh.position.y = 1;

        return mesh;
    }

    private makeTeapot(): THREE.Mesh {
        const geometry = new TeapotGeometry(1, 18);
        const material = new THREE.MeshStandardMaterial({ color: 0xeeeeee });
        const teapot = new THREE.Mesh(geometry, material);
        return teapot;
    }

    private setupPostProcessing() {
        this.bloomPass = new SelectiveBloomEffect(this.scene, this.camera, {
            blendFunction: BlendFunction.ADD,
            mipmapBlur: true,
            intensity: 0.75,
        });
        this.bloomPass.inverted = true;
        this.bloomPass.ignoreBackground = true;
        this.bloomPass.selection.add(this.underlay);

        this.bloomPass.luminanceMaterial.threshold = 0.1;
        this.bloomPass.luminanceMaterial.smoothing = 0.8;
        this.bloomPass.mipmapBlurPass.radius = 1.0

        const effectPass = new EffectPass(this.camera, this.bloomPass);
        this.composer.addPass(effectPass);
    }


    private setupHDR() {
        new RGBELoader()
            .setDataType(THREE.FloatType)
            .load(hdrTextureURL, (texture) => {
                const pmremGenerator = new THREE.PMREMGenerator(this.renderer);
                pmremGenerator.compileEquirectangularShader();

                const envMap = pmremGenerator.fromEquirectangular(texture)
                                             .texture;

                this.scene.background = envMap;
                this.scene.environment = envMap;

                texture.dispose();
                pmremGenerator.dispose();

                this.updateMaterialEnvMap(this.underlay, envMap);
                this.updateMaterialEnvMap(this.makeTeapot(), envMap);
            });
    }

    private updateMaterialEnvMap(mesh: THREE.Mesh, envMap: THREE.Texture) {
        if (mesh.material instanceof THREE.MeshStandardMaterial||
            mesh.material instanceof THREE.MeshPhysicalMaterial) {
            mesh.material.envMap = envMap;
            mesh.material.needsUpdate = true;
        }
    }

    private setupControls() {
        this.controls = new OrbitControls(this.camera,
                                          this.renderer.domElement);
    }

    private setupGUI() {
        this.stats = new Stats();
        document.body.appendChild(this.stats.dom);
        this.gui = new GUI();

        const glassMaterialFolder = this.gui.addFolder('Glass Material');
        const glassMaterial = this.underlay.material as THREE.MeshPhysicalMaterial;
        glassMaterialFolder.add(glassMaterial, 'transmission', 0, 1, 0.01)
                           .name('Transmission');
        glassMaterialFolder.add(glassMaterial, 'roughness', 0, 1, 0.01)
                           .name('Roughness');
        glassMaterialFolder.add(glassMaterial, 'ior', 1, 2.333, 0.01)
                           .name('IOR');
        glassMaterialFolder.add(glassMaterial, 'envMapIntensity', 0, 5, 0.1)
                           .name('Env Map Intensity');
        glassMaterialFolder.add(glassMaterial, 'emissiveIntensity', 0, 2, 0.01)
                           .name('Emissive Intensity');

        const bloomFolder = this.gui.addFolder('Bloom');
        bloomFolder.add(this.bloomPass, 'intensity', 0, 3)
                   .name('Intensity');
        bloomFolder.add(this.bloomPass.luminanceMaterial, 'threshold', 0, 1)
                   .name('Threshold');
        bloomFolder.add(this.bloomPass.luminanceMaterial, 'smoothing', 0, 1)
                   .name('Smoothing');
        bloomFolder.add(this.bloomPass.mipmapBlurPass, 'radius', 0.1, 2, 0.01)
                   .name('Mipmap Blur Radius');
    }

    private setupEventListeners() {
        window.addEventListener('resize', this.onWindowResize.bind(this));
    }

    private onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.composer.setSize(window.innerWidth, window.innerHeight);
    }

    private animate() {
        this.underlay.rotation.y += 0.005;

        const time = performance.now() * 0.001;
        const color = new THREE.Color();
        color.setHSL(
            0.02 + Math.sin(time * 0.5) * 0.08,
            1,
            0.5 + Math.sin(time * 0.75) * 0.2
        );
        if (this.underlay.material instanceof THREE.MeshPhysicalMaterial) {
            this.underlay.material.color.copy(color);
        }

        this.stats.update();
        this.composer.render();
    }

    private startAnimationLoop() {
        this.renderer.setAnimationLoop(this.animate.bind(this));
    }
}

async function main() {
    new StarterScene();
}

main();
