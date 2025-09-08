import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

import init, { Simulation } from "./pkg/simulation";

import {
    EffectComposer,
    RenderPass,
    SelectiveBloomEffect,
    BlendFunction,
    EffectPass,
} from 'postprocessing';

const hdrTextureURL = new URL('/hdri/machine_shop.hdr', import.meta.url).href;

class GlassRendering {
    boundingBoxDim: number;
    private camera!: THREE.PerspectiveCamera;
    private scene!: THREE.Scene;
    private renderer!: THREE.WebGLRenderer;
    private stats!: Stats;
    private gui!: GUI;
    private composer!: EffectComposer;
    private orbitControls!: OrbitControls;
    private transformControls!: TransformControls;
    private bloomPass!: SelectiveBloomEffect;
    private floor!: THREE.Mesh;
    private particles!: THREE.Points;
    private simulation!: Simulation;

    private pipe!: THREE.Mesh;
    private pipeHeight!: number;
    private pipeRadius!: number;
    private pipeRPM!: number;

    constructor(boundingBoxDim: number) {
        this.boundingBoxDim = boundingBoxDim;
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
        this.setupEventListeners();

        this.setupSimulation();
        this.setupGUI();
        this.startAnimationLoop();
    }

    private createScene() {
        this.scene = new THREE.Scene();
    }

    private createCamera() {
        this.camera = new THREE.PerspectiveCamera(
            50, window.innerWidth / window.innerHeight, 0.1, 100);
        this.camera.position.set(-10, 10, 10);
        this.camera.layers.enable(0);
        this.camera.layers.enable(1);
    }

    private createRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            powerPreference: "high-performance",
            antialias: false,
            stencil: false,
            depth: false,
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
        this.createParticles();
        this.createFloor();
        this.createPipe();
    }

    private createParticles() {
        const geometry = new THREE.BufferGeometry();
        const positions: number[] = [];

        geometry.setAttribute('position',
            new THREE.Float32BufferAttribute(positions, 3));

        const texture = new THREE.TextureLoader()
            .load('textures/sprites/disc.png');

        const material = new THREE.PointsMaterial({
            size: 0.1,
            color: 0xffffff,
            map: texture,
            transparent: true,
            alphaTest: 0.5,
        });


        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
    }

    private createFloor() {
        const geometry = new THREE.PlaneGeometry(this.boundingBoxDim,
                                                 this.boundingBoxDim);
        const material = new THREE.MeshPhysicalMaterial({
            color: 0x808080,
            roughness: 0.1,
            metalness: 0.2,
            reflectivity: 1,
            side: THREE.DoubleSide
        });
        const floor = new THREE.Mesh(geometry, material);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -0.1;
        this.floor = floor;
        this.scene.add(floor);
    }

    private createPipe() {
        this.pipeRadius = 0.25;
        this.pipeHeight = 4.0;
        this.pipeRPM = 10.0;

        const geometry = new THREE.CylinderGeometry(this.pipeRadius,
            this.pipeRadius, this.pipeHeight, 32);
        const material = new THREE.MeshStandardMaterial({
            color: 0xcccccc,
            metalness: 0.8,
            roughness: 0.2,
        });
        this.pipe = new THREE.Mesh(geometry, material);
        this.pipe.position.y = 4.0;
        this.scene.add(this.pipe);
    }

    private setupPostProcessing() {
      this.bloomPass = new SelectiveBloomEffect(this.scene, this.camera, {
          blendFunction: BlendFunction.ADD,
          mipmapBlur: true,
          intensity: 0.75,
      });
      this.bloomPass.inverted = true;
      this.bloomPass.ignoreBackground = true;

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

                this.updateMaterialEnvMap(this.particles, envMap);
                this.updateMaterialEnvMap(this.floor, envMap);
            });
    }

    private updateMaterialEnvMap(mesh: THREE.Object3D, envMap: THREE.Texture) {
        if (mesh instanceof THREE.Mesh &&
            (mesh.material instanceof THREE.MeshStandardMaterial ||
             mesh.material instanceof THREE.MeshPhysicalMaterial)) {
            mesh.material.envMap = envMap;
            mesh.material.needsUpdate = true;
        }
    }

    private setupControls() {
        this.orbitControls = new OrbitControls(this.camera,
            this.renderer.domElement);

        this.transformControls = new TransformControls(this.camera,
            this.renderer.domElement);
        this.transformControls.attach(this.pipe);
        this.scene.add(this.transformControls.getHelper());

        // Disable OrbitControls when using TransformControls
        this.transformControls.addEventListener('dragging-changed', (event) => {
            this.orbitControls.enabled = !event.value;
        });
        this.transformControls.addEventListener('change',
            () => this.renderer.render(this.scene, this.camera));

        window.addEventListener('keydown', (event) => {
            switch (event.key) {
                case 't':
                    this.transformControls.setMode('translate');
                    break;
                case 'r':
                    this.transformControls.setMode('rotate');
                    break;
            }
        });
    }

    private setupGUI() {
        this.stats = new Stats();
        document.body.appendChild(this.stats.dom);
        this.gui = new GUI();

        const bloomFolder = this.gui.addFolder('Bloom');
        bloomFolder.add(this.bloomPass, 'intensity', 0, 3)
            .name('Intensity');
        bloomFolder.add(this.bloomPass.luminanceMaterial, 'threshold', 0, 1)
            .name('Threshold');
        bloomFolder.add(this.bloomPass.luminanceMaterial, 'smoothing', 0, 1)
            .name('Smoothing');
        bloomFolder.add(this.bloomPass.mipmapBlurPass, 'radius', 0.1, 2, 0.01)
            .name('Mipmap Blur Radius');
        bloomFolder.close();

        const fluidFolder = this.gui.addFolder('Fluid');

        const actions = {
            resetParticles: () => this.simulation.reset_particles()
        };

        fluidFolder.add(this.simulation, 'stiffness').name('Stiffness');
        fluidFolder.add(this.simulation, 'viscosity', 0.01, 0.05, 0.01).name('Viscosity')
            .onChange((value: number) => {
                    this.simulation.viscosity = value;
                });
        fluidFolder.add(actions, 'resetParticles').name('Reset Particles');

        const pipeFolder = this.gui.addFolder('Pipe Controls');
        pipeFolder.add(this, 'pipeRPM', 0, 60, 1).name('Spin RPM');
        pipeFolder.open();
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

    private updateParticlePositions() {
        this.simulation.update();
        const positions = this.simulation.get_particle_positions();
        this.particles.geometry.setAttribute('position',
            new THREE.Float32BufferAttribute(positions, 3));

        this.particles.geometry.attributes.position.needsUpdate = true;
    }

    private setupSimulation() {
        this.simulation = new Simulation(this.boundingBoxDim);
    }

    private updateAndSendPipePosition(delta: number) {
        // 1. Apply the automatic spin based on RPM
        const rotationPerSecond = (this.pipeRPM / 60.0) * Math.PI * 2;
        this.pipe.rotateY(rotationPerSecond * delta); // Rotate around its own Y-axis

        // 2. Calculate the pipe's world transform
        this.pipe.updateWorldMatrix(true, false);
        const pipeHeight = this.pipeHeight;
        const tip = new THREE.Vector3(0, -pipeHeight / 2, 0).applyMatrix4(this.pipe.matrixWorld);
        const end = new THREE.Vector3(0, pipeHeight / 2, 0).applyMatrix4(this.pipe.matrixWorld);

        // Extract the final world orientation as a quaternion
        const orientation = new THREE.Quaternion();
        this.pipe.getWorldQuaternion(orientation);

        // 3. Send the full transform to the Rust simulation
        this.simulation.update_pipe_transform(
            tip.x, tip.y, tip.z,
            end.x, end.y, end.z,
            orientation.x, orientation.y, orientation.z, orientation.w
        );
    }

    private animate(delta: number) {
        this.stats.update();
        this.updateAndSendPipePosition(delta);
        this.updateParticlePositions();
        this.composer.render();
    }

    private startAnimationLoop() {
        let lastTime = 0;
        const animateLoop = (time: number) => {
            const delta = (time - lastTime) / 1000.0;
            lastTime = time;
            this.animate(delta || 0);
        };
        this.renderer.setAnimationLoop(animateLoop);
    }
}

async function main() {
    await init();
    new GlassRendering(5.0);
}

main();
