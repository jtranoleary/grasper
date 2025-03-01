import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

import init, { greet } from "./pkg/simulation";

import {
    EffectComposer,
    RenderPass,
    SelectiveBloomEffect,
    BlendFunction,
    EffectPass,
} from 'postprocessing';

const hdrTextureURL = new URL('/hdri/machine_shop.hdr', import.meta.url).href;

class GlassRendering {
    private camera!: THREE.PerspectiveCamera;
    private scene!: THREE.Scene;
    private renderer!: THREE.WebGLRenderer;
    private stats!: Stats;
    private gui!: GUI;
    private composer!: EffectComposer;
    private controls!: OrbitControls;
    private bloomPass!: SelectiveBloomEffect;
    private floor!: THREE.Mesh;
    private particles!: THREE.Points;

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

        const light = new THREE.DirectionalLight(0xffffff, /*intensity=*/1);
        light.position.set(1, 3, 2);
        this.scene.add(light);
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
    }

    private createParticles() {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const numParticles = 10000;
        const boundingBoxDim = 5;
        const yOffset = 2;

        for (let i = 0; i < numParticles; i++) {
            positions.push((Math.random() - 0.5) * boundingBoxDim); // x
            positions.push((Math.random() - 0.5) * boundingBoxDim + yOffset); // y
            positions.push((Math.random() - 0.5) * boundingBoxDim); // z
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

        const texture = new THREE.TextureLoader().load('textures/sprites/disc.png'); // Load the texture

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
        const geometry = new THREE.PlaneGeometry(10, 10);
        const material = new THREE.MeshPhysicalMaterial({
            color: 0x808080,
            roughness: 0.1,
            metalness: 0.2,
            reflectivity: 1,
            side: THREE.DoubleSide
        });
        const floor = new THREE.Mesh(geometry, material);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -1;
        this.floor = floor;
        this.scene.add(floor);
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
        if (mesh instanceof THREE.Mesh && (mesh.material instanceof THREE.MeshStandardMaterial||
          mesh.material instanceof THREE.MeshPhysicalMaterial)) {
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
        this.stats.update();
        this.composer.render();
    }

    private startAnimationLoop() {
        this.renderer.setAnimationLoop(this.animate.bind(this));
    }
}

async function main() {
    new GlassRendering();
    await init();
    const greeting = greet();
    console.log(greeting);
}

main();
