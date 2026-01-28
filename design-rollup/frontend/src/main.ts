import initWasmModule, { GlassSimulation } from './pkg/simulation.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let camera: THREE.PerspectiveCamera;
let scene: THREE.Scene;
let renderer: THREE.WebGLRenderer;
let mesh: THREE.Mesh;
let material: THREE.MeshStandardMaterial;
let glassSim: GlassSimulation;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let previewRing: THREE.LineLoop;
let selectionRing: THREE.LineLoop;
let selectedY: number | null = null;

main();

async function main() {
    await init();
}

async function init() {
    await initWasm();

    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 2000);
    camera.position.z = 500;

    scene = new THREE.Scene();


    material = new THREE.MeshStandardMaterial({ color: 0xffffff });
    material.side = THREE.DoubleSide;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(100, 100, 100);
    scene.add(directionalLight);

    glassSim = new GlassSimulation(100, 200, 64);
    console.log("GlassSimulation created. Points:", glassSim.get_points());

    const geometry = createGeometryFromSimulation();
    console.log("Geometry created. Vertex count:", geometry.attributes.position.count);
    console.log("Geometry groups:", geometry.groups.length);
    console.log("Geometry UVs:", geometry.attributes.uv ? "Present" : "Missing");



    const wireframeGeometry = new THREE.WireframeGeometry(geometry);
    const wireframe = new THREE.LineSegments(wireframeGeometry);
    const wireframeScale = 1.01;
    wireframe.scale.set(wireframeScale, wireframeScale, wireframeScale);
    wireframe.material = new THREE.LineBasicMaterial({
        color: 0xff0000,
        linewidth: 1
    });

    mesh = new THREE.Mesh(geometry, material);
    mesh.add(wireframe);
    scene.add(mesh);

    const boxHelper = new THREE.BoxHelper(mesh, 0xffff00);
    scene.add(boxHelper);

    const axesHelper = new THREE.AxesHelper(500);
    scene.add(axesHelper);

    const gridHelper = new THREE.GridHelper(1000, 50);
    scene.add(gridHelper);

    const lineGeo = new THREE.BufferGeometry().setFromPoints(
        new THREE.Path().absarc(0, 0, 1, 0, Math.PI * 2).getPoints(64)
    );

    previewRing = new THREE.LineLoop(lineGeo, new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.5 }));
    previewRing.rotation.x = Math.PI / 2; // Flat on XZ plane (since Y is up)
    previewRing.visible = false;
    scene.add(previewRing);

    selectionRing = new THREE.LineLoop(lineGeo, new THREE.LineBasicMaterial({ color: 0xffff00, linewidth: 2 }));
    selectionRing.rotation.x = Math.PI / 2;
    selectionRing.visible = false;
    scene.add(selectionRing);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setAnimationLoop(animate);
    document.body.appendChild(renderer.domElement);

    window.addEventListener('resize', onWindowResize);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('click', onClick);

    window.addEventListener('keydown', (e) => {
        switch (e.key.toLowerCase()) {
            case 'j': // Jack
                if (selectedY !== null) {
                    glassSim.apply_jack(selectedY, 0.1, 10.0);
                    updateGeometryFromSimulation();
                } else {
                    console.log("No height selected for Jack. Click on the glass first.");
                }
                break;
            case 'arrowup':
                if (selectedY !== null) {
                    const offset = 5.0;
                    glassSim.apply_boundary_stretch(selectedY, offset);

                    // Update selection to follow the moving part
                    selectedY += offset;
                    selectionRing.position.y = selectedY;

                    updateGeometryFromSimulation();
                } else {
                    console.log("No height selected. Click on the glass first.");
                }
                break;
            case 'arrowdown':
                if (selectedY !== null) {
                    const offset = -5.0;
                    glassSim.apply_boundary_stretch(selectedY, offset);

                    selectedY += offset;
                    selectionRing.position.y = selectedY;

                    updateGeometryFromSimulation();
                } else {
                    console.log("No height selected. Click on the glass first.");
                }
                break;
        }
    });

    new OrbitControls(camera, renderer.domElement);

    const sideMaterial = mesh.material as THREE.MeshStandardMaterial;
    connectToFigmaBridge(sideMaterial);
}

function onPointerMove(event: PointerEvent) {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);

    const intersects = raycaster.intersectObject(mesh);

    if (intersects.length > 0) {
        const hit = intersects[0];
        const y = hit.point.y;

        const radius = getRadiusAtY(y);

        previewRing.visible = true;
        previewRing.position.y = y;
        previewRing.scale.set(radius, radius, 1);
    } else {
        previewRing.visible = false;
    }
}

function onClick(event: Event) {
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObject(mesh);

    if (intersects.length > 0) {
        const hit = intersects[0];
        selectedY = hit.point.y;

        const radius = getRadiusAtY(selectedY);

        selectionRing.visible = true;
        selectionRing.position.y = selectedY;
        selectionRing.scale.set(radius, radius, 1);
    }
}

function getRadiusAtY(y: number): number {
    const points = glassSim.get_points();
    // Points: [r0, y0, r1, y1, ...]
    // Search for segment containing y

    // Simple nearest neighbor or linear interp
    let closestR = 100;
    let minDiff = Infinity;

    for (let i = 0; i < points.length; i += 2) {
        const py = points[i + 1];
        const diff = Math.abs(y - py);
        if (diff < minDiff) {
            minDiff = diff;
            closestR = points[i];
        }
    }
    return closestR + 2; // +2 for slight visual offset outside
}

async function initWasm() {
    await initWasmModule();
}

function createGeometryFromSimulation(): THREE.LatheGeometry {
    const rawPoints = glassSim.get_points();
    const points: THREE.Vector2[] = [];

    // Points are [r0, y0, r1, y1...]
    for (let i = 0; i < rawPoints.length; i += 2) {
        points.push(new THREE.Vector2(rawPoints[i], rawPoints[i + 1]));
    }


    const geometry = new THREE.LatheGeometry(points, 64);
    applyCylindricalUVs(geometry);

    return geometry;
}

function applyCylindricalUVs(geometry: THREE.BufferGeometry) {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox!;
    const minY = box.min.y;
    const rangeY = box.max.y - minY;

    const posAttribute = geometry.attributes.position;
    const uvAttribute = geometry.attributes.uv || new THREE.BufferAttribute(new Float32Array(posAttribute.count * 2), 2);

    for (let i = 0; i < posAttribute.count; i++) {
        const x = posAttribute.getX(i);
        const y = posAttribute.getY(i);
        const z = posAttribute.getZ(i);

        // Cylindrical mapping
        // Angle from -PI to PI
        const angle = Math.atan2(x, z);
        // Normalize angle to 0..1
        // atan2 returns angle in radians. We want 0 at one seam.
        // x=sin, z=cos usually.
        // U = (angle / (2 * PI)) + 0.5
        const u = (angle / (2 * Math.PI)) + 0.5;

        // V = Normalized Height
        const v = (y - minY) / rangeY;

        uvAttribute.setXY(i, u, v);
    }

    geometry.setAttribute('uv', uvAttribute);
    geometry.attributes.uv.needsUpdate = true;
}

function updateGeometryFromSimulation() {
    const newGeometry = createGeometryFromSimulation();
    newGeometry.computeBoundingSphere();

    mesh.geometry.dispose();
    mesh.geometry = newGeometry;

    const wireframe = mesh.children.find(c => c.type === 'LineSegments') as THREE.LineSegments;
    if (wireframe) {
        wireframe.geometry.dispose();
        wireframe.geometry = new THREE.WireframeGeometry(newGeometry);
    }
}

function connectToFigmaBridge(targetMaterial: THREE.MeshStandardMaterial) {
    const ws = new WebSocket('ws://localhost:3002');

    ws.onopen = () => {
        console.log('Connected to Figma Bridge');
    };

    ws.onmessage = async (event) => {
        const blob = event.data;
        const url = URL.createObjectURL(blob as Blob);

        new THREE.TextureLoader().load(url, (texture) => {
            if (targetMaterial.map) targetMaterial.map.dispose();

            texture.colorSpace = THREE.SRGBColorSpace;
            texture.flipY = true;

            targetMaterial.map = texture;
            targetMaterial.needsUpdate = true;
            targetMaterial.map.needsUpdate = true; // Ensure texture updates

            console.log('Texture updated from Figma. Image info:', texture.image);
        });
    };
}

function onWindowResize(): void {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate(): void {
    renderer.render(scene, camera);
}