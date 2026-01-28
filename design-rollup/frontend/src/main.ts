import initWasmModule, { GlassSimulation } from './pkg/simulation.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// 1. Explicit types for globals
// The '!:' tells TS these will be initialized later (in init) so we don't need to make them nullable.
let camera: THREE.PerspectiveCamera;
let scene: THREE.Scene;
let renderer: THREE.WebGLRenderer;
let mesh: THREE.Mesh;
let material: THREE.MeshBasicMaterial; // Specific type needed to access .map property later
let orbitControls: OrbitControls;
let glassSim: GlassSimulation;

// Interaction State
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let previewRing: THREE.LineLoop;
let selectionRing: THREE.LineLoop;
let selectedY: number | null = null;

// Interaction Modes
// let isStretching = false; // Deprecated in favor of keyboard controls

const drawStartPos = new THREE.Vector2();

main();

async function main() {
    await init();
    setupCanvasDrawing();
}

async function init() {
    await initWasm();

    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 2000);
    camera.position.z = 500;

    scene = new THREE.Scene();

    const glassCapMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
    material = new THREE.MeshBasicMaterial();
    material.side = THREE.DoubleSide;

    // Initialize Rust Simulation
    // Radius 100, Height 200, 64 segments
    glassSim = new GlassSimulation(100, 200, 64);
    console.log("GlassSimulation created. Points:", glassSim.get_points());

    // Initial Geometry from Simulation
    const geometry = createGeometryFromSimulation();
    console.log("Geometry created. Vertex count:", geometry.attributes.position.count);

    // 2. Create a Wireframe geometry from the original geometry
    const wireframeGeometry = new THREE.WireframeGeometry(geometry);

    // 3. Create a LineSegments object
    const wireframe = new THREE.LineSegments(wireframeGeometry);
    const wireframeScale = 1.01;
    wireframe.scale.set(wireframeScale, wireframeScale, wireframeScale);
    wireframe.material = new THREE.LineBasicMaterial({
        color: 0xff0000, // Red wireframe for visibility
        linewidth: 1
    });

    // mesh = new THREE.Mesh( new THREE.BoxGeometry( 200, 200, 200 ), material );
    mesh = new THREE.Mesh(geometry, [material, glassCapMaterial, glassCapMaterial]);
    mesh.add(wireframe);
    scene.add(mesh);

    // BoxHelper to visualize bounds
    const boxHelper = new THREE.BoxHelper(mesh, 0xffff00);
    scene.add(boxHelper);

    // Initialize Rings
    const ringGeo = new THREE.RingGeometry(1, 1.1, 64);
    // Actually LineLoop is better for a thin wireframe ring
    const lineGeo = new THREE.BufferGeometry().setFromPoints(
        new THREE.Path().absarc(0, 0, 1, 0, Math.PI * 2).getPoints(64)
    );

    // Preview Ring (Faint, Gray)
    previewRing = new THREE.LineLoop(lineGeo, new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.5 }));
    previewRing.rotation.x = Math.PI / 2; // Flat on XZ plane (since Y is up)
    previewRing.visible = false;
    scene.add(previewRing);

    // Selection Ring (Bright, Yellow)
    selectionRing = new THREE.LineLoop(lineGeo, new THREE.LineBasicMaterial({ color: 0xffff00, linewidth: 2 }));
    selectionRing.rotation.x = Math.PI / 2;
    selectionRing.visible = false;
    scene.add(selectionRing);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setClearColor(0xf0f0f0); // Light Gray to see transparent textures
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setAnimationLoop(animate);
    document.body.appendChild(renderer.domElement);

    window.addEventListener('resize', onWindowResize);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('click', onClick);

    // Keyboard Controls for Physics
    window.addEventListener('keydown', (e) => {
        switch (e.key.toLowerCase()) {
            case 'j': // Jack
                if (selectedY !== null) {
                    console.log(`Applying Jack at Y=${selectedY.toFixed(2)}`);
                    glassSim.apply_jack(selectedY, 0.1, 10.0); // Reduced sigma for sharper cut
                    updateGeometryFromSimulation();
                } else {
                    console.log("No height selected for Jack. Click on the glass first.");
                }
                break;
                break;
            case 'arrowup':
                if (selectedY !== null) {
                    const offset = 5.0;
                    console.log(`Stretching Up at Y=${selectedY.toFixed(2)}`);
                    glassSim.apply_boundary_stretch(selectedY, offset); // Up

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
                    console.log(`Stretching Down at Y=${selectedY.toFixed(2)}`);
                    glassSim.apply_boundary_stretch(selectedY, offset); // Down

                    // Update selection to follow the moving part
                    selectedY += offset;
                    selectionRing.position.y = selectedY;

                    updateGeometryFromSimulation();
                } else {
                    console.log("No height selected. Click on the glass first.");
                }
                break;
        }
    });

    orbitControls = new OrbitControls(camera, renderer.domElement);

    const sideMaterial = (mesh.material as THREE.Material[])[0];
    connectToFigmaBridge(sideMaterial as THREE.MeshBasicMaterial);
}

function onPointerMove(event: PointerEvent) {
    // Calculate pointer position in normalized device coordinates (-1 to +1)
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);

    const intersects = raycaster.intersectObject(mesh);

    if (intersects.length > 0) {
        const hit = intersects[0];
        const y = hit.point.y;

        // Find radius at this Y (simple interpolation or nearest)
        const radius = getRadiusAtY(y);

        previewRing.visible = true;
        previewRing.position.y = y;
        previewRing.scale.set(radius, radius, 1); // Scale X/Y of the ring geometry (which is on XZ plane after rotation)
    } else {
        previewRing.visible = false;
    }
}

function onClick(event: Event) {
    // We can reuse the raycaster from pointer move if we want, or re-cast.
    // Re-casting is safer for exact click pos.
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObject(mesh);

    if (intersects.length > 0) {
        const hit = intersects[0];
        selectedY = hit.point.y;

        const radius = getRadiusAtY(selectedY);

        selectionRing.visible = true;
        selectionRing.position.y = selectedY;
        selectionRing.scale.set(radius, radius, 1);

        console.log(`Selected Y: ${selectedY}`);
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

    // Add thickness implementation if needed, for now just single layer or similar logic to before
    // The previous updateLatheGeometry added thickness. Let's replicate that simple thickness logic
    // or just stick to the simulation profile for now.
    // The previous code had: R-thick, R.
    // Let's keep it simple first: Just the outer shell?
    // Or replicate the thickness loop.

    const thickness = 2.0;
    const finalPoints: THREE.Vector2[] = [];

    // Inner wall (reverse order for correct winding if we want double sided)
    // Actually LatheGeometry just spins the profile.
    // Previous logic:
    // points.push(new THREE.Vector2(radius - thickness, height / 2));
    // ...
    // It was a simple box profile.

    // New logic: We have a full curve.
    // Let's just output the curve for now.
    // If we want thickness, we need to generate an inner curve.

    // Simplified: Just use the profile directly.
    return new THREE.LatheGeometry(points, 64);
}

function updateGeometryFromSimulation() {
    const newGeometry = createGeometryFromSimulation();
    newGeometry.computeBoundingSphere();

    mesh.geometry.dispose();
    mesh.geometry = newGeometry;

    // Update wireframe
    const wireframe = mesh.children.find(c => c.type === 'LineSegments') as THREE.LineSegments;
    if (wireframe) {
        wireframe.geometry.dispose();
        wireframe.geometry = new THREE.WireframeGeometry(newGeometry);
    }
}

function connectToFigmaBridge(targetMaterial: THREE.MeshBasicMaterial) {
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

            console.log('Texture updated from Figma');
            // We do NOT update geometry from Figma anymore, physics drives geometry.
            // Or maybe Figma drives the valid shape? 
            // For now, let's decouple them: Physics sets shape, Figma sets texture.
        });
    };
}

function setupCanvasDrawing(): void {

    // 2. Type assertion for DOM element
    // We cast to HTMLCanvasElement because getElementById returns generic HTMLElement | null
    const drawingCanvas = document.getElementById('drawing-canvas') as HTMLCanvasElement;

    // Safety check: ensure canvas exists before continuing
    if (!drawingCanvas) {
        console.error("Canvas element 'drawing-canvas' not found");
        return;
    }

    const drawingContext = drawingCanvas.getContext('2d');

    if (!drawingContext) return; // Safety check for context

    // draw white background
    drawingContext.fillStyle = '#FFFFFF';
    drawingContext.fillRect(0, 0, 256, 128);

    // set canvas as material.map
    material.map = new THREE.CanvasTexture(drawingCanvas);

    let paint = false;

    // 3. Typed event listeners
    drawingCanvas.addEventListener('pointerdown', function (e: PointerEvent) {

        paint = true;
        drawStartPos.set(e.offsetX, e.offsetY);

    });

    drawingCanvas.addEventListener('pointermove', function (e: PointerEvent) {

        if (paint) draw(drawingContext, e.offsetX, e.offsetY);

    });

    drawingCanvas.addEventListener('pointerup', function () {

        paint = false;

    });

    drawingCanvas.addEventListener('pointerleave', function () {

        paint = false;

    });

}

function draw(drawContext: CanvasRenderingContext2D, x: number, y: number): void {

    drawContext.moveTo(drawStartPos.x, drawStartPos.y);
    drawContext.strokeStyle = '#000000';
    drawContext.lineTo(x, y);
    drawContext.stroke();

    // reset drawing start position to current position.
    drawStartPos.set(x, y);

    // 4. Non-null assertion (!)
    // We know map exists because we set it in setupCanvasDrawing, but TS sees it as optional on the material.
    if (material.map) material.map.needsUpdate = true;

}

function onWindowResize(): void {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);

}

function animate(): void {

    // mesh.rotation.x += 0.01;
    // mesh.rotation.y += 0.01;

    renderer.render(scene, camera);

}