// GlassblowPlayer.tsx
import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { Pane } from 'tweakpane';
import type { GlassState, Keyframe } from '../../animation/glassMath';
import { generateGlassProfile, lerpGlassState, copyGlassState, DEFAULT_JACKS, DEFAULT_SOFIETTA } from '../../animation/glassMath';

interface GlassblowPlayerProps {
    initialKeyframes?: Keyframe[];
    showTweakpane?: boolean;
    loadSequenceName?: string;
}

export const GlassblowPlayer: React.FC<GlassblowPlayerProps> = ({
    initialKeyframes = [],
    showTweakpane = true,
    loadSequenceName
}) => {
    const canvasRef = useRef<HTMLDivElement>(null);
    const paneRef = useRef<HTMLDivElement>(null);
    const tracksRef = useRef<HTMLDivElement>(null);

    // Three.js Refs
    const sceneRef = useRef<THREE.Scene | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const glassMeshRef = useRef<THREE.Mesh | null>(null);
    const helperMeshRef = useRef<THREE.Mesh | null>(null);
    const jacksMeshRef = useRef<THREE.Group | null>(null);
    const sofiettaMeshRef = useRef<THREE.Mesh | null>(null);

    // Mutable simulation states in Refs to avoid component re-render loops
    const keyframesRef = useRef<Keyframe[]>(initialKeyframes);
    const activeKeyframeIndexRef = useRef<number>(0);
    const editableStateRef = useRef<GlassState | null>(null);

    const isPlayingRef = useRef<boolean>(false);
    const isLoopRef = useRef<boolean>(true);
    const speedRef = useRef<number>(1.0);
    const animationTimeRef = useRef<number>(0);
    const paneInstanceRef = useRef<any>(null);

    // React state only for simple sync items in the toolbar
    const [isPlaying, setIsPlaying] = useState(false);
    const [isLoop, setIsLoop] = useState(true);
    const [speed, setSpeed] = useState(1.0);
    const [sequenceName, setSequenceName] = useState("");
    const [savedSequences, setSavedSequences] = useState<string[]>([]);
    const [selectedSeq, setSelectedSeq] = useState("");
    const frameParamsRef = useRef({ label: 'Frame 1' });

    useEffect(() => {
        if (loadSequenceName) {
            handleLoad(loadSequenceName, /*silent=*/true);
        }
    }, [loadSequenceName]);

    useEffect(() => {
        if (!canvasRef.current) return;

        // --- 1. SETUP THREEJS ---
        const scene = new THREE.Scene();
        sceneRef.current = scene;

        const camera = new THREE.PerspectiveCamera(45, canvasRef.current!.clientWidth / canvasRef.current!.clientHeight, 0.1, 100);
        camera.position.set(1.5, 2.2, 8); // Offset right and down to clear absolute top-right GUI

        const currentCanvas = canvasRef.current!; // Closure reference for reliable cleanup
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(currentCanvas.clientWidth, currentCanvas.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio); // Fix blurry resolution on High-DPI screens
        renderer.shadowMap.enabled = true;
        currentCanvas.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.target.set(0, 2, 0); // Point camera at the middle of the glass

        // Lighting - relying fully on environment map to match prototype exactly

        // Env lighting for PBR MeshPhysicalMaterial
        const pmremGenerator = new THREE.PMREMGenerator(renderer);
        const envTexture = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
        scene.environment = envTexture;
        scene.background = envTexture; // Use environment as the background flaws flawlessly

        // Grid
        const gridHelper = new THREE.GridHelper(20, 40, 0x888888, 0x444444);
        gridHelper.position.y = -0.01; // Avoid floor Z-fighting
        scene.add(gridHelper);

        // Core Glass geometry
        const glassMaterial = new THREE.MeshPhysicalMaterial({
            transmission: 1.0,   // 100% transparent like glass
            roughness: 0.02,     // Very smooth, shiny surface
            ior: 1.5,            // Standard index of refraction for glass
            thickness: 0.8,      // Helps calculate volume refraction
            color: 0xffffff,     // Base color
            side: THREE.FrontSide,// FrontSide only
            vertexColors: true   // Enable heat gradient coloring
        });
        const glassMesh = new THREE.Mesh(new THREE.BufferGeometry(), glassMaterial);
        scene.add(glassMesh);
        glassMeshRef.current = glassMesh;

        // Inside Helper mesh
        const insideMaterial = new THREE.MeshBasicMaterial({
            color: 0xff4400, wireframe: true, transparent: true, opacity: 0.3
        });
        const helperMesh = new THREE.Mesh(new THREE.BufferGeometry(), insideMaterial);
        helperMesh.visible = false; // Hide on default, only show in Wireframe mode
        scene.add(helperMesh);
        helperMeshRef.current = helperMesh;

        // --- 1.5 SETUP HAND TOOLS ---
        const toolMaterial = new THREE.MeshStandardMaterial({ color: 0x888888, metalness: 0.8, roughness: 0.2, side: THREE.DoubleSide });

        // Jacks: Group of two parallel blades
        const jacksGroup = new THREE.Group();
        const bladeGeometry = new THREE.BoxGeometry(0.1, 4.0, 0.3);
        const blade1 = new THREE.Mesh(bladeGeometry, toolMaterial);
        const blade2 = new THREE.Mesh(bladeGeometry, toolMaterial);
        blade1.position.z = -0.25;
        blade2.position.z = 0.25;
        jacksGroup.add(blade1, blade2);
        jacksGroup.visible = false;
        scene.add(jacksGroup);
        jacksMeshRef.current = jacksGroup;

        // Sofietta: Cone with tip downwards
        const sofiettaMesh = new THREE.Mesh(new THREE.ConeGeometry(1.0, 1.6, 32), toolMaterial);
        sofiettaMesh.rotation.x = Math.PI; // flip tip down
        sofiettaMesh.visible = false;
        scene.add(sofiettaMesh);
        sofiettaMeshRef.current = sofiettaMesh;

        // --- 2. SETUP INITIAL STATE ---
        if (keyframesRef.current.length === 0) {
            keyframesRef.current = [
                { // Frame 1: Standard base template
                    state: {
                        points: [
                            { y: 0.0, radius: 0.6, thickness: 0.4, baseThickness: 0.4, heat: 1.0 },
                            { y: 1.0, radius: 1.2, thickness: 0.3, heat: 1.0 },
                            { y: 2.5, radius: 1.2, thickness: 0.3, heat: 0.8 },
                            { y: 3.5, radius: 0.3, thickness: 0.2, heat: 0.3 },
                            { y: 4.5, radius: 0.3, thickness: 0.2, heat: 0.0 }
                        ],
                        rotation: 0,
                        jacks: { visible: false, position: { x: 2, y: 1, z: 0 }, rotationZ: 0, bladeSpacing: 0.5 },
                        sofietta: { visible: false, position: { x: 0, y: 5, z: 0 } }
                    }
                },
                { // Frame 2: Fluted Balloon
                    state: {
                        points: [
                            { y: 0.0, radius: 0.8, thickness: 0.2, baseThickness: 0.2, heat: 0.5 },
                            { y: 1.5, radius: 2.2, thickness: 0.1, heat: 1.0 },
                            { y: 3.0, radius: 1.5, thickness: 0.1, heat: 0.7 },
                            { y: 4.0, radius: 0.6, thickness: 0.2, heat: 0.2 },
                            { y: 4.5, radius: 0.3, thickness: 0.2, heat: 0.0 }
                        ],
                        rotation: Math.PI,
                        jacks: { visible: false, position: { x: 2, y: 1, z: 0 }, rotationZ: 0, bladeSpacing: 0.5 },
                        sofietta: { visible: false, position: { x: 0, y: 5, z: 0 } }
                    }
                }
            ];
        }
        editableStateRef.current = { points: [], rotation: 0 };
        copyGlassState(keyframesRef.current[0].state, editableStateRef.current!);

        // --- 3. SETUP TWEAKPANE ---
        let pane: Pane | null = null;
        if (showTweakpane && paneRef.current) {
            pane = new Pane({ container: paneRef.current, title: 'Glass Parameters' });
            paneInstanceRef.current = pane;
            setupTweakpane(pane);
        }

        function setupTweakpane(p: any) {
            const params = { viewMode: 'glass' };

            p.addBinding(frameParamsRef.current, 'label', { label: 'Frame Name' }).on('change', (ev: any) => {
                const kf = keyframesRef.current[activeKeyframeIndexRef.current];
                if (kf) {
                    kf.name = ev.value;
                    renderTracksUI();
                }
            });

            p.addBinding(params, 'viewMode', {
                options: { Glass: 'glass', Solid: 'solid', Wireframe: 'wireframe' }
            }).on('change', (ev: any) => {
                const mode = ev.value;
                if (glassMeshRef.current) {
                    const mat = glassMeshRef.current.material as THREE.MeshPhysicalMaterial;
                    if (mode === 'glass') {
                        mat.wireframe = false;
                        mat.transmission = 1.0;
                    } else if (mode === 'solid') {
                        mat.wireframe = false;
                        mat.transmission = 0.0;
                    } else {
                        mat.wireframe = true;
                        mat.transmission = 0.0;
                    }
                }
                if (helperMeshRef.current) {
                    helperMeshRef.current.visible = mode === 'wireframe';
                }
            });

            p.addBinding(editableStateRef.current!, 'rotation', { min: -Math.PI, max: Math.PI, step: 0.01 });
            const pts = p.addFolder({ title: 'Control Points' });
            const labels = ['Base', 'Lower Bulge', 'Upper Bulge', 'Neck', 'Rim'];
            editableStateRef.current!.points.forEach((cp: any, i: number) => {
                const label = labels[i] || `Point ${i + 1}`;
                const folder = pts.addFolder({ title: `${label}` });
                if (i === 0 && cp.baseThickness !== undefined) {
                    folder.addBinding(cp, 'baseThickness', { min: 0.05, max: 1.0, label: 'Base Thickness' });
                }
                folder.addBinding(cp, 'radius', { min: 0.1, max: 5.0, label: 'Radius' });
                folder.addBinding(cp, 'thickness', { min: 0.01, max: 0.5, label: i === 0 ? 'Radial Thickness' : 'Thickness' });
                folder.addBinding(cp, 'heat', { min: 0, max: 1, label: 'Heat' });
            });

            // --- Hand Tools Folders ---
            const est = editableStateRef.current!;
            if (!est.jacks) est.jacks = { ...DEFAULT_JACKS };
            if (!est.sofietta) est.sofietta = { ...DEFAULT_SOFIETTA };

            const jacksFolder = p.addFolder({ title: 'Jacks' });
            jacksFolder.addBinding(est.jacks, 'visible', { label: 'Visible' });
            jacksFolder.addBinding(est.jacks.position, 'x', { min: -10, max: 10, step: 0.05, label: 'Pos X' });
            jacksFolder.addBinding(est.jacks.position, 'y', { min: -10, max: 10, step: 0.05, label: 'Pos Y' });
            jacksFolder.addBinding(est.jacks.position, 'z', { min: -10, max: 10, step: 0.05, label: 'Pos Z' });
            jacksFolder.addBinding(est.jacks, 'rotationZ', { min: -Math.PI, max: Math.PI, step: 0.01, label: 'Rotation Z' });
            jacksFolder.addBinding(est.jacks, 'bladeSpacing', { min: 0.1, max: 1.5, step: 0.01, label: 'Blade Spacing' });

            const sofiettaFolder = p.addFolder({ title: 'Sofietta' });
            sofiettaFolder.addBinding(est.sofietta, 'visible', { label: 'Visible' });
            sofiettaFolder.addBinding(est.sofietta.position, 'x', { min: -10, max: 10, step: 0.05, label: 'Pos X' });
            sofiettaFolder.addBinding(est.sofietta.position, 'y', { min: -10, max: 10, step: 0.05, label: 'Pos Y' });
            sofiettaFolder.addBinding(est.sofietta.position, 'z', { min: -10, max: 10, step: 0.05, label: 'Pos Z' });
        }

        // --- 3. TOOLS UPDATER ---
        function updateTools(state: GlassState) {
            if (jacksMeshRef.current) {
                if (state.jacks) {
                    const j = state.jacks;
                    jacksMeshRef.current.position.set(j.position.x, j.position.y, j.position.z);
                    if (j.rotationZ !== undefined) jacksMeshRef.current.rotation.z = j.rotationZ;

                    const spacing = j.bladeSpacing !== undefined ? j.bladeSpacing : 0.5;
                    if (jacksMeshRef.current.children.length >= 2) {
                        jacksMeshRef.current.children[0].position.z = -spacing / 2;
                        jacksMeshRef.current.children[1].position.z = spacing / 2;
                    }

                    jacksMeshRef.current.visible = j.visible;
                } else {
                    jacksMeshRef.current.visible = false;
                }
            }
            if (sofiettaMeshRef.current) {
                if (state.sofietta) {
                    const s = state.sofietta;
                    sofiettaMeshRef.current.position.set(s.position.x, s.position.y, s.position.z);
                    sofiettaMeshRef.current.visible = s.visible;
                } else {
                    sofiettaMeshRef.current.visible = false;
                }
            }
        }

        // --- 4. RENDER LOOP ---
        const clock = new THREE.Clock();

        let animationId: number;
        function animate() {
            if (!rendererRef.current) return;
            animationId = requestAnimationFrame(animate);
            controls.update();

            const delta = clock.getDelta();
            if (isPlayingRef.current) {
                animationTimeRef.current += delta * speedRef.current;
                const totalDur = (keyframesRef.current.length - 1) * 2.0;

                if (animationTimeRef.current >= totalDur) {
                    if (isLoopRef.current) {
                        animationTimeRef.current = 0;
                    } else {
                        animationTimeRef.current = totalDur;
                        isPlayingRef.current = false;
                        setIsPlaying(false);

                        if (editableStateRef.current) {
                            copyGlassState(keyframesRef.current[activeKeyframeIndexRef.current].state, editableStateRef.current);
                            refreshPane();
                        }
                    }
                }

                // --- HIGHLIGHT TIMELINE ---
                const currentKfIndex = Math.floor(animationTimeRef.current / 2.0) + 1;
                const container = tracksRef.current;
                if (container) {
                    const tracks = container.querySelectorAll('.frame-track');
                    tracks.forEach((t: any, idx: number) => {
                        const isActive = idx === currentKfIndex;
                        t.style.background = isActive ? 'rgba(255, 87, 34, 0.15)' : '#222';
                        t.style.borderColor = isActive ? '#FF5722' : '#333';
                        t.style.color = isActive ? '#FF5722' : 'white';
                    });
                }

                const curState = getStateAtTime(animationTimeRef.current);
                if (curState && glassMeshRef.current) {
                    updateGlassGeometry(glassMeshRef.current, curState);
                    if (helperMeshRef.current) updateGlassGeometry(helperMeshRef.current, curState); // wireframe
                    updateTools(curState);

                    if (editableStateRef.current) {
                        copyGlassState(curState, editableStateRef.current);
                        refreshPane();
                    }
                }
            } else {
                if (editableStateRef.current && glassMeshRef.current) {
                    updateGlassGeometry(glassMeshRef.current, editableStateRef.current);
                    if (helperMeshRef.current) updateGlassGeometry(helperMeshRef.current, editableStateRef.current);
                    updateTools(editableStateRef.current);
                }
            }

            rendererRef.current.render(scene, camera);
        }
        animate();

        function getStateAtTime(time: number) {
            const kfs = keyframesRef.current;
            if (kfs.length < 2) return null;
            const dur = 2.0;
            const fIdx = Math.floor(time / dur);
            const nextIdx = Math.min(fIdx + 1, kfs.length - 1);
            const progress = (time % dur) / dur;
            if (fIdx === nextIdx) return kfs[fIdx].state;
            return lerpGlassState(kfs[fIdx].state, kfs[nextIdx].state, progress);
        }

        function updateGlassGeometry(mesh: THREE.Mesh, state: GlassState) {
            const profile = generateGlassProfile(state);
            const points: THREE.Vector2[] = profile.map(p => new THREE.Vector2(p.x, p.y));
            const segments = 64;
            const geometry = new THREE.LatheGeometry(points, segments);

            const colors = [];
            for (let j = 0; j <= segments; j++) {
                for (let i = 0; i < profile.length; i++) {
                    const heat = profile[i].z; // heat stored in Z by generateGlassProfile
                    const color = new THREE.Color();

                    if (heat > 0.05) {
                        // Glow: Red to Orange to Yellow
                        color.setHSL(0.02 + heat * 0.08, 1.0, 0.4 + heat * 0.2);
                    } else {
                        color.set(0xffffff);
                    }
                    colors.push(color.r, color.g, color.b);
                }
            }

            geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));

            mesh.geometry.dispose();
            mesh.geometry = geometry;
        }

        // Initial tracks render
        renderTracksUI();
        updateSavedList();

        const handleResize = () => {
            if (!canvasRef.current || !rendererRef.current) return;
            const w = canvasRef.current.clientWidth;
            const h = canvasRef.current.clientHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        };
        window.addEventListener('resize', handleResize);
        setTimeout(handleResize, 100); // Guard layout paint timing

        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
            cancelAnimationFrame(animationId);
            renderer.dispose();
            if (pane) pane.dispose();
            currentCanvas.removeChild(renderer.domElement);
        };
    }, [showTweakpane]);

    // Helpers to manage track renders with direct DOM manipulation is fine for isolating re-renders
    function renderTracksUI() {
        const container = tracksRef.current;
        if (!container) return;
        container.innerHTML = '';

        keyframesRef.current.forEach((kf, index) => {
            const div = document.createElement('div');
            div.className = `frame-track ${index === activeKeyframeIndexRef.current ? 'active' : ''}`;
            div.style.cssText = `background: ${index === activeKeyframeIndexRef.current ? 'rgba(255, 87, 34, 0.15)' : '#222'}; border: 1px solid ${index === activeKeyframeIndexRef.current ? '#FF5722' : '#333'}; padding: 6px; border-radius: 4px; flex-shrink: 0; min-width: 90px; text-align: center; cursor: pointer; font-size: 13px; color: ${index === activeKeyframeIndexRef.current ? '#FF5722' : 'white'};`;

            const name = kf.name || `Frame ${index + 1}`;
            div.innerHTML = `
                 <div>${name}</div>
                 <div style="display:flex; gap: 4px; margin-top: 4px; justify-content: center;">
                     <button class="action-btn" style="background:#444;border:none;color:white;padding:2px 4px;cursor:pointer;border-radius:2px;" onclick="window.duplicateFrame(${index})">📋</button>
                     <button class="action-btn" style="background:#d32f2f;border:none;color:white;padding:2px 4px;cursor:pointer;border-radius:2px;" onclick="window.deleteFrame(${index})">✕</button>
                 </div>
             `;

            div.addEventListener('click', (e: any) => {
                if (e.target.closest('.action-btn')) return;
                selectFrame(index);
            });

            container.appendChild(div);
        });
    }

    // Expose helpers globally or bind correctly
    (window as any).duplicateFrame = (index: number) => {
        copyGlassState(editableStateRef.current!, keyframesRef.current[activeKeyframeIndexRef.current].state);
        const source = keyframesRef.current[index];
        const deepCopy = JSON.parse(JSON.stringify(source));
        keyframesRef.current.splice(index + 1, 0, deepCopy);
        activeKeyframeIndexRef.current = index + 1;
        copyGlassState(keyframesRef.current[activeKeyframeIndexRef.current].state, editableStateRef.current!);
        refreshPane();
        renderTracksUI();
    };

    (window as any).deleteFrame = (index: number) => {
        if (keyframesRef.current.length <= 2) {
            alert("Minimum 2 keyframes required for animation.");
            return;
        }
        copyGlassState(editableStateRef.current!, keyframesRef.current[activeKeyframeIndexRef.current].state);
        keyframesRef.current.splice(index, 1);
        if (activeKeyframeIndexRef.current >= keyframesRef.current.length) {
            activeKeyframeIndexRef.current = keyframesRef.current.length - 1;
        }
        copyGlassState(keyframesRef.current[activeKeyframeIndexRef.current].state, editableStateRef.current!);
        refreshPane();
        renderTracksUI();
    };

    function selectFrame(index: number) {
        if (isPlayingRef.current) return;
        copyGlassState(editableStateRef.current!, keyframesRef.current[activeKeyframeIndexRef.current].state);
        activeKeyframeIndexRef.current = index;
        copyGlassState(keyframesRef.current[index].state, editableStateRef.current!);

        frameParamsRef.current.label = keyframesRef.current[index].name || `Frame ${index + 1}`;

        refreshPane();
        renderTracksUI();
    }

    function refreshPane() {
        if (paneInstanceRef.current) paneInstanceRef.current.refresh();
    }

    async function updateSavedList() {
        try {
            const res = await fetch('/api/sequences');
            const list = await res.json();
            setSavedSequences(list);
        } catch (e) {
            console.error("Failed to fetch sequences:", e);
        }
    }

    const handleSave = async () => {
        copyGlassState(editableStateRef.current!, keyframesRef.current[activeKeyframeIndexRef.current].state);
        const name = sequenceName.trim() || `Seq ${new Date().toLocaleTimeString()}`;
        const payload = { name, data: keyframesRef.current };

        try {
            const res = await fetch('/api/sequences', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await res.json();
            alert(result.message || `Saved "${name}"`);
            updateSavedList();
            setSequenceName("");
        } catch (e) {
            console.error("Failed to save sequence:", e);
            alert("Error saving sequence to Cloud Storage.");
        }
    };

    const handleDeleteShared = async (name: string) => {
        if (!name) return;
        if (!confirm(`Are you sure you want to DELETE "${name}" from Cloud Storage?`)) return;

        try {
            const res = await fetch(`/api/sequences/${encodeURIComponent(name)}`, { method: 'DELETE' });
            const result = await res.json();
            alert(result.message || `Deleted ${name}`);
            updateSavedList();
            setSelectedSeq("");
            if (sequenceName === name) setSequenceName("");
        } catch (e) {
            console.error("Failed to delete sequence:", e);
            alert("Error deleting sequence.");
        }
    };

    const handleLoad = async (name: string, silent: boolean = false) => {
        if (!name) return;

        try {
            const res = await fetch(`/api/sequences/${encodeURIComponent(name)}`);
            if (!res.ok) throw new Error("Failed to load");
            const data = await res.json();

            keyframesRef.current = data;
            activeKeyframeIndexRef.current = 0;
            copyGlassState(keyframesRef.current[0].state, editableStateRef.current!);
            refreshPane();
            renderTracksUI();
            setSequenceName(name); // Populate name field for easy overwrite!
        } catch (e) {
            console.error("Failed to load sequence:", e);
            if (!silent) alert("Error loading sequence.");
        }
    };

    return (
        <div style={{ position: 'relative', width: '100%', height: '600px', background: '#111', borderRadius: '8px', overflow: 'hidden' }}>
            {/* 1. Full-Bleed Canvas */}
            <div ref={canvasRef} style={{ width: '100%', height: '100%' }} />

            {/* 2. Absolute Tweakpane Overlay */}
            <div ref={paneRef} style={{ 
                position: 'absolute', 
                top: '12px', 
                right: '12px', 
                width: '240px', 
                background: 'rgba(28, 28, 28, 0.85)', 
                backdropFilter: 'blur(4px)',
                borderRadius: '6px',
                border: '1px solid rgba(255,255,255,0.05)',
                display: showTweakpane ? 'block' : 'none', 
                maxHeight: 'calc(100% - 24px)', 
                overflowY: 'auto',
                zIndex: 10 
            }} />

            {/* 3. Absolute Timeline Row */}
            <div style={{ 
                position: 'absolute', 
                bottom: '12px', 
                left: '12px', 
                width: 'calc(100% - 300px)', 
                padding: '8px', 
                background: 'rgba(26, 26, 26, 0.85)', 
                backdropFilter: 'blur(4px)',
                borderRadius: '6px',
                border: '1px solid rgba(255,255,255,0.05)',
                display: 'flex', 
                flexDirection: 'column', 
                gap: '8px',
                zIndex: 10
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', width: '100%' }}>
                    <button
                        onClick={() => {
                            if (!isPlayingRef.current) {
                                const totalDur = (keyframesRef.current.length - 1) * 2.0;
                                if (animationTimeRef.current >= totalDur) {
                                    animationTimeRef.current = 0;
                                }
                            }
                            isPlayingRef.current = !isPlayingRef.current;
                            setIsPlaying(isPlayingRef.current);

                            if (!isPlayingRef.current) {
                                if (editableStateRef.current) {
                                    copyGlassState(keyframesRef.current[activeKeyframeIndexRef.current].state, editableStateRef.current);
                                    refreshPane();
                                }
                                renderTracksUI(); // Clear dynamic highlights
                            }
                        }}
                        style={{ padding: '4px 8px', background: isPlaying ? '#FF5722' : '#222', color: 'white', border: '1px solid #444', cursor: 'pointer', borderRadius: '4px', fontSize: '13px' }}
                    >
                        {isPlaying ? '⏸' : '▶'}
                    </button>
                    <div ref={tracksRef} style={{ display: 'flex', flexGrow: 1, gap: '8px', overflowX: 'auto', paddingBottom: '2px' }} />
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '6px' }}>
                    <label style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#ccc' }}>
                        <input type="checkbox" checked={isLoop} onChange={(e) => { setIsLoop(e.target.checked); isLoopRef.current = e.target.checked; }} style={{ accentColor: '#FF5722' }} /> Loop
                    </label>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: '120px' }}>
                        <span style={{ fontSize: '12px', color: '#aaa' }}>Speed:</span>
                        <input type="range" min="0.1" max="4.0" step="0.1" value={speed} onChange={(e) => { const s = parseFloat(e.target.value); setSpeed(s); speedRef.current = s; }} style={{ flexGrow: 1 }} />
                    </div>

                    <div style={{ display: 'flex', gap: '6px', marginLeft: 'auto', alignItems: 'center' }}>
                        <input type="text" placeholder="Name..." value={sequenceName} onChange={(e) => setSequenceName(e.target.value)} style={{ background: '#222', border: '1px solid #444', color: 'white', borderRadius: '4px', padding: '3px 6px', fontSize: '12px', width: '90px' }} />
                        <button onClick={handleSave} style={{ background: 'rgba(76,175,80,0.15)', border: '1px solid #4CAF50', color: '#4CAF50', padding: '3px 6px', cursor: 'pointer', borderRadius: '4px', fontSize: '12px' }}>Save</button>
                        <select
                            value={selectedSeq}
                            onChange={(e) => {
                                const named = e.target.value;
                                setSelectedSeq(named);
                                if (named) handleLoad(named);
                            }}
                            style={{ background: '#222', border: '1px solid #444', color: 'white', padding: '3px', fontSize: '12px', borderRadius: '4px' }}
                        >
                            <option value="">-- Load --</option>
                            {savedSequences.map(name => <option key={name} value={name}>{name}</option>)}
                        </select>
                        <button
                            onClick={() => handleDeleteShared(selectedSeq)}
                            disabled={!selectedSeq}
                            style={{ background: 'rgba(211, 47, 47, 0.15)', border: '1px solid #d32f2f', color: '#d32f2f', padding: '3px 6px', cursor: 'pointer', borderRadius: '4px', fontSize: '12px', opacity: selectedSeq ? 1 : 0.5 }}
                        >🗑️</button>
                    </div>
                </div>
            </div>
        </div>
    );
};
