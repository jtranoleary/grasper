import * as THREE from 'three';

export interface ControlPoint {
    y: number;
    radius: number;
    thickness: number;
    heat: number;
    baseThickness?: number; // optional, for index 0
}

export interface ToolState {
    visible: boolean;
    position: { x: number; y: number; z: number };
    rotationZ?: number; // for jacks
    bladeSpacing?: number; // for jacks
}

export interface GlassState {
    points: ControlPoint[];
    rotation: number;
    jacks?: ToolState;
    sofietta?: ToolState;
}

export const DEFAULT_JACKS: ToolState = { visible: false, position: { x: 2, y: -1, z: 0 }, rotationZ: 0, bladeSpacing: 0.5 };
export const DEFAULT_SOFIETTA: ToolState = { visible: false, position: { x: 0, y: 5, z: 0 } };

export interface Keyframe {
    state: GlassState;
    name?: string;
}

// --- 2. PROFILE GENERATOR ---

export function generateGlassProfile(state: GlassState, resolution: number = 50): THREE.Vector3[] {
    const outerPoints: THREE.Vector3[] = [];
    const innerPoints: THREE.Vector3[] = [];

    for (let i = 0; i < state.points.length; i++) {
        const p = state.points[i];
        outerPoints.push(new THREE.Vector3(p.radius, p.y, p.heat));
        const innerR = Math.max(0.001, p.radius - p.thickness);

        const bThickness = (i === 0 && p.baseThickness !== undefined) ? p.baseThickness : p.thickness;
        const innerY = i === 0 ? p.y + bThickness : p.y;
        innerPoints.push(new THREE.Vector3(innerR, innerY, p.heat));
    }

    const outerSpline = new THREE.CatmullRomCurve3(outerPoints);
    const innerSpline = new THREE.CatmullRomCurve3(innerPoints);

    const smoothOuter = outerSpline.getPoints(resolution);
    const smoothInner = innerSpline.getPoints(resolution);

    const profilePoints: THREE.Vector3[] = [];

    // Prepend Outer Center Bottom on axis
    profilePoints.push(new THREE.Vector3(0, smoothOuter[0].y, smoothOuter[0].z));

    for (let i = 0; i <= resolution; i++) {
        profilePoints.push(smoothOuter[i]);
    }

    for (let i = resolution; i >= 0; i--) {
        profilePoints.push(smoothInner[i]);
    }

    // Append Inner Center Bottom on axis to seal volume
    profilePoints.push(new THREE.Vector3(0, smoothInner[0].y, smoothInner[0].z));

    return profilePoints;
}

// --- 3. MATH HELPERS ---

export function lerpControlPoint(a: ControlPoint, b: ControlPoint, t: number): ControlPoint {
    const res: ControlPoint = {
        y: a.y + (b.y - a.y) * t,
        radius: a.radius + (b.radius - a.radius) * t,
        thickness: a.thickness + (b.thickness - a.thickness) * t,
        heat: a.heat + (b.heat - a.heat) * t
    };
    if (a.baseThickness !== undefined && b.baseThickness !== undefined) {
        res.baseThickness = a.baseThickness + (b.baseThickness - a.baseThickness) * t;
    }
    return res;
}

export function lerpGlassState(a: GlassState, b: GlassState, t: number): GlassState {
    const points: ControlPoint[] = [];
    for (let i = 0; i < a.points.length; i++) {
        points.push(lerpControlPoint(a.points[i], b.points[i], t));
    }

    const lerpTool = (t1?: ToolState, t2?: ToolState, def?: ToolState): ToolState | undefined => {
        const s1 = t1 || def;
        const s2 = t2 || def;
        if (!s1 || !s2) return undefined;
        return {
            visible: t < 0.5 ? s1.visible : s2.visible,
            position: {
                x: s1.position.x + (s2.position.x - s1.position.x) * t,
                y: s1.position.y + (s2.position.y - s1.position.y) * t,
                z: s1.position.z + (s2.position.z - s1.position.z) * t
            },
            rotationZ: s1.rotationZ !== undefined && s2.rotationZ !== undefined ?
                       s1.rotationZ + (s2.rotationZ - s1.rotationZ) * t : undefined,
            bladeSpacing: s1.bladeSpacing !== undefined && s2.bladeSpacing !== undefined ?
                          s1.bladeSpacing + (s2.bladeSpacing - s1.bladeSpacing) * t : undefined
        };
    };

    return {
        points,
        rotation: a.rotation + (b.rotation - a.rotation) * t,
        jacks: lerpTool(a.jacks, b.jacks, DEFAULT_JACKS),
        sofietta: lerpTool(a.sofietta, b.sofietta, DEFAULT_SOFIETTA)
    };
}

export function copyGlassState(src: GlassState, dest: GlassState) {
    while (dest.points.length < src.points.length) {
        dest.points.push({ y: 0, radius: 0, thickness: 0, heat: 0 });
    }
    while (dest.points.length > src.points.length) {
        dest.points.pop();
    }

    for (let i = 0; i < src.points.length; i++) {
        dest.points[i].y = src.points[i].y;
        dest.points[i].radius = src.points[i].radius;
        dest.points[i].thickness = src.points[i].thickness;
        dest.points[i].heat = src.points[i].heat;

        if (i === 0) {
            dest.points[i].baseThickness = src.points[i].baseThickness;
        }
    }
    dest.rotation = src.rotation;

    const copyTool = (s?: ToolState, d?: ToolState, def?: ToolState): ToolState => {
        const srcTool = s || def;
        const destTool = d || { visible: false, position: { x: 0, y: 0, z: 0 } };
        destTool.visible = srcTool!.visible;
        if (!destTool.position) destTool.position = { x: 0, y: 0, z: 0 };
        destTool.position.x = srcTool!.position.x;
        destTool.position.y = srcTool!.position.y;
        destTool.position.z = srcTool!.position.z;
        if (srcTool!.rotationZ !== undefined) {
            destTool.rotationZ = srcTool!.rotationZ;
        } else {
            delete destTool.rotationZ;
        }
        if (srcTool!.bladeSpacing !== undefined) {
            destTool.bladeSpacing = srcTool!.bladeSpacing;
        } else {
            delete destTool.bladeSpacing;
        }
        return destTool;
    };

    dest.jacks = copyTool(src.jacks, dest.jacks, DEFAULT_JACKS);
    dest.sofietta = copyTool(src.sofietta, dest.sofietta, DEFAULT_SOFIETTA);
}
