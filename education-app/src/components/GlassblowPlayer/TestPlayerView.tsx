import { GlassblowPlayer } from './GlassblowPlayer';

export default function TestPlayerView() {
    return (
        <div style={{ padding: '2rem' }}>
            <h2>Animation Sandbox</h2>
            <p>Develop and test the glassblowing animation component independently.</p>
            <div style={{ width: '100%', margin: '0 auto', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', borderRadius: '8px', overflow: 'hidden' }}>
                <GlassblowPlayer showTweakpane={true} />
            </div>
        </div>
    );
}
