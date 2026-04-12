import { useState, useEffect, useRef } from 'react'
import cupExpandedMiddle from './assets/cup_expanded_middle.jpg'
import cupFlaredLip from './assets/cup_flared_lip.jpg'
import brokenThinBase from './assets/broken_thin_base.jpg'
import cupWithWaist from './assets/cup_with_waist.jpg'
import TestPlayerView from './components/GlassblowPlayer/TestPlayerView'
import { GlassblowPlayer } from './components/GlassblowPlayer/GlassblowPlayer'

function App() {
    const [currentView, setCurrentView] = useState<'dashboard' | 'upload' | 'sandbox'>('dashboard')

    const renderView = () => {
        switch (currentView) {
            case 'dashboard':
                return <DashboardView onViewChange={setCurrentView} />
            case 'upload':
                return <UploadView />
            case 'sandbox':
                return <TestPlayerView />
            default:
                return <DashboardView onViewChange={setCurrentView} />
        }
    }

    return (
        <div className="app-container">
            <nav className="navbar">
                <h1>Glassblowing Education</h1>
                <div className="nav-links">
                    <span
                        className={`nav-link ${currentView === 'dashboard' ? 'active' : ''}`}
                        onClick={() => setCurrentView('dashboard')}
                    >
                        Dashboard
                    </span>
                    <span
                        className={`nav-link ${currentView === 'upload' ? 'active' : ''}`}
                        onClick={() => setCurrentView('upload')}
                    >
                        Upload & Analyze
                    </span>
                    <span
                        className={`nav-link ${currentView === 'sandbox' ? 'active' : ''}`}
                        onClick={() => setCurrentView('sandbox')}
                        style={{ borderLeft: '1px solid #444', paddingLeft: '12px', marginLeft: '6px', color: '#e67e22', cursor: 'pointer' }}
                    >
                        Sandbox 🧪
                    </span>
                </div>
            </nav>

            <main className="main-content">
                {renderView()}
            </main>
        </div>
    )
}

interface ViewProps {
    onViewChange: (view: 'dashboard' | 'upload') => void
}

function DashboardView({ onViewChange }: ViewProps) {
    return (
        <div className="view-container">
            <h2>Welcome to Glassblowing Education</h2>
            <p>Select a module to get started with your glassblowing learning journey.</p>

            <div className="dashboard-grid">
                <div className="card" onClick={() => onViewChange('upload')} style={{ cursor: 'pointer' }}>
                    <h3>Analyze a Failure</h3>
                    <p>Upload a photo of your piece where something went wrong. We'll help identify what happened and how to fix it.</p>
                </div>
            </div>
        </div>
    )
}

function UploadView() {
    const [images, setImages] = useState<any[]>([
        { id: 'preset-1', url: cupExpandedMiddle, name: 'cup_expanded_middle.jpg', isPreset: true },
        { id: 'preset-2', url: cupFlaredLip, name: 'cup_flared_lip.JPG', isPreset: true },
        { id: 'preset-3', url: brokenThinBase, name: 'broken_thin_base.jpg', isPreset: true },
        { id: 'preset-4', url: cupWithWaist, name: 'cup_with_waist.png', isPreset: true }
    ]);
    const [selectedImageId, setSelectedImageId] = useState<string>('preset-1');
    const [hoveredPoint, setHoveredPoint] = useState<string | null>(null);
    const [activePoint, setActivePoint] = useState<string | null>(null);
    const [videos, setVideos] = useState<any[]>([]);
    const [hotspots, setHotspots] = useState<any[]>([]);
    const [overallAnalysis, setOverallAnalysis] = useState<string>('');
    const [analyzing, setAnalyzing] = useState(false);
    const [selectedSequence, setSelectedSequence] = useState<string | undefined>(undefined);
    const [hasLoadedPlayer, setHasLoadedPlayer] = useState(false);
    const [activeTab, setActiveTab] = useState<'video' | '3d'>('3d');
    const [imageAspect, setImageAspect] = useState<number>(1);

    useEffect(() => {
        if (selectedSequence) {
            setHasLoadedPlayer(true);
        }
    }, [selectedSequence]);

    const currentActivePointId = activePoint || hoveredPoint;
    const activeHotspot = hotspots.find(p => p.id === currentActivePointId);

    const containerRef = useRef<HTMLDivElement>(null);
    const detailsPanelRef = useRef<HTMLDivElement>(null);
    const [lineTargetY, setLineTargetY] = useState<number>(40);
    const [resizeCount, setResizeCount] = useState<number>(0);

    useEffect(() => {
        const handleResize = () => setResizeCount(prev => prev + 1);
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const selectedImage = images.find(img => img.id === selectedImageId);

    useEffect(() => {
        if (selectedImage) {
            setAnalyzing(true);
            setActivePoint(null); // Reset tooltips on image change
            setHasLoadedPlayer(false); // Hide the animation component on image change
            setSelectedSequence(undefined); // Clear sequence on image change
            setImageAspect(1); // Reset aspect ratio to prevent stale layout flashes

            fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: selectedImage.name, isPreset: selectedImage.isPreset })
            })
                .then(res => res.json())
                .then(data => {
                    setHotspots(data.hotspots || []);
                    setOverallAnalysis(data.overallAnalysis || '');
                    setAnalyzing(false);
                })
                .catch(err => {
                    console.error(err);
                    setAnalyzing(false);
                });
        } else {
            setHotspots([]);
        }
    }, [selectedImageId]); // Only trigger when ID changes

    useEffect(() => {
        fetch('/api/videos')
            .then(res => res.json())
            .then(data => setVideos(data))
            .catch(console.error);
    }, []);

    useEffect(() => {
        if (activeHotspot && detailsPanelRef.current && containerRef.current) {
            const detailsRect = detailsPanelRef.current.getBoundingClientRect();
            const containerRect = containerRef.current.getBoundingClientRect();
            setLineTargetY(Math.max(40, detailsRect.top - containerRect.top + 30));
        }
    }, [activeHotspot, overallAnalysis, resizeCount]);

    const handleFileUpload = (file: File) => {
        const url = URL.createObjectURL(file);
        const newImage = {
            id: `user-${Date.now()}`,
            url: url,
            name: file.name,
            isPreset: false
        };
        setImages(prev => [...prev, newImage]);
        setSelectedImageId(newImage.id); // Auto-select new upload
    };

    const handleDeleteImage = (id: string, e: React.MouseEvent) => {
        e.stopPropagation(); // Don't trigger selection
        setImages(prev => prev.filter(img => img.id !== id));
        if (selectedImageId === id) {
            setSelectedImageId('preset-1'); // Fallback
        }
        // Revoke object URL if it was a user upload
        const img = images.find(i => i.id === id);
        if (img && !img.isPreset) {
            URL.revokeObjectURL(img.url);
        }
    };

    const getClipUrl = (clipKey: string) => {
        const video = videos.find(v => v.name.includes(clipKey));
        return video ? video.url : null;
    };



    return (
        <div className="view-container">
            {selectedImage && (
                <>
                    <div style={{ display: 'flex', gap: '2rem', alignItems: 'stretch', height: '900px' }}>
                        {/* Left Column: Photo Upload / Image with Pins */}
                        <div style={{ flex: '0 0 500px', maxWidth: '500px', width: '100%', position: 'relative', display: 'flex', flexDirection: 'column' }}>
                            <h2 style={{ marginTop: 0 }}>Interactive Failure Analysis</h2>
                            <p style={{ color: '#555', marginBottom: '1.5rem', fontSize: '0.95rem' }}>Select or upload a photo of your glass piece for analysis.</p>
                            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', overflowX: 'auto', paddingBottom: '0.8rem' }}>
                                {images.map(img => (
                                    <div key={img.id} onClick={() => setSelectedImageId(img.id)} style={{ position: 'relative', width: '80px', height: '80px', flexShrink: 0, borderRadius: '8px', border: selectedImageId === img.id ? '3px solid #3498db' : '1px solid #ddd', cursor: 'pointer', overflow: 'hidden' }}>
                                        <img src={img.url} alt={img.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                        {!img.isPreset && (
                                            <button onClick={(e) => handleDeleteImage(img.id, e)} style={{ position: 'absolute', top: '4px', right: '4px', background: 'rgba(231, 76, 60, 0.8)', color: 'white', border: 'none', borderRadius: '50%', width: '18px', height: '18px', fontSize: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', padding: 0 }}>✕</button>
                                        )}
                                    </div>
                                ))}
                                <div onClick={() => document.getElementById('file-input')?.click()} style={{ width: '80px', height: '80px', flexShrink: 0, borderRadius: '8px', border: '2px dashed #bbb', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', background: '#f9f9f9', color: '#666', fontSize: '0.8rem' }}>
                                    <span>+ Add</span>
                                    <input id="file-input" type="file" hidden accept="image/*" onChange={(e) => { if (e.target.files && e.target.files[0]) handleFileUpload(e.target.files[0]); }} />
                                </div>
                            </div>

                            {analyzing && <p style={{ textAlign: 'center', color: '#3498db' }}>Analyzing image diagnostics...</p>}

                            <div style={{ flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                                <div ref={containerRef} className="analysis-viewer" style={{ position: 'relative', aspectRatio: imageAspect, width: 'auto', height: 'auto', maxWidth: '100%', maxHeight: '100%', opacity: analyzing ? 0.5 : 1 }}>
                                    <img
                                        src={selectedImage.url}
                                        alt="Overlay Diagnostic Container"
                                        onLoad={(e) => setImageAspect(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
                                        style={{ display: 'block', width: '100%', height: '100%', maxWidth: '100%', maxHeight: '100%', borderRadius: '8px' }}
                                    />

                                    {/* SVG Connector Line */}
                                    <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 1, overflow: 'visible' }}>
                                        {activeHotspot && containerRef.current && (
                                            <line
                                                x1={`${(parseFloat(activeHotspot.left) / 100) * containerRef.current.offsetWidth}px`}
                                                y1={`${(parseFloat(activeHotspot.top) / 100) * containerRef.current.offsetHeight}px`}
                                                x2={`${containerRef.current.offsetWidth + 32}px`}
                                                y2={`${lineTargetY}px`}
                                                stroke="#e74c3c"
                                                strokeWidth="2"
                                                strokeDasharray="4 4"
                                            />
                                        )}
                                    </svg>

                                    {!analyzing && hotspots.map(p => (
                                        <div
                                            key={p.id}
                                            className="hotspot-pin"
                                            style={{
                                                position: 'absolute',
                                                top: p.top,
                                                left: p.left,
                                                width: '16px',
                                                height: '16px',
                                                borderRadius: '50%',
                                                background: activePoint === p.id || hoveredPoint === p.id ? '#e74c3c' : 'rgba(231, 76, 60, 0.6)',
                                                border: '2px solid white',
                                                cursor: 'pointer',
                                                transform: 'translate(-50%, -50%)',
                                                boxShadow: '0 0 8px rgba(231,76,60,0.6)',
                                                zIndex: 2
                                            }}
                                            onMouseEnter={() => {
                                                setHoveredPoint(p.id);
                                                if (p.sequenceKey) {
                                                    setSelectedSequence(p.sequenceKey);
                                                    setActiveTab('3d');
                                                }
                                            }}
                                            onMouseLeave={() => setHoveredPoint(null)}
                                            onClick={() => {
                                                const isExpanding = activePoint !== p.id;
                                                setActivePoint(isExpanding ? p.id : null);
                                                if (isExpanding && p.sequenceKey) {
                                                    setSelectedSequence(p.sequenceKey);
                                                } else {
                                                    setSelectedSequence(undefined);
                                                }
                                            }}
                                        />
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Right Column: Overall + Details Cards */}
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem', minHeight: 0 }}>
                            {overallAnalysis && (
                                <div style={{ background: '#ffffff', padding: '1.5rem', borderRadius: '8px', border: '1px solid #eee', boxShadow: '0 2px 8px rgba(0,0,0,0.05)' }}>
                                    <p style={{ margin: 0, color: '#444', fontSize: '1rem', lineHeight: '1.5' }}><strong>Overall Diagnostic analysis:</strong> {overallAnalysis}</p>
                                </div>
                            )}

                            <div ref={detailsPanelRef} style={{ flex: 1, background: '#f9f9f9', padding: '1.5rem', borderRadius: '8px', border: '1px solid #eee', position: 'relative', overflowY: 'auto' }}>
                                {activeHotspot ? (
                                    <div>
                                        <h4 style={{ margin: '0 0 0.5rem 0', color: '#e74c3c', fontSize: '1.2rem' }}>{activeHotspot.label}</h4>
                                        <p style={{ margin: '0 0 1.5rem 0', color: '#555' }}>{activeHotspot.desc}</p>

                                        {/* Tabs Toolbar */}
                                        <div style={{ display: 'flex', gap: '4px', background: '#e0e0e0', padding: '3px', borderRadius: '6px', marginBottom: '1rem', width: 'fit-content' }}>
                                            <button
                                                onClick={() => { setActiveTab('3d'); setTimeout(() => window.dispatchEvent(new Event('resize')), 50); }}
                                                style={{ padding: '6px 12px', border: 'none', background: activeTab === '3d' ? '#ffffff' : 'transparent', color: activeTab === '3d' ? '#2c3e50' : '#666', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '0.85rem', transition: 'all 0.2s' }}
                                            >
                                                3D Simulation
                                            </button>
                                            <button
                                                onClick={() => setActiveTab('video')}
                                                style={{ padding: '6px 12px', border: 'none', background: activeTab === 'video' ? '#ffffff' : 'transparent', color: activeTab === 'video' ? '#2c3e50' : '#666', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '0.85rem', transition: 'all 0.2s' }}
                                            >
                                                Demo Video
                                            </button>
                                        </div>

                                        {/* Tab Contents */}
                                        <div style={{ display: activeTab === 'video' ? 'block' : 'none' }}>
                                            {getClipUrl(activeHotspot.clipKey) ? (
                                                <div>
                                                    <video key={activeHotspot.id} controls autoPlay muted crossOrigin="anonymous" width="100%" style={{ marginTop: '1.5rem', height: '600px', objectFit: 'contain', borderRadius: '4px', backgroundColor: '#000', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
                                                        <source src={getClipUrl(activeHotspot.clipKey)!} type="video/mp4" />
                                                    </video>
                                                </div>
                                            ) : (
                                                <p style={{ fontSize: '0.9rem', color: '#888' }}>Loading demonstration clip...</p>
                                            )}
                                        </div>

                                        <div style={{ display: activeTab === '3d' ? 'block' : 'none' }}>
                                            {/* 3D Player Sub-toggles (Mistake vs Correct) */}
                                            {activeHotspot?.correctSequenceKey && (
                                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                                    <p style={{ margin: 0, fontSize: '0.9rem', fontWeight: 'bold', color: '#333' }}></p>
                                                    <div style={{ display: 'flex', background: '#f0f0f0', borderRadius: '4px', padding: '3px', border: '1px solid #ddd' }}>
                                                        <button
                                                            onClick={() => setSelectedSequence(activeHotspot.sequenceKey)}
                                                            style={{ padding: '4px 10px', border: 'none', background: selectedSequence === activeHotspot.sequenceKey ? '#e74c3c' : 'transparent', color: selectedSequence === activeHotspot.sequenceKey ? 'white' : '#666', borderRadius: '3px', cursor: 'pointer', fontWeight: 'bold', fontSize: '0.8rem' }}
                                                        >
                                                            Mistake
                                                        </button>
                                                        <button
                                                            onClick={() => setSelectedSequence(activeHotspot.correctSequenceKey)}
                                                            style={{ padding: '4px 10px', border: 'none', background: selectedSequence === activeHotspot.correctSequenceKey ? '#2ecc71' : 'transparent', color: selectedSequence === activeHotspot.correctSequenceKey ? 'white' : '#666', borderRadius: '3px', cursor: 'pointer', fontWeight: 'bold', fontSize: '0.8rem' }}
                                                        >
                                                            Correct
                                                        </button>
                                                    </div>
                                                </div>
                                            )}
                                            {hasLoadedPlayer && (
                                                <GlassblowPlayer showTweakpane={false} loadSequenceName={selectedSequence} />
                                            )}
                                        </div>
                                    </div>
                                ) : (
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#999', fontSize: '0.95rem', fontStyle: 'italic', padding: '1rem', textAlign: 'center' }}>
                                        Select or hover a diagnostic pin for deep analysis view
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Bottom Row: Overall Analysis */}

                </>
            )}
        </div>
    )
}

export default App
