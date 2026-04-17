import React, { useState, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Activity, 
  AlertCircle, 
  Camera, 
  Construction, 
  Cpu, 
  Navigation, 
  Signal, 
  Truck, 
  Upload, 
  Wind 
} from 'lucide-react';
import { 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer, 
  AreaChart, 
  Area,
  XAxis,
  YAxis
} from 'recharts';
import { cn } from '@/src/lib/utils';
import { 
  VehicleDetection, 
  EmissionData, 
  AgentMessage, 
  EMISSION_FACTORS, 
  HEAVY_DIESEL_CLASSES 
} from '@/src/types';
import { analyzeFootage, getCommanderResponse } from '@/src/lib/gemini';
import { detectWithYOLO, startVideoDetection } from '@/src/lib/yolo';

const HEAVY_VEHICLE_LIMIT = 3;

export default function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeSignal, setActiveSignal] = useState<'RED' | 'GREEN'>('RED');
  const [emissionsHistory, setEmissionsHistory] = useState<EmissionData[]>([]);
  const [agentLog, setAgentLog] = useState<AgentMessage[]>([]);
  const [detections, setDetections] = useState<VehicleDetection[]>([]);
  const [lastAIResponseTime, setLastAIResponseTime] = useState(0);
  const [useLocalModel, setUseLocalModel] = useState(true);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileType, setFileType] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const lastInferenceTime = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Real-time Inference Loop for Video
  React.useEffect(() => {
    if (useLocalModel && previewUrl && fileType?.startsWith('video/') && videoRef.current && overlayCanvasRef.current) {
      const stop = startVideoDetection(videoRef.current, overlayCanvasRef.current, (results) => {
        setDetections(results);
        
        // Update history periodically
        const now = Date.now();
        if (now - lastInferenceTime.current > 1000) {
          const currentCo2 = results.reduce((acc, d) => acc + (EMISSION_FACTORS[d.class] || 0), 0);
          const timeLabel = videoRef.current?.currentTime.toFixed(1) + 's';
          setEmissionsHistory(prev => {
            const next = [...prev, { time: timeLabel, co2: currentCo2 }];
            return next.slice(-20); // Keep last 20 points
          });
          lastInferenceTime.current = now;

          // Trigger AI Commander
          triggerCommanderAI(results);
        }

        const heavyCount = results.filter(d => HEAVY_DIESEL_CLASSES.includes(d.class)).length;
        setActiveSignal(heavyCount > HEAVY_VEHICLE_LIMIT ? 'GREEN' : 'RED');
      });
      return () => stop();
    }
  }, [previewUrl, fileType, useLocalModel]);

  const triggerCommanderAI = async (results: VehicleDetection[]) => {
    if (Date.now() - lastAIResponseTime < 15000) return;
    
    const heavyCount = results.filter(d => HEAVY_DIESEL_CLASSES.includes(d.class)).length;
    const intervention = heavyCount > HEAVY_VEHICLE_LIMIT;
    if (heavyCount > 0) {
      const commanderMsg = await getCommanderResponse(heavyCount, intervention);
      setAgentLog(prev => [{ 
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), 
        text: commanderMsg 
      }, ...prev]);
      setLastAIResponseTime(Date.now());
    }
  };

  const stats = useMemo(() => {
    const liveCo2 = detections.reduce((acc, d) => acc + (EMISSION_FACTORS[d.class] || 0), 0);
    const cumulativeCo2 = emissionsHistory.reduce((acc, d) => acc + d.co2, 0);
    const heavyUnits = detections.filter(d => HEAVY_DIESEL_CLASSES.includes(d.class)).length;
    return { liveCo2, cumulativeCo2, heavyUnits };
  }, [detections, emissionsHistory]);

  const runInferenceOnFrame = async (base64: string, timeLabel: string) => {
    try {
      const results = await detectWithYOLO(base64, 'image/jpeg');
      setDetections(results);

      const currentCo2 = results.reduce((acc, d) => acc + (EMISSION_FACTORS[d.class] || 0), 0);
      setEmissionsHistory(prev => {
        const next = [...prev, { time: timeLabel, co2: currentCo2 }];
        return next.slice(-20);
      });

      const heavyCount = results.filter(d => HEAVY_DIESEL_CLASSES.includes(d.class)).length;
      const intervention = heavyCount > HEAVY_VEHICLE_LIMIT;
      setActiveSignal(intervention ? 'GREEN' : 'RED');

      triggerCommanderAI(results);

      return { heavyCount, intervention };
    } catch (e) {
      return null;
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsAnalyzing(true);
    setFileType(file.type);
    lastInferenceTime.current = 0;

    const reader = new FileReader();
    reader.onload = async () => {
      const base64 = (reader.result as string).split(',')[1];
      setPreviewUrl(reader.result as string);

      try {
        let results: VehicleDetection[] = [];

        if (useLocalModel) {
          try {
            results = await detectWithYOLO(base64, file.type);
          } catch (e) {
            console.warn("Local model failed, falling back to Gemini...");
            results = await analyzeFootage(base64, file.type);
          }
        } else {
          results = await analyzeFootage(base64, file.type);
        }

        setDetections(results);

        const currentCo2 = results.reduce((acc, d) => acc + (EMISSION_FACTORS[d.class] || 0), 0);
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        setEmissionsHistory(prev => [...prev, { time: timeStr, co2: currentCo2 }]);

        const heavyCount = results.filter(d => HEAVY_DIESEL_CLASSES.includes(d.class)).length;
        const intervention = heavyCount > HEAVY_VEHICLE_LIMIT;
        setActiveSignal(intervention ? 'GREEN' : 'RED');

        try {
          const response = await fetch('/api/dispatch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: `Status Update: ${heavyCount} heavy units detected. Grid Intervention ${intervention ? 'required' : 'not required'}.`,
              sessionId: 'user-session'
            })
          });

          if (!response.ok) throw new Error("Backend dispatch failed");
          const data = await response.json();
          if (data.error) throw new Error(data.error);
          setAgentLog(prev => [{ time: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), text: data.message }, ...prev]);
        } catch (err) {
          console.warn("Falling back to local Gemini Commander...");
          const commanderMsg = await getCommanderResponse(heavyCount, intervention);
          setAgentLog(prev => [{ time: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), text: commanderMsg }, ...prev]);
        }

      } catch (error) {
        console.error("Analysis failed:", error);
        alert(String(error));
      } finally {
        setIsAnalyzing(false);
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="min-h-screen font-sans selection:bg-cf-green/10">
      <div className="max-w-[1400px] mx-auto p-10 md:p-14 space-y-12 bg-cf-bg min-h-screen">

        {/* Header */}
        <header className="flex flex-col md:flex-row items-baseline justify-between border-b border-cf-border pb-8 gap-4">
          <div>
            <h1 className="text-2xl font-medium tracking-tight text-white">
              SynthaCarbon <span className="text-cf-green">AI</span>
            </h1>
            <div className="text-[11px] text-gray-500 uppercase tracking-[0.1em] mt-2 flex items-center gap-2">
              <Signal className="w-3 h-3 text-cf-green" />
              Infrastructure Monitoring & Emission Audit
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <StatusBadge label="PHASE 02" color="blue" />
            <StatusBadge label="AUDIT ACTIVE" color="green" />
          </div>
        </header>

        {/* Upload Section */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="text-[11px] font-semibold text-[#737373] uppercase tracking-[0.1em]">Source Telemetry</div>
            <div className="flex items-center gap-3">
              <span className="text-[11px] text-[#737373] font-medium uppercase tracking-wider">Detection Mode:</span>
              <button
                onClick={() => setUseLocalModel(!useLocalModel)}
                className={cn(
                  "px-3 py-1 text-[10px] font-semibold rounded-full border transition-all cursor-pointer",
                  useLocalModel
                    ? "bg-[#1A1A1A] text-white border-[#1A1A1A]"
                    : "bg-white text-[#737373] border-cf-border hover:border-[#1A1A1A]"
                )}
              >
                {useLocalModel ? "Local Model (best.onnx)" : "Gemini Vision AI"}
              </button>
            </div>
          </div>
          <div
            onClick={() => fileInputRef.current?.click()}
            className="group relative flex flex-col items-center justify-center p-12 border border-cf-border rounded-lg bg-cf-surface hover:border-cf-green/50 transition-all cursor-pointer overflow-hidden shadow-2xl"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-cf-green/5 to-transparent pointer-events-none" />
            {isAnalyzing && (
              <motion.div
                className="absolute inset-x-0 bottom-0 h-1 bg-cf-green shadow-[0_0_15px_#00E676] origin-left"
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              />
            )}
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*,video/*"
              onChange={handleFileUpload}
            />
            <div className="flex flex-col items-center gap-4 text-center relative z-10">
              <div className="p-4 rounded-full bg-cf-bg border border-cf-border group-hover:bg-cf-green/10 group-hover:border-cf-green/30 transition-all">
                {isAnalyzing ? <Cpu className="w-8 h-8 text-cf-green animate-spin" /> : <Upload className="w-8 h-8 text-gray-600" />}
              </div>
              <div className="space-y-1">
                <p className="text-[15px] font-medium text-white">
                  {isAnalyzing ? "Initiating Grid Diagnostic..." : "Ingest Telemetry Data (Video/Image)"}
                </p>
                <p className="text-[13px] text-gray-500">
                  Secure transmission optimized for .mp4, .webp, .png
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Metrics Bar */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            label="Live Intensity"
            value={`${stats.liveCo2.toLocaleString()} g/hr`}
            icon={<Wind className="w-4 h-4 text-[#737373]" />}
          />
          <MetricCard
            label="Cumulative CO₂"
            value={`${stats.cumulativeCo2.toLocaleString()} g`}
            icon={<Activity className="w-4 h-4 text-[#737373]" />}
          />
          <MetricCard
            label="Heavy Units"
            value={`${stats.heavyUnits} / ${HEAVY_VEHICLE_LIMIT} limit`}
            icon={<Construction className="w-4 h-4 text-[#737373]" />}
          />
          <MetricCard
            label="System Status"
            value={activeSignal === 'GREEN' ? "INTERVENTION" : "NOMINAL"}
            icon={<AlertCircle className="w-4 h-4 text-[#737373]" />}
            statusDot={activeSignal === 'GREEN' ? 'amber' : 'green'}
          />
        </section>

        {/* Main Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-12">

          {/* Left Column */}
          <div className="lg:col-span-3 space-y-12">

            {/* Perception Feed */}
            <div className="space-y-4">
              <div className="text-[11px] font-semibold text-[#737373] uppercase tracking-[0.1em] flex items-center justify-between">
                <span>Perception Preview</span>
                {detections.length > 0 && <span className="text-cf-green">Certified Analysis</span>}
              </div>
              <div className="relative aspect-video rounded-lg bg-[#F0F0F0] border border-cf-border overflow-hidden flex items-center justify-center">
                {previewUrl ? (
                  <>
                    {fileType?.startsWith('video/') ? (
                      <div className="relative w-full h-full">
                        <video
                          ref={videoRef}
                          src={previewUrl}
                          className="w-full h-full object-cover"
                          autoPlay
                          muted
                          loop
                          controls
                        />
                        <canvas 
                          ref={overlayCanvasRef}
                          className="absolute inset-0 w-full h-full pointer-events-none"
                        />
                      </div>
                    ) : (
                      <img src={previewUrl} className="w-full h-full object-cover" alt="Site preview" referrerPolicy="no-referrer" />
                    )}
                    <div className="absolute inset-0 pointer-events-none">
                      <div className="absolute inset-0 bg-cf-green/5 opacity-20" />
                      {!fileType?.startsWith('video/') && detections.map((d, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="absolute border-2 border-cf-green bg-cf-green/5 shadow-[0_0_15px_rgba(0,230,118,0.2)]"
                          style={{
                            left: `${d.bbox[0]}%`,
                            top: `${d.bbox[1]}%`,
                            width: `${d.bbox[2]}%`,
                            height: `${d.bbox[3]}%`,
                          }}
                        >
                          <div className="absolute -top-7 left-[-2px] px-2 py-1.5 flex items-center gap-1.5 bg-cf-green text-black text-[11px] font-black uppercase tracking-tighter">
                            <Activity className="w-3.5 h-3.5" />
                            {d.class}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="flex flex-col items-center gap-4 text-[#737373] italic">
                    <Camera className="w-12 h-12 opacity-30" />
                    <p className="text-sm">Awaiting footage preview</p>
                  </div>
                )}
                {isAnalyzing && (
                  <div className="absolute inset-0 bg-white/60 backdrop-blur-[2px] flex items-center justify-center">
                    <div className="flex flex-col items-center gap-3">
                      <Cpu className="w-10 h-10 text-[#1A1A1A] animate-spin" />
                      <p className="text-xs font-medium uppercase tracking-widest text-[#1A1A1A]">Scanning...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Emission Analytics */}
            <div className="space-y-4">
              <div className="text-[11px] font-semibold text-[#737373] uppercase tracking-[0.1em]">Emission Analytics</div>
              <div className="h-[280px] w-full bg-cf-surface border border-cf-border rounded-lg p-8 shadow-sm">
                {emissionsHistory.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={emissionsHistory}>
                      <defs>
                        <linearGradient id="colorCo2" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00E676" stopOpacity={0.2} />
                          <stop offset="95%" stopColor="#00E676" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1F1F1F" />
                      <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fill: '#4B5563', fontSize: 10 }} />
                      <YAxis axisLine={false} tickLine={false} tick={{ fill: '#4B5563', fontSize: 10 }} unit="g" />
                      <RechartsTooltip
                        contentStyle={{ backgroundColor: '#121212', border: '1px solid #1F1F1F', borderRadius: '4px' }}
                        itemStyle={{ color: '#00E676', fontSize: '13px', fontWeight: 600 }}
                        labelStyle={{ fontSize: '11px', color: '#6B7280', marginBottom: '4px' }}
                      />
                      <Area type="monotone" dataKey="co2" stroke="#00E676" strokeWidth={2} fillOpacity={1} fill="url(#colorCo2)" animationDuration={500} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-[13px] text-[#737373] italic">
                    Telemetric data awaiting initialization
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="lg:col-span-2 space-y-12">

            {/* AI Command Center */}
            <div className="space-y-4">
              <div className="text-[11px] font-semibold text-[#737373] uppercase tracking-[0.1em] flex items-center justify-between">
                <span>AI DISPATCH CENTER</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full border border-cf-border bg-cf-bg">ACTIVE STREAM</span>
              </div>
              <div className="bg-[#0A0A0A] border border-[#1A1A1A] rounded-lg min-h-[200px] flex flex-col shadow-sm text-gray-400">
                <AnimatePresence mode="wait">
                  {agentLog.length > 0 ? (
                    <motion.div
                      key={agentLog[0].time}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex-1 flex flex-col"
                    >
                      <div className="px-8 py-6 border-b border-white/5">
                        <div className="flex items-center gap-4 mb-2">
                          <div className="w-8 h-8 rounded-full bg-[#1A1A1A] border border-white/10 flex items-center justify-center text-[10px] font-bold text-cf-green">AI</div>
                          <div>
                            <h3 className="text-sm font-semibold text-white">Grid Intervention Commander</h3>
                            <p className="text-[10px] text-gray-500">Live Situation Report</p>
                          </div>
                        </div>
                      </div>
                      <div className="px-8 py-8 flex-1">
                        <p className="text-[13px] leading-relaxed font-light">{agentLog[0].text}</p>
                      </div>
                      <div className="px-8 py-6 border-t border-white/5 flex gap-3">
                        <button className="px-4 py-2 bg-cf-green text-black text-[12px] font-bold rounded hover:bg-cf-green/90 transition-colors uppercase tracking-wider">Authorize Protocol</button>
                        <button className="px-4 py-2 border border-white/10 text-white text-[12px] font-medium rounded hover:bg-white/5 transition-colors">DEFER</button>
                      </div>
                    </motion.div>
                  ) : (
                    <div className="flex-1 flex flex-col items-center justify-center p-12 text-center">
                      <div className="flex items-center gap-4 mb-6">
                        <div className="w-10 h-10 rounded-full bg-[#1A1A1A] border border-white/10 flex items-center justify-center text-xs font-bold text-gray-600">AI</div>
                        <div className="text-left">
                          <h3 className="text-sm font-medium text-white/80">Grid Intervention Commander</h3>
                          <p className="text-[10px] text-gray-500 italic">Awaiting scan result</p>
                        </div>
                      </div>
                      <p className="text-[13px] font-light text-gray-600 max-w-[500px] leading-relaxed">
                        No dispatch issued yet. Upload site footage to trigger the AI commander.
                      </p>
                    </div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Emission Baseline */}
            <div className="space-y-4">
              <div className="text-[11px] font-semibold text-gray-500 uppercase tracking-[0.1em]">Emission Baseline</div>
              <div className="bg-cf-surface border border-cf-border rounded-lg p-6 space-y-6 shadow-xl">
                {Object.entries(EMISSION_FACTORS).map(([vehicle, ef]) => (
                  <div key={vehicle} className="group">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <span className="text-[14px] font-medium capitalize text-white/80">{vehicle}</span>
                        <span className="text-[10px] px-2 py-0.5 rounded-sm border border-white/5 text-gray-400 font-mono tracking-tighter bg-white/2">{ef}g</span>
                      </div>
                    </div>
                    <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${(ef / 15000) * 100}%` }}
                        className="h-full bg-cf-green shadow-[0_0_8px_#00E676]"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Eco-Path Corridor Map */}
        <section className="space-y-4">
          <div className="text-[11px] font-semibold text-[#737373] uppercase tracking-[0.1em] flex items-center justify-between">
            <span>ECO-PATH CORRIDOR MAP</span>
            <div className="flex gap-6">
              <LegendItem color="cf-green" label="Optimized" />
              <LegendItem color="cf-blue" label="Secondary" />
            </div>
          </div>
          <div className="relative bg-[#0A0A0A] border border-[#1A1A1A] rounded-lg h-[450px] overflow-hidden shadow-2xl">
            <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(#333 1px, transparent 1px)', backgroundSize: '20px 20px' }} />
            <div className="relative h-full flex flex-col justify-center px-12">
              <div className="absolute top-8 left-12 right-12 flex justify-between items-center z-10">
                <div className="text-[10px] font-mono text-gray-500 uppercase tracking-widest">
                  R&R SKUDAI CORRIDOR — LIVE VIEW
                </div>
                <div className={cn(
                  "px-3 py-1 rounded text-[9px] font-bold border uppercase tracking-wider",
                  activeSignal === 'GREEN'
                    ? "bg-cf-green/10 text-cf-green border-cf-green/30"
                    : "bg-cf-red/10 text-cf-red border-cf-red/30"
                )}>
                  {activeSignal === 'GREEN' ? "GRID INTERVENTION — HEAVY FLOW" : "STANDARD MONITORING — NORMAL FLOW"}
                </div>
              </div>

              <div className="relative h-48 w-full border-t border-b border-white/5 bg-[#121212] overflow-hidden">
                <div className={cn(
                  "absolute top-0 h-1/2 w-full transition-colors duration-1000",
                  activeSignal === 'GREEN' ? "bg-cf-red/5" : "bg-transparent"
                )} />
                <div className="absolute top-1/2 left-0 right-0 h-0.5 border-t border-dashed border-white/10" />
                <div className="absolute top-2 left-6 text-[9px] font-bold text-cf-red/40 tracking-widest uppercase">
                  HEAVY DIESEL CORRIDOR
                </div>
                <div className="absolute inset-0 flex items-center">
                  <div className="flex-1 relative h-full">
                    <AnimatePresence>
                      {detections.length > 0 ? detections.map((v, i) => (
                        <motion.div
                          key={`${v.class}-${i}`}
                          initial={{ x: -200, opacity: 0 }}
                          animate={{ x: 100 + (i * 150), opacity: 1 }}
                          exit={{ x: 1400, opacity: 0 }}
                          transition={{ duration: 1.5, delay: i * 0.2, ease: "easeOut" }}
                          className={cn(
                            "absolute w-24 h-16 rounded border bg-[#1A1A1A]/80 flex flex-col items-center justify-center gap-1",
                            HEAVY_DIESEL_CLASSES.includes(v.class) ? "border-cf-red/30 shadow-[0_0_15px_-5px_#ff3d57]" : "border-white/10"
                          )}
                          style={{ top: HEAVY_DIESEL_CLASSES.includes(v.class) ? '10%' : '55%' }}
                        >
                          <div className="text-[8px] font-black text-white/30 uppercase tracking-tighter">{v.class}</div>
                          <div className="flex gap-0.5">
                            <div className="w-1.5 h-1.5 rounded-full bg-white/5" />
                            <div className="w-1.5 h-1.5 rounded-full bg-white/5" />
                          </div>
                        </motion.div>
                      )) : (
                        <div className="h-full flex items-center justify-center opacity-10">
                          <div className="flex gap-24">
                            {[1, 2, 3].map(i => <div key={i} className="w-24 h-16 border border-white/20 rounded-lg flex items-center justify-center italic text-[10px]">No unit</div>)}
                          </div>
                        </div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
                <div className="absolute bottom-4 right-12 flex gap-4 text-white/5">
                  <Navigation className="w-3 h-3 rotate-90" />
                  <Navigation className="w-3 h-3 rotate-90" />
                  <Navigation className="w-3 h-3 rotate-90" />
                </div>
                <div className="absolute bottom-8 right-24 text-[10px] font-mono text-gray-700 tracking-tighter flex items-center gap-2">
                  <div className="h-px w-12 bg-gray-800" />
                  2.4 km
                  <div className="h-px w-12 bg-gray-800" />
                </div>
              </div>

              <div className="absolute left-8 top-1/2 -translate-y-1/2 z-20">
                <TacticalTrafficLight active={activeSignal === 'GREEN'} />
              </div>
              <div className="absolute right-8 top-1/2 -translate-y-1/2 z-20">
                <TacticalTrafficLight active={activeSignal === 'GREEN'} />
              </div>
              <div className="absolute bottom-12 left-1/2 -translate-x-1/2 flex gap-12 text-white/5">
                <Navigation className="w-5 h-5 -rotate-45" />
                <Navigation className="w-5 h-5 -rotate-45" />
              </div>
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}

function MetricCard({ label, value, icon, statusDot }: { label: string, value: string, icon: React.ReactNode, statusDot?: 'green' | 'amber' }) {
  const dots = { green: 'bg-cf-green', amber: 'bg-cf-amber' };
  return (
    <div className="bg-cf-surface border border-cf-border rounded-lg p-6 flex flex-col justify-between aspect-square shadow-xl hover:border-cf-green/50 transition-all group overflow-hidden relative">
      <div className="absolute -right-4 -top-4 w-24 h-24 bg-white/5 rounded-full blur-3xl group-hover:bg-cf-green/10 transition-all" />
      <div className="flex items-center justify-between relative z-10">
        <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-[0.1em]">{label}</span>
        {icon}
      </div>
      <div className="space-y-2 relative z-10">
        {statusDot && (
          <div className="flex items-center gap-2 mb-1">
            <div className={cn("w-2 h-2 rounded-full animate-pulse", dots[statusDot])} />
            <span className="text-[12px] font-medium text-gray-400 capitalize">{statusDot === 'green' ? 'Nominal' : 'Warning'}</span>
          </div>
        )}
        <div className="text-2xl font-light tracking-tight text-white">{value}</div>
      </div>
    </div>
  );
}

function StatusBadge({ label }: { label: string, color: 'green' | 'blue' }) {
  return (
    <span className="px-3 py-1 rounded-sm text-[10px] font-bold border border-white/10 bg-white/5 text-gray-300 uppercase tracking-widest">
      {label}
    </span>
  );
}

function LegendItem({ color, label }: { color: string, label: string }) {
  const colorMap = { 'cf-green': 'bg-cf-green', 'cf-blue': 'bg-cf-blue' };
  return (
    <div className="flex items-center gap-2">
      <div className={cn("w-2 h-2 rounded-full", colorMap[color as keyof typeof colorMap])} />
      <span className="text-[10px] text-[#737373] uppercase tracking-wider font-semibold">{label}</span>
    </div>
  );
}

function TacticalTrafficLight({ active }: { active: boolean }) {
  return (
    <div className="w-12 h-32 bg-[#1A1A1A] rounded-xl border border-white/5 p-2 flex flex-col justify-between shadow-2xl">
      <div className={cn("w-full aspect-square rounded-full shadow-inner transition-all duration-500", !active ? "bg-cf-red shadow-[0_0_20px_#ff3d57]" : "bg-white/5")} />
      <div className="w-full aspect-square rounded-full shadow-inner bg-white/5" />
      <div className={cn("w-full aspect-square rounded-full shadow-inner transition-all duration-500", active ? "bg-cf-green shadow-[0_0_20px_#00e676]" : "bg-white/5")} />
    </div>
  );
}