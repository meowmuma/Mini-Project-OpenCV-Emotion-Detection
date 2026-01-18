"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // States สำหรับ UI
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [status, setStatus] = useState<string>("กำลังโหลดระบบ...");
  const [emotion, setEmotion] = useState<string>("Neutral");
  const [confidence, setConfidence] = useState<number>(0);

  // Refs สำหรับ AI
  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  // --- 1. ตรรกะการโหลดระบบ AI ---
  async function loadOpenCV() {
    if (typeof window === "undefined") return;
    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;
      script.onload = () => {
        const cv = (window as any).cv;
        const waitReady = () => {
          if (cv.Mat) { cvRef.current = cv; resolve(); }
          else { setTimeout(waitReady, 50); }
        };
        if (cv.onRuntimeInitialized) cv.onRuntimeInitialized = waitReady;
        else waitReady();
      };
      script.onerror = () => reject(new Error("โหลด OpenCV ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  async function loadCascade() {
    const cv = cvRef.current;
    const res = await fetch("/opencv/haarcascade_frontalface_default.xml");
    const data = new Uint8Array(await res.arrayBuffer());
    cv.FS_createDataFile("/", "face.xml", data, true, false, false);
    const faceCascade = new cv.CascadeClassifier();
    faceCascade.load("face.xml");
    faceCascadeRef.current = faceCascade;
  }

  async function loadModel() {
    try {
      // ใช้ path /moduls/ ตามที่คุณตั้งชื่อไว้ในโฟลเดอร์ public
      const session = await ort.InferenceSession.create("/moduls/emotion_yolo11n_cls.onnx", {
        executionProviders: ["wasm"],
      });
      sessionRef.current = session;
      const clsRes = await fetch("/moduls/classes.json");
      classesRef.current = await clsRes.json();
    } catch (e) {
      throw new Error("หาไฟล์โมเดลไม่เจอใน /moduls/");
    }
  }

  // --- 2. การควบคุมกล้อง ---
  const handleStartCamera = async () => {
    if (isCameraActive) {
      // ปิดกล้อง
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
      }
      setIsCameraActive(false);
      setStatus("Camera is sleeping");
    } else {
      // เปิดกล้อง
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setIsCameraActive(true);
          setStatus("Detecting faces...");
          requestAnimationFrame(loop);
        }
      } catch (e) {
        setStatus("เข้าถึงกล้องไม่ได้");
      }
    }
  };

  // --- 3. การประมวลผล AI Loop ---
  function preprocess(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size; tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);
    const data = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(3 * size * size);
    for (let i = 0; i < size * size; i++) {
      float[i] = data[i * 4] / 255;
      float[i + size * size] = data[i * 4 + 1] / 255;
      float[i + 2 * size * size] = data[i * 4 + 2] / 255;
    }
    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  async function loop() {
    const cv = cvRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !cv || video.paused) return;

    const ctx = canvas.getContext("2d")!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    const faces = new cv.RectVector();
    faceCascadeRef.current.detectMultiScale(gray, faces, 1.1, 3, 0);

    if (faces.size() > 0) {
      const r = faces.get(0);
      ctx.strokeStyle = "#FF69B4"; // สีชมพูให้น่ารักตามธีม
      ctx.lineWidth = 4;
      ctx.strokeRect(r.x, r.y, r.width, r.height);

      const faceCanvas = document.createElement("canvas");
      faceCanvas.width = r.width; faceCanvas.height = r.height;
      faceCanvas.getContext("2d")?.drawImage(canvas, r.x, r.y, r.width, r.height, 0, 0, r.width, r.height);

      if (sessionRef.current && classesRef.current) {
        const input = preprocess(faceCanvas);
        const feeds: any = {};
        feeds[sessionRef.current.inputNames[0]] = input;
        const output = await sessionRef.current.run(feeds);
        const logits = output[sessionRef.current.outputNames[0]].data as Float32Array;
        const exps = logits.map(v => Math.exp(v - Math.max(...logits)));
        const sum = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(v => v / sum);
        const maxIdx = probs.indexOf(Math.max(...probs));

        setEmotion(classesRef.current[maxIdx]);
        setConfidence(Math.round(probs[maxIdx] * 100));
      }
    }
    src.delete(); gray.delete(); faces.delete();
    requestAnimationFrame(loop);
  }

  useEffect(() => {
    (async () => {
      try {
        await loadOpenCV();
        await loadCascade();
        await loadModel();
        setStatus("Ready to go!");
      } catch (e: any) {
        setStatus(`Error: ${e.message}`);
      }
    })();
  }, []);

  // --- 4. ฟังก์ชันจัดการ Style ตามอารมณ์ ---
  const getEmotionStyle = (emo: string) => {
    switch (emo.toLowerCase()) {
      case "happy": return { emoji: "😊", color: "bg-green-100 text-green-600 border-green-200", label: "Happy" };
      case "sad": return { emoji: "😢", color: "bg-blue-100 text-blue-600 border-blue-200", label: "Sad" };
      case "angry": return { emoji: "😠", color: "bg-red-100 text-red-600 border-red-200", label: "Angry" };
      case "neutral": return { emoji: "😐", color: "bg-gray-100 text-gray-600 border-gray-200", label: "Neutral" };
      case "surprise": return { emoji: "😲", color: "bg-yellow-100 text-yellow-600 border-yellow-200", label: "Surprise" };
      default: return { emoji: "🤔", color: "bg-purple-100 text-purple-600 border-purple-200", label: "Thinking" };
    }
  };

  const currentStyle = getEmotionStyle(emotion);

  return (
    <div className="min-h-screen bg-[#FFFBF5] text-slate-700 font-sans selection:bg-pink-200">
      {/* Background Blobs */}
      <div className="fixed top-10 left-10 w-64 h-64 bg-pink-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-pulse" />
      <div className="fixed top-10 right-10 w-64 h-64 bg-blue-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-pulse delay-700" />
      <div className="fixed bottom-10 left-1/2 w-64 h-64 bg-yellow-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-pulse delay-1000" />

      <main className="relative z-10 flex flex-col items-center justify-center min-h-screen p-4 md:p-8">
        {/* Header */}
        <div className="text-center mb-8 space-y-2">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full shadow-sm border border-slate-100 mb-2">
            <span className="text-lg">✨</span>
            <span className="text-sm font-medium text-slate-500">Face Emotion (OpenCV + YOLO11-CLS)</span>
            <span className="text-lg">✨</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold text-slate-800 tracking-tight">
            Hu suek ja dai?🤔
          </h1>
          <p className="text-slate-400 font-medium">group ธีร์กิต เคียนลี</p>
        </div>

        {/* Main Card */}
        <div className="w-full max-w-4xl bg-white rounded-[2.5rem] shadow-[0_20px_50px_-12px_rgba(0,0,0,0.08)] border border-slate-100 p-6 md:p-10">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-8">
            
            {/* Left: Camera Area */}
            <div className="md:col-span-7 space-y-4">
              <div className="relative w-full aspect-[4/3] bg-slate-50 rounded-[2rem] overflow-hidden border-4 border-white shadow-inner flex items-center justify-center">
                <video ref={videoRef} className="hidden" playsInline muted />
                <canvas 
                  ref={canvasRef} 
                  className={`w-full h-full object-cover ${!isCameraActive ? 'hidden' : ''}`} 
                />
                {!isCameraActive && (
                  <div className="flex flex-col items-center gap-4 text-slate-300">
                    <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center">
                      <span className="text-4xl">😴</span>
                    </div>
                    <p className="font-medium text-slate-400">Camera is sleeping</p>
                  </div>
                )}
              </div>
              <div className="flex items-center justify-center gap-2">
                <div className={`w-3 h-3 rounded-full ${isCameraActive ? "bg-green-400 animate-bounce" : "bg-slate-300"}`} />
                <span className="text-sm font-medium text-slate-500">{status}</span>
              </div>
            </div>

            {/* Right: Stats Area */}
            <div className="md:col-span-5 flex flex-col gap-4">
              <div className={`flex flex-col items-center justify-center p-8 rounded-[2rem] border-2 transition-all duration-500 ${currentStyle.color} h-full`}>
                <p className="text-sm font-bold uppercase tracking-wider opacity-60 mb-2">Detected Mood</p>
                <div className="text-8xl drop-shadow-md transform transition-transform cursor-default">
                  {currentStyle.emoji}
                </div>
                <h2 className="text-3xl font-black mt-4">{currentStyle.label}</h2>
              </div>

              <div className="bg-slate-50 p-6 rounded-[2rem] border border-slate-100">
                <div className="flex justify-between items-end mb-2">
                   <span className="text-slate-400 font-semibold text-sm">Confidence</span>
                   <span className="text-2xl font-black text-slate-700">{confidence}%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-4 overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-pink-300 to-purple-400 h-full rounded-full transition-all duration-500 ease-out" 
                    style={{ width: `${confidence}%` }}
                  />
                </div>
              </div>

              <button
                onClick={handleStartCamera}
                className={`w-full py-4 rounded-2xl font-bold text-lg transition-all shadow-lg active:scale-95 text-white ${
                  isCameraActive ? "bg-red-500 hover:bg-red-600 shadow-red-100" : "bg-slate-900 hover:bg-slate-800 shadow-slate-200"
                }`}
              >
                {isCameraActive ? "Stop Camera ⏹️" : "Start Camera 📸"}
              </button>
            </div>
          </div>
        </div>

        <p className="mt-8 text-xs text-slate-400 text-center max-w-md">
          * หมายเหตุ: ต้องกดปุ่ม Start เพื่อขอสิทธิ์เปิดกล้องและเริ่มระบบตรวจจับ *
        </p>
      </main>
    </div>
  );
}