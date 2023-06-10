import React, { useEffect, useRef } from 'react';

import cv2 from "@techstark/opencv-js";





const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef();
  
  let ws = useRef(null);

  useEffect(() => {
    const startCamera = async () => {

        ws.current = new WebSocket('ws://localhost:8000/ws');
        ws.current.onopen = () => {
          //ws.current.send("Connected");
          
          console.log('WebSocket connection established');
        };

      try {

        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
        console.log(videoRef.current)

        
        videoRef.current.addEventListener('loadedmetadata', () => { 
        console.log("inti lisnter")    
        // Capture face frames and send them to the backend for predictions
        const captureFrames = () => {
        
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        const video = videoRef.current;
       
        
        
        // Set canvas dimensions to match the video frame size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Capture face frame and send it to the backend
        const sendFrameToBackend = () => {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        console.log("this canvas",canvas)
        
        const imageData = cv2.imread(canvas);

        const res = canvas.toDataURL();
        console.log("this is imagedata",res)

        
        console.log("---------------------------------------")
        
        ws.current.send(res);

        // Schedule the next frame capture
        const delayTime = 1000; // 1 second delay
        setTimeout(() => {
          requestAnimationFrame(sendFrameToBackend);
        }, delayTime);
        };
        /*requestAnimationFrame(sendFrameToBackend);
        };*/

        // Start capturing frames
        sendFrameToBackend();
        };
      
       captureFrames();
      },)
      
      } catch (error) {
        console.error('Error accessing camera:', error);
      }
    }

    startCamera();
  }, []);








  useEffect(() => {
    const handlePrediction = (event) => {
      const data = JSON.parse(event.data);
      console.log('Prediction:', data);
    };

    // Subscribe to WebSocket messages for predictions
    ws.current.onmessage = handlePrediction;

    // Cleanup function to close WebSocket connection
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  return (
    <div>
      <video ref={videoRef} autoPlay muted />
    </div>
  );
};

export default App;
