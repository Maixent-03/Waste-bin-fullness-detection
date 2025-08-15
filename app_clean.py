from flask import Flask, render_template, request, jsonify, Response
import requests
import base64
import io
import logging
import sys
import os
import threading
import time
import cv2
import numpy as np
import subprocess
import socket
import re
from datetime import datetime, timedelta

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration YOLO - Remplacez par vos vraies clés
MODEL_CONFIG = {
    'api_url': 'https://predict.ultralytics.com',
    'api_key': 'bd41b19d868948887eaddef8d532d913090b8650d3',
    #v2 model
    'model_url': 'https://hub.ultralytics.com/models/q5kI8MteQ82AjWUlnHJU',
    #v1 model
    # 'model_url': "https://hub.ultralytics.com/models/NSshsr9eErhT2IvRAwmH"
    'imgsz': 640,
    'conf': 0.25,
    'iou': 0.45
}

# Camera Configuration
RTSP_URL = "rtsp://admin:camera1234@10.10.52.38/onvif1"  # Camera Configuration
RTSP_OPTIONS = {
    'rtsp_transport': 'udp',  # Force UDP transport
    'buffer_size': 1024*1024,
    'max_delay': 5000000
}

# Authorized Classes 
ALLOWED_CLASSES = ['empty', 'full', 'overloaded']  

# Fillness level configuration
FILL_LEVELS = {
    'empty': {'label': 'Empty', 'class': 'empty', 'icon': '🟢', 'priority': 1},
    'full': {'label': 'Full', 'class': 'full', 'icon': '🔴', 'priority': 2},
    'overloaded': {'label': 'Overloaded', 'class': 'overloaded', 'icon': '🚨', 'priority': 3}
}

@app.route('/')
def index():
    """Main page"""
    return render_template('index2.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Endpoint to analyze an image with YOLO"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read and process the image
        image_data = file.read()
        logger.info(f"Image received: {len(image_data)} bytes")
        
        # Get iou/conf from form or use default
        iou = float(request.form.get('iou', MODEL_CONFIG['iou']))
        conf = float(request.form.get('conf', MODEL_CONFIG['conf']))
        
        # Call YOLO API
        detections = run_yolo_detection(image_data, iou=iou, conf=conf)
        
        # Calculate statistics
        stats = calculate_statistics(detections)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add these new global variables for rate limiting
api_rate_limited = False
next_api_call_time = datetime.now()
api_calls_in_period = 0
API_CALL_LIMIT = 90  # Leave some margin below the 100/hour limit
API_CALL_PERIOD = timedelta(hours=1)

# Add near other global variables
current_iou = MODEL_CONFIG['iou']
current_conf = MODEL_CONFIG['conf']

def run_yolo_detection(image_data, iou=None, conf=None):
    """Run YOLO detection on an image with rate limiting"""
    global api_rate_limited, next_api_call_time, api_calls_in_period
    
    try:
        # Check for rate limiting
        now = datetime.now()
        if api_rate_limited and now < next_api_call_time:
            wait_minutes = round((next_api_call_time - now).total_seconds() / 60)
            logger.warning(f"API still rate limited. Please wait ~{wait_minutes} minutes.")
            return []
            
        # Reset rate limiting if cool-down period has passed
        if api_rate_limited and now >= next_api_call_time:
            api_rate_limited = False
            api_calls_in_period = 0
            logger.info("API rate limit cool-down period has ended. Resuming API calls.")
            
        # If we've reached our safe threshold, enforce rate limiting
        if api_calls_in_period >= API_CALL_LIMIT:
            api_rate_limited = True
            next_api_call_time = now + timedelta(minutes=35)  # Wait 35 minutes
            logger.warning(f"Approaching API rate limit. Pausing API calls until {next_api_call_time}.")
            return []
            
        # Track this API call
        api_calls_in_period += 1
        
        logger.info("Starting YOLO detection...")
        
        # Prepare API request
        files = {'file': ('image.jpg', image_data, 'image/jpeg')}
        data = {
            'model': MODEL_CONFIG['model_url'],
            'imgsz': MODEL_CONFIG['imgsz'],
            'conf': conf if conf is not None else MODEL_CONFIG['conf'],
            'iou': iou if iou is not None else MODEL_CONFIG['iou']
        }
        headers = {
            'x-api-key': MODEL_CONFIG['api_key']
        }
        
        # Make API request
        response = requests.post(
            MODEL_CONFIG['api_url'],
            files=files,
            data=data,
            headers=headers,
            timeout=30
        )
        
        logger.info(f"API Response: {response.status_code}")
        
        # Handle 429 Rate Limit response specifically
        if response.status_code == 429:
            api_rate_limited = True
            # Parse the wait time from the error message if possible
            wait_time = 35  # Default 35 minutes
            try:
                error_data = response.json()
                message = error_data.get('message', '')
                import re
                wait_match = re.search(r'wait (\d+) minutes', message)
                if wait_match:
                    wait_time = int(wait_match.group(1))
            except:
                pass
                
            next_api_call_time = now + timedelta(minutes=wait_time)
            logger.warning(f"API rate limited. Waiting until {next_api_call_time}")
            return []
            
        if response.status_code == 200:
            result = response.json()
            detections = parse_yolo_results(result)
            # Filter by confidence
            conf_threshold = conf if conf is not None else MODEL_CONFIG['conf']
            detections = [d for d in detections if d['confidence'] >= conf_threshold]
            return detections
        else:
            logger.warning(f"API error: {response.status_code} - {response.text}")
            return []
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"YOLO error: {str(e)}")
        return []

def parse_yolo_results(results):
    """Parse YOLO API results"""
    detections = []
    try:
        logger.info(f"Parsing results: {results}")

        # New Ultralytics format
        if results and 'images' in results and isinstance(results['images'], list):
            for image in results['images']:
                if 'results' in image and isinstance(image['results'], list):
                    for i, detection in enumerate(image['results']):
                        try:
                            class_name = map_class_name(detection.get('name', detection.get('class', 'empty')))
                            confidence = detection.get('confidence', detection.get('conf', 0.5))
                            if 'box' in detection:
                                box = detection['box']
                                bbox = [
                                    float(min(box.get('x1', 0), box.get('x2', 0))),
                                    float(min(box.get('y1', 0), box.get('y2', 0))),
                                    float(abs(box.get('x2', 100) - box.get('x1', 0))),
                                    float(abs(box.get('y2', 100) - box.get('y1', 0)))
                                ]
                            else:
                                bbox = detection.get('bbox', [100 + i * 200, 100, 150, 200])
                            if class_name in ALLOWED_CLASSES:
                                detections.append({
                                    'class': class_name,
                                    'confidence': float(confidence),
                                    'bbox': bbox,
                                    'original_class': detection.get('name', detection.get('class', 'unknown'))
                                })
                        except Exception as e:
                            logger.warning(f"Error parsing detection {i}: {str(e)}")
                            continue
        else:
            # Old format or other
            data_array = None
            if results and 'data' in results and isinstance(results['data'], list):
                data_array = results['data']
            elif results and isinstance(results, list):
                data_array = results
            elif results and 'predictions' in results:
                data_array = results['predictions']

            if data_array:
                for i, detection in enumerate(data_array):
                    try:
                        class_name = map_class_name(detection.get('name', detection.get('class', 'empty')))
                        confidence = detection.get('confidence', detection.get('conf', 0.5))
                        if 'box' in detection:
                            box = detection['box']
                            bbox = [
                                float(min(box.get('x1', 0), box.get('x2', 0))),
                                float(min(box.get('y1', 0), box.get('y2', 0))),
                                float(abs(box.get('x2', 100) - box.get('x1', 0))),
                                float(abs(box.get('y2', 100) - box.get('y1', 0)))
                            ]
                        else:
                            bbox = detection.get('bbox', [100 + i * 200, 100, 150, 200])
                        if class_name in ALLOWED_CLASSES:
                            detections.append({
                                'class': class_name,
                                'confidence': float(confidence),
                                'bbox': bbox,
                                'original_class': detection.get('name', detection.get('class', 'unknown'))
                            })
                    except Exception as e:
                        logger.warning(f"Error parsing detection {i}: {str(e)}")
                        continue

        logger.info(f"Parsed detections: {len(detections)}")
        return detections

    except Exception as e:
        logger.error(f"Error parsing: {str(e)}")
        return []  # Return empty list on error

def map_class_name(original_name):
    """Map class names to authorized classes"""
    if not original_name:
        return 'empty'
    
    name = str(original_name).lower()
    
    # Class mapping
    class_mapping = {
        'empty': 'empty',
        'full': 'full',
        'overloaded': 'overloaded',
        'bin_empty': 'empty',
        'bin_full': 'full',
        'bin_overloaded': 'overloaded',
        '0': 'empty',
        '1': 'full',
        '2': 'overloaded',
        'low': 'empty',
        'high': 'full',
        'medium': 'full',  # Medium -> Full for simplicity
        'class_one': 'empty',        
        'class_two': 'full',         
        'class_three': 'overloaded'  
    }
    
    return class_mapping.get(original_name, 'empty')


def calculate_statistics(detections):
    """Calculate detection statistics"""
    if not detections:
        return {
            'fill_status': 'No Data',
            'bin_count': 0,
            'alert_level': 'None',
            'avg_confidence': 0
        }
    
    # Calculate average confidence
    avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
    
    # Determine overall status based on priority
    highest_priority = 0
    overall_status = FILL_LEVELS['empty']
    
    for detection in detections:
        level = FILL_LEVELS.get(detection['class'], FILL_LEVELS['empty'])
        if level['priority'] > highest_priority:
            highest_priority = level['priority']
            overall_status = level
    
    # Determine alert level
    alert_level = 'Good'
    if any(d['class'] == 'overloaded' for d in detections):
        alert_level = 'Critical'
    elif any(d['class'] == 'full' for d in detections):
        alert_level = 'Warning'
    
    return {
        'fill_status': overall_status['label'],
        'bin_count': len(detections),
        'alert_level': alert_level,
        'avg_confidence': round(avg_confidence * 100)
    }

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_config': {
            'classes': ALLOWED_CLASSES,
            'confidence_threshold': MODEL_CONFIG['conf'],
            'iou_threshold': MODEL_CONFIG['iou']
        }
    })

# Global variables to store the current frame and detections
current_frame = None
frame_lock = threading.Lock()  # Properly initialize frame_lock
current_detections = []
detections_lock = threading.Lock()
analyze_frames = True  # Flag to control whether to analyze frames
last_analysis_time = 0

def gen_rtsp_stream():
    """Generate MJPEG stream from FFmpeg frames"""
    global current_frame, last_analysis_time
    
    def create_status_frame(message, color=(255, 255, 255)):
        """Create a status frame with helpful information"""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background gradient for better visibility
        for i in range(480):
            intensity = int(30 + (i / 480.0) * 20)
            blank[i, :] = [intensity, intensity, intensity]
        
        # Main message
        cv2.putText(blank, message, (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(blank, f"Time: {timestamp}", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Camera info
        cv2.putText(blank, "Camera: 10.10.52.38/onvif1", (50, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Check if we actually have current frames
        frame_available = current_frame is not None
        
        # Connection status based on actual frame availability
        if frame_available:
            status_color = (0, 255, 0)  # Green
            status_text = "CONNECTED"
        else:
            status_color = (0, 255, 255)  # Yellow
            status_text = "INITIALIZING"
            
        cv2.putText(blank, f"Status: {status_text}", (50, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return blank
    
    # Send initial frame
    initial_frame = create_status_frame("Connecting to camera...", (0, 255, 255))
    ret, jpeg = cv2.imencode('.jpg', initial_frame)
    frame_bytes = jpeg.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    frame_count = 0
    last_frame_time = time.time()
    
    while True:
        try:
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                    last_frame_time = time.time()
                    frame_count += 1
                    
                    # Log occasionally to confirm frames are flowing
                    if frame_count % 50 == 0:
                        print(f"📺 Video stream: {frame_count} frames sent to browser")
                else:
                    # Check how long we've been waiting for frames
                    waiting_time = time.time() - last_frame_time
                    
                    if waiting_time < 10:
                        message = "Establishing connection..."
                        color = (0, 255, 255)  # Yellow
                    elif waiting_time < 30:
                        message = "Waiting for camera response..."
                        color = (255, 255, 0)  # Cyan
                    else:
                        message = "Checking camera connection..."
                        color = (255, 165, 0)  # Orange
                    
                    frame = create_status_frame(message, color)
                    frame_count += 1
            
            # Convert frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Sleep to control frame rate (~5 FPS server-side)
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error in stream generator: {e}")
            error_frame = create_status_frame(f"Stream Error: {str(e)[:50]}", (0, 0, 255))
            ret, jpeg = cv2.imencode('.jpg', error_frame)
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.2)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    try:
        logger.info(f"Video feed requested - current_frame available: {current_frame is not None}")
        response = Response(gen_rtsp_stream(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        logger.error(f"Video feed error: {e}")
        return f"Video feed error: {e}", 500

@app.route('/test-frame')
def test_frame():
    """Serve a single frame as JPEG for testing"""
    try:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    response = Response(jpeg.tobytes(), mimetype='image/jpeg')
                    response.headers['Cache-Control'] = 'no-cache'
                    return response
        
        # Return a test frame if no camera frame available
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, "No camera frame available", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', test_frame)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Test frame error: {e}")
        return f"Error: {e}", 500

@app.route('/api/camera-status')
def camera_status():
    """Check camera connectivity status"""
    try:
        import socket
        # Test network connectivity
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('10.10.52.38', 554))
        sock.close()
        
        network_ok = result == 0
        
        # Check if we have recent frames
        frame_available = current_frame is not None
        
        return jsonify({
            'network_reachable': network_ok,
            'stream_active': frame_available,
            'current_frame_available': frame_available,
            'message': 'Camera connected and frames available' if frame_available else 'No frames available'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'network_reachable': False,
            'stream_active': False
        })

@app.route('/api/latest-detections')
def get_latest_detections():
    """Return the latest trash bin detections"""
    try:
        # Include rate limit information
        if api_rate_limited:
            wait_seconds = max(0, (next_api_call_time - datetime.now()).total_seconds())
            return jsonify({
                'success': False,
                'rate_limited': True,
                'wait_seconds': wait_seconds,
                'next_api_call_time': next_api_call_time.isoformat(),
                'message': f'API rate limited. Please wait {wait_seconds/60:.1f} minutes.'
            })
            
        with detections_lock:
            if current_detections and len(current_detections) > 0:
                stats = calculate_statistics(current_detections)
                return jsonify({
                    'success': True,
                    'detections': current_detections,
                    'statistics': stats,
                    'last_update': last_analysis_time,
                    'rate_limited': False
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No detections available yet',
                    'rate_limited': False
                })
    except Exception as e:
        print(f"Error in /api/latest-detections: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'rate_limited': False
        })

@app.route('/api/set-params', methods=['POST'])
def set_params():
    """Update detection parameters for real-time analysis"""
    global current_iou, current_conf
    try:
        data = request.get_json()
        current_iou = float(data.get('iou', MODEL_CONFIG['iou']))
        current_conf = float(data.get('conf', MODEL_CONFIG['conf']))
        return jsonify({'success': True, 'iou': current_iou, 'conf': current_conf})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def ffmpeg_udp_camera_stream():
    """Use FFmpeg with UDP transport to capture camera frames - THE SOLUTION!"""
    global current_frame
    
    print("🎯 Starting FFmpeg UDP camera connection...")
    print(f"📡 Camera: {RTSP_URL}")
    print("🔌 Transport: UDP (confirmed working)")
    
    # FFmpeg command for UDP transport
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "udp",
        "-i", RTSP_URL,
        "-f", "image2pipe",
        "-pix_fmt", "bgr24", 
        "-vcodec", "rawvideo",
        "-s", "640x480",  # Resize to 640x480
        "-"
    ]
    
    try:
        print("🚀 Starting FFmpeg process...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        frame_size = 640 * 480 * 3  # width * height * channels
        frame_count = 0
        
        print("✅ FFmpeg process started, reading frames...")
        
        while True:
            # Read raw frame data
            raw_frame = process.stdout.read(frame_size)
            
            if len(raw_frame) != frame_size:
                print(f"❌ Frame size mismatch: expected {frame_size}, got {len(raw_frame)}")
                break
                
            # Convert raw data to numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((480, 640, 3))
            
            # Update global frame
            with frame_lock:
                current_frame = frame
                
            frame_count += 1
            if frame_count % 150 == 0:  # Every ~10 seconds at 15 FPS
                print(f"📺 FFmpeg Camera: {frame_count} frames processed successfully")
                
    except Exception as e:
        print(f"❌ FFmpeg camera error: {e}")
    finally:
        if 'process' in locals():
            process.terminate()
            process.wait()
            print("📹 FFmpeg camera process terminated")

if __name__ == '__main__':
    # Create a directory for snapshots if it doesn't exist
    if not os.path.exists('snapshots'):
        os.makedirs('snapshots')
        print("Created snapshots directory")
    
    # Start FFmpeg UDP camera connection (THE WORKING SOLUTION!)
    camera_thread = threading.Thread(target=ffmpeg_udp_camera_stream, daemon=True)
    camera_thread.start()
    
    # Start the Flask server
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
