from flask import Flask, render_template, request, redirect, session, url_for, send_file
import sqlite3
import os
import shutil
import subprocess
from gemini import analyze_text
import json
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from datetime import datetime
from datetime import timedelta
import cv2

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Lazy-load YOLO model to avoid import-time crash if the file is missing/corrupt
_model = None
def get_model():
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "yolo_model.pt")

            # Prefer a local model if present and non-empty
            if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                _model = YOLO(model_path)
            else:
                app.logger.warning(f"Model file missing or empty: {model_path}. Attempting pretrained fallback 'yolov8n.pt'.")
                try:
                    # Attempt to load a small pretrained model from ultralytics package
                    _model = YOLO('yolov8n.pt')
                except Exception as fe:
                    app.logger.error(f"Failed to load fallback pretrained model 'yolov8n.pt': {fe}")
                    _model = None
        except Exception as e:
            app.logger.error(f"Failed to load YOLO model: {e}")
            _model = None
    return _model


def _is_celebrating(box, frame_height, frame_width):
    """
    Simple heuristic for celebration detection.
    Returns True if bounding box suggests raised arms or upright posture (celebration-like pose).
    """
    try:
        x1, y1, x2, y2 = map(float, box)
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        frame_area = frame_height * frame_width
        
        # if box takes up a significant portion of frame and is tall-ish (raised arms):
        # assume vertical-oriented box -> might be celebrating
        if box_area > 0:
            aspect = box_height / max(box_width, 1.0)
            # tall, narrow box (arms up) vs wide, shorter box (lying down)
            if aspect > 1.2 and box_width < frame_width * 0.25:
                return True
        return False
    except Exception:
        return False


def _convert_video_to_mp4(src_path, dest_path):
    """Convert a video to MP4 (mp4v) for browser playback. Returns True on success."""
    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if width <= 0 or height <= 0:
            ret, frm = cap.read()
            if not ret or frm is None:
                return False
            height, width = frm.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))
        if not out.isOpened():
            return False

        wrote = 0
        while True:
            ret, frm = cap.read()
            if not ret or frm is None:
                break
            if frm.shape[1] != width or frm.shape[0] != height:
                frm = cv2.resize(frm, (width, height))
            out.write(frm)
            wrote += 1

        return wrote > 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    except Exception:
        return False
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if out is not None:
                out.release()
        except Exception:
            pass


def _transcode_to_h264_mp4(src_path, dest_path):
    """Transcode to browser-friendly H.264 MP4 (yuv420p + faststart)."""
    try:
        ffmpeg_bin = shutil.which('ffmpeg')
        if not ffmpeg_bin:
            app.logger.warning("ffmpeg not found in PATH, cannot transcode")
            return False

        if not os.path.exists(src_path) or os.path.getsize(src_path) == 0:
            app.logger.warning(f"Source file missing or empty: {src_path}")
            return False

        target_dir = os.path.dirname(dest_path) or '.'
        os.makedirs(target_dir, exist_ok=True)

        # Use a temp file to support in-place conversion (src == dest)
        tmp_dest = os.path.splitext(dest_path)[0] + '.tmp_h264.mp4'
        cmd = [
            ffmpeg_bin,
            '-y',
            '-i', src_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-preset', 'veryfast',
            '-crf', '23',
            '-an',
            tmp_dest,
        ]

        app.logger.debug(f"Running ffmpeg transcode: {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=300)
        if proc.returncode != 0:
            app.logger.warning(f"ffmpeg failed: {proc.stderr[:200] if proc.stderr else 'no stderr'}")
            try:
                if os.path.exists(tmp_dest):
                    os.remove(tmp_dest)
            except Exception:
                pass
            return False

        if not os.path.exists(tmp_dest) or os.path.getsize(tmp_dest) == 0:
            app.logger.warning(f"Transcode produced empty file: {tmp_dest}")
            try:
                if os.path.exists(tmp_dest):
                    os.remove(tmp_dest)
            except Exception:
                pass
            return False

        os.replace(tmp_dest, dest_path)
        app.logger.info(f"Successfully transcoded: {src_path} -> {dest_path}")
        return True
    except subprocess.TimeoutExpired:
        app.logger.error(f"Transcode timeout for {src_path}")
        try:
            if 'tmp_dest' in locals() and os.path.exists(tmp_dest):
                os.remove(tmp_dest)
        except Exception:
            pass
        return False
    except Exception as e:
        app.logger.error(f"Transcode error for {src_path}: {e}", exc_info=True)
        try:
            if 'tmp_dest' in locals() and os.path.exists(tmp_dest):
                os.remove(tmp_dest)
        except Exception:
            pass
        return False


def _is_h264_mp4(src_path):
    """Return True if src_path is an MP4 file with h264 video codec. Returns True (assume OK) if ffprobe unavailable."""
    try:
        ffprobe = shutil.which('ffprobe')
        if not ffprobe or not os.path.exists(src_path):
            # ffprobe not available, assume file is OK to avoid unnecessary transcoding
            app.logger.debug(f"ffprobe unavailable, assuming {src_path} is playable")
            return True
        cmd = [ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', src_path]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if p.returncode != 0:
            app.logger.debug(f"ffprobe returned error for {src_path}, assuming OK")
            return True
        codec = p.stdout.strip().lower()
        is_h264 = 'h264' in codec or 'avc' in codec
        app.logger.debug(f"Codec check: {src_path} = {codec} (h264={is_h264})")
        return is_h264
    except Exception as e:
        app.logger.debug(f"Codec check failed for {src_path}: {e}, assuming OK")
        return True

# DB connection
def get_db():
    return sqlite3.connect("database.db")


def init_db():
    db = get_db()
    cur = db.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    ''')
    # add created_at column for join timestamp if missing
    try:
        cur.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
    except Exception:
        pass
    cur.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            file TEXT,
            analysis TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    try:
        cur.execute("ALTER TABLE uploads ADD COLUMN created_at TEXT")
    except Exception:
        pass
    cur.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT,
            from_role TEXT,
            created_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    # ensure additional columns exist for older DBs
    try:
        cur.execute("ALTER TABLE messages ADD COLUMN from_role TEXT")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE messages ADD COLUMN created_at TEXT")
    except Exception:
        pass
    cur.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            file TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    db.commit()
    db.close()

# HOME
@app.route('/')
def home():
    return render_template("index.html")

# VIDEO SERVING (with proper MIME types)
@app.route('/video/<filename>')
def serve_video(filename):
    """Serve video files with proper MIME types and range support."""
    # Sanitize filename to prevent path traversal
    safe_name = secure_filename(filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    
    # Verify file exists and is in uploads folder
    if not os.path.exists(video_path) or not os.path.isfile(video_path):
        return "File not found", 404
    
    # Map file extensions to MIME types
    ext = os.path.splitext(safe_name)[1].lower()
    mime_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm',
        '.flv': 'video/x-flv',
        '.wmv': 'video/x-ms-wmv',
    }
    mime_type = mime_map.get(ext, 'video/mp4')

    # If this is a video file, ensure it's browser-playable. Transcode on-demand when needed.
    try:
        video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        served_path = video_path

        if ext in video_exts:
            # If not mp4, transcode to h264 mp4 next to original (cached)
            if ext != '.mp4':
                target_name = os.path.splitext(safe_name)[0] + '.mp4'
                target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_name)
                if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
                    app.logger.info(f"Transcoding {ext} to H.264 MP4: {safe_name}")
                    ok = _transcode_to_h264_mp4(video_path, target_path)
                    if ok and os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                        served_path = target_path
                        app.logger.info(f"Successfully transcoded to {target_name}")
                    else:
                        app.logger.warning(f"Transcode failed for {safe_name}, will serve original")
                else:
                    served_path = target_path
            else:
                # ext == .mp4: verify codec is h264; if not, transcode to cached h264 version
                is_h264 = _is_h264_mp4(video_path)
                if not is_h264:
                    target_name = os.path.splitext(safe_name)[0] + '_h264.mp4'
                    target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_name)
                    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
                        app.logger.info(f"MP4 detected but not H.264, transcoding: {safe_name}")
                        ok = _transcode_to_h264_mp4(video_path, target_path)
                        if ok and os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                            served_path = target_path
                            app.logger.info(f"Successfully transcoded to H.264: {target_name}")
                        else:
                            app.logger.warning(f"H.264 transcode failed for {safe_name}, will serve original")
                    else:
                        served_path = target_path

        # Final check: ensure served file exists and is readable
        if not os.path.exists(served_path) or os.path.getsize(served_path) == 0:
            app.logger.error(f"Served file missing or empty: {served_path}")
            return "File not available", 404

        app.logger.debug(f"Serving video: {os.path.basename(served_path)}")
        return send_file(served_path, mimetype=mime_type, conditional=True)
    except Exception as e:
        app.logger.error(f"Error serving video {safe_name}: {e}", exc_info=True)
        return "Error serving video", 500

# REGISTER
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form.get('role', 'user').lower()

        db = get_db()
        db.execute("INSERT INTO users (name,email,password,role) VALUES (?,?,?,?)",
               (name, email, password, role))
        db.commit()

        return redirect('/login')

    return render_template("register.html")
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email','').strip()
        password = request.form.get('password','').strip()

        db = get_db()
        try:
            # normalize email compare to avoid casing/whitespace mismatches
            user = db.execute("SELECT * FROM users WHERE lower(email)=lower(?) AND password=?", (email, password)).fetchone()
        except Exception:
            user = db.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password)).fetchone()

        if user:
            # defensive role extraction and normalization
            try:
                role = (user[4] if len(user) > 4 and user[4] else 'user').lower()
            except Exception:
                role = 'user'
            session['user_id'] = user[0]
            # store the user's display name in session for templates
            try:
                session['user_name'] = user[1] if len(user) > 1 and user[1] else ''
            except Exception:
                session['user_name'] = ''
            session['role'] = role
            if role in ('admin', 'coach'):
                return redirect('/admin_dashboard')
            return redirect('/user_dashboard')

    return render_template("login.html")


@app.route('/logout')
def logout():
    # Clear the user's session and redirect to login
    try:
        session.clear()
    except Exception:
        pass
    return redirect('/login')

# USER DASHBOARD
@app.route('/user_dashboard')
def user_dashboard():
    # ensure user is logged in
    user_id = session.get('user_id')
    if not user_id:
        return redirect('/login')

    db = get_db()
    user = db.execute("SELECT id, name, email FROM users WHERE id=?", (user_id,)).fetchone()
    db.close()
    if not user:
        session.clear()
        return redirect('/login')

    name = user[1] or ''
    email = user[2] or ''
    return render_template("dashboard_user.html", user_name=name, user_email=email)


# Analysis entrypoints
@app.route('/analyze_single')
def analyze_single():
    return redirect('/upload')


@app.route('/analyze_group')
@app.route('/group-analysis')
def analyze_group():
    db = get_db()
    row = db.execute("SELECT * FROM uploads WHERE user_id=? ORDER BY id DESC LIMIT 1",
                     (session.get('user_id'),)).fetchone()
    if not row:
        return redirect('/user_dashboard')

    filename = row[2]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return redirect('/results')

    model = get_model()
    if model is None:
        return render_template('group_analysis.html', error='YOLO model not available')

    # Run detection and save annotated outputs to a run folder so we can play annotated video
    import time
    run_dir = os.path.join(os.getcwd(), 'runs', 'group_detect', f'run_{int(time.time())}')
    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception:
        pass

    results = None
    annotated_src = None
    try:
        # Try to save annotated outputs (preferred)
        results = model.predict(source=filepath, save=True, save_dir=run_dir)
        # attempt to find annotated file in run_dir
        for root, _, files in os.walk(run_dir):
            for f in files:
                if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                    annotated_src = os.path.join(root, f)
                    break
            if annotated_src:
                break
    except Exception as e:
        # if saving annotated outputs failed (e.g., stacking error), fall back to detection without saving
        app.logger.debug(f"model.predict(save=True) failed: {e}; falling back to non-saving predict")
        try:
            results = model.predict(source=filepath, save=False)
        except Exception:
            try:
                results = model(filepath)
            except Exception as e2:
                return render_template('group_analysis.html', error=f'Detection failed: {e2}')

    results_list = results if isinstance(results, (list, tuple)) else [results]

    detections = []
    ball_detections = []
    names = getattr(model, 'names', {}) or {}
    
    # First pass: collect all ball detections
    for frame_idx, r in enumerate(results_list):
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        cls_idxs = getattr(boxes, 'cls', None)
        confs = getattr(boxes, 'conf', None)
        xyxy = getattr(boxes, 'xyxy', None)
        cls_list = cls_idxs.tolist() if (cls_idxs is not None and hasattr(cls_idxs, 'tolist')) else (list(cls_idxs) if cls_idxs is not None else [])
        conf_list = confs.tolist() if (confs is not None and hasattr(confs, 'tolist')) else (list(confs) if confs is not None else None)
        xy_list = xyxy.tolist() if (xyxy is not None and hasattr(xyxy, 'tolist')) else (list(xyxy) if xyxy is not None else None)
        
        for i, cls_idx in enumerate(cls_list):
            try:
                cls_name = names.get(int(cls_idx), str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
            except Exception:
                cls_name = str(cls_idx)
            lname = cls_name.lower()
            
            if lname in ('sports ball', 'ball'):
                # Collect ball detections first
                bconf = None
                if conf_list is not None and i < len(conf_list):
                    try:
                        bconf = float(conf_list[i])
                    except Exception:
                        bconf = None
                bbox = None
                if xy_list is not None and i < len(xy_list):
                    try:
                        bbox = xy_list[i]
                    except Exception:
                        bbox = None
                if bbox is not None:
                    ball_detections.append({'frame': frame_idx, 'box': bbox, 'conf': bconf})
    
    # Second pass: track only players handling the ball
    for frame_idx, r in enumerate(results_list):
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        cls_idxs = getattr(boxes, 'cls', None)
        confs = getattr(boxes, 'conf', None)
        xyxy = getattr(boxes, 'xyxy', None)
        cls_list = cls_idxs.tolist() if (cls_idxs is not None and hasattr(cls_idxs, 'tolist')) else (list(cls_idxs) if cls_idxs is not None else [])
        conf_list = confs.tolist() if (confs is not None and hasattr(confs, 'tolist')) else (list(confs) if confs is not None else None)
        xy_list = xyxy.tolist() if (xyxy is not None and hasattr(xyxy, 'tolist')) else (list(xyxy) if xyxy is not None else None)
        
        # Get frame dimensions for celebration detection
        frame_h = getattr(r, 'orig_shape', (480, 640))[0] if hasattr(r, 'orig_shape') else 480
        frame_w = getattr(r, 'orig_shape', (480, 640))[1] if hasattr(r, 'orig_shape') else 640
        
        # Check if there's a ball in this frame
        has_ball_this_frame = any(b['frame'] == frame_idx for b in ball_detections)
        
        for i, cls_idx in enumerate(cls_list):
            try:
                cls_name = names.get(int(cls_idx), str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
            except Exception:
                cls_name = str(cls_idx)
            lname = cls_name.lower()
            
            if lname in ('person',):
                # Always detect players
                conf = None
                if conf_list is not None and i < len(conf_list):
                    try:
                        conf = float(conf_list[i])
                    except Exception:
                        conf = None
                box = None
                if xy_list is not None and i < len(xy_list):
                    try:
                        box = xy_list[i]
                    except Exception:
                        box = None
                if box is not None:
                    # Check if this person is handling the ball (when ball is present)
                    is_handling_ball = False
                    if has_ball_this_frame:
                        ball_in_frame = [b for b in ball_detections if b['frame'] == frame_idx]
                        for ball in ball_in_frame:
                            try:
                                px1, py1, px2, py2 = map(float, box)
                                bx1, by1, bx2, by2 = map(float, ball['box'])
                                # Ball center
                                bcx = (bx1 + bx2) / 2.0
                                bcy = (by1 + by2) / 2.0
                                
                                # Check if ball is INSIDE player box (very close contact)
                                ball_in_player_box = (px1 <= bcx <= px2) and (py1 <= bcy <= py2)
                                
                                # If inside, it's definitely handling
                                if ball_in_player_box:
                                    is_handling_ball = True
                                    break
                                
                                # If outside, check distance but VERY strict
                                # Only within player height/4 (foot/hand area only)
                                player_h = max(1.0, py2 - py1)
                                max_dist = player_h / 4.0  # Much stricter
                                
                                pcx = (px1 + px2) / 2.0
                                pcy = (py1 + py2) / 2.0
                                dist = ((pcx - bcx) ** 2 + (pcy - bcy) ** 2) ** 0.5
                                
                                if dist <= max_dist:
                                    is_handling_ball = True
                                    break
                            except Exception:
                                continue
                    
                    # Check if this person is celebrating
                    is_celebrating = _is_celebrating(box, frame_h, frame_w)
                    detections.append({'frame': frame_idx, 'box': box, 'conf': conf, 'celebrating': is_celebrating, 'handling_ball': is_handling_ball})

    def iou(a, b):
        try:
            ax1, ay1, ax2, ay2 = map(float, a)
            bx1, by1, bx2, by2 = map(float, b)
        except Exception:
            return 0.0
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    tracks = []
    for det in detections:
        if det['box'] is None:
            continue
        assigned = None
        for tr in tracks:
            if iou(det['box'], tr['last_box']) > 0.3:
                assigned = tr
                break
        if assigned:
            assigned['boxes'].append((det['frame'], det['box']))
            assigned['conf_sum'] += (det['conf'] or 0.0)
            assigned['count'] += 1
            # Track only ball-handling frames for scoring
            if det.get('handling_ball', False):
                assigned['ball_handling_count'] = assigned.get('ball_handling_count', 0) + 1
                assigned['ball_handling_conf_sum'] = assigned.get('ball_handling_conf_sum', 0.0) + (det['conf'] or 0.0)
            assigned['last_box'] = det['box']
            # Track celebrations
            if det.get('celebrating', False):
                assigned['celebration_frames'].append(det['frame'])
        else:
            tid = len(tracks) + 1
            tracks.append({
                'id': tid, 
                'last_box': det['box'], 
                'boxes': [(det['frame'], det['box'])], 
                'conf_sum': (det['conf'] or 0.0), 
                'count': 1,
                'ball_handling_count': 1 if det.get('handling_ball', False) else 0,
                'ball_handling_conf_sum': (det['conf'] or 0.0) if det.get('handling_ball', False) else 0.0,
                'celebration_frames': [det['frame']] if det.get('celebrating', False) else []
            })

    # initialize ball touch counters on tracks and associate ball detections to nearest player
    for tr in tracks:
        tr['ball_touches'] = 0
        tr['ball_positions'] = []

    try:
        # build simple center points for each track's last_box
        def center(box):
            x1,y1,x2,y2 = map(float, box)
            return ((x1+x2)/2.0, (y1+y2)/2.0)

        for b in ball_detections:
            if b.get('box') is None:
                continue
            bx1,by1,bx2,by2 = map(float, b['box'])
            bcx = (bx1+bx2)/2.0
            bcy = (by1+by2)/2.0
            best_tr = None
            best_dist = None
            for tr in tracks:
                try:
                    tx, ty = center(tr['last_box'])
                except Exception:
                    continue
                dist = ((tx-bcx)**2 + (ty-bcy)**2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_tr = tr
            # threshold based on typical frame size: accept if within 2x bbox diagonal
            if best_tr is not None:
                try:
                    ax1,ay1,ax2,ay2 = map(float, best_tr['last_box'])
                    diag = ((ax2-ax1)**2 + (ay2-ay1)**2) ** 0.5
                    if best_dist is not None and best_dist <= max(50, diag*2.0):
                        best_tr['ball_touches'] += 1
                        best_tr['ball_positions'].append((b['frame'], b['box']))
                except Exception:
                    continue
    except Exception:
        pass

    if not tracks:
        return render_template('group_analysis.html', error='No players detected in video')

    scores = {}
    for tr in tracks:
        # Ball-handling focused scoring: use ball_handling_count (frames while handling)
        ball_handling_frames = tr.get('ball_handling_count', 0)
        avg_conf_while_handling = (tr.get('ball_handling_conf_sum', 0.0) / ball_handling_frames) if ball_handling_frames else 0.0
        celebrations = len(tr.get('celebration_frames', []))
        # Primary score: ball handling time + celebration bonus
        scores[tr['id']] = (ball_handling_frames * avg_conf_while_handling) + (celebrations * 3.0)
    best_id = max(scores.items(), key=lambda x: x[1])[0]
    best_track = next(t for t in tracks if t['id'] == best_id)

    thumbs = []
    try:
        cap = cv2.VideoCapture(filepath)
        saved = 0
        for frame_idx, box in best_track['boxes']:
            if saved >= 5:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frm = cap.read()
            if not ret:
                continue
            try:
                x1, y1, x2, y2 = map(int, box)
                h, w = frm.shape[:2]
                x1 = max(0, min(w-1, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h-1, y1))
                y2 = max(0, min(h, y2))
                crop = frm[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                thumb_name = f"best_player_{row[0]}_{best_id}_{saved}.jpg"
                thumb_path = os.path.join(app.config['UPLOAD_FOLDER'], thumb_name)
                cv2.imwrite(thumb_path, crop)
                thumbs.append(thumb_name)
                saved += 1
            except Exception:
                continue
        cap.release()
    except Exception as e:
        app.logger.error(f"Thumbnail extraction failed: {e}")

    # copy annotated_src to uploads and expose to template if available
    annotated_file = None
    try:
        if annotated_src and os.path.exists(annotated_src):
            ann_ext = os.path.splitext(annotated_src)[1].lower()
            if ann_ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm'):
                annotated_file = f"annotated_group_{os.path.splitext(filename)[0]}.mp4"
            else:
                annotated_file = f"annotated_group_{os.path.splitext(filename)[0]}{ann_ext}"
            annotated_dest = os.path.join(app.config['UPLOAD_FOLDER'], annotated_file)
            if ann_ext == '.mp4':
                shutil.copy2(annotated_src, annotated_dest)
            elif ann_ext in ('.mov', '.avi', '.mkv', '.webm'):
                if not _transcode_to_h264_mp4(annotated_src, annotated_dest):
                    _convert_video_to_mp4(annotated_src, annotated_dest)
                    _transcode_to_h264_mp4(annotated_dest, annotated_dest)
            else:
                shutil.copy2(annotated_src, annotated_dest)

            # Final normalize pass for browser compatibility
            if annotated_file and annotated_file.lower().endswith('.mp4') and os.path.exists(annotated_dest):
                _transcode_to_h264_mp4(annotated_dest, annotated_dest)
    except Exception as e:
        app.logger.debug(f"Could not copy annotated group video: {e}")

    # compute performance accuracy for best performer (average confidence *100)
    perf_accuracy = 0.0
    try:
        if best_track and best_track.get('count', 0):
            perf_accuracy = (best_track.get('conf_sum', 0.0) / best_track.get('count', 1)) * 100.0
    except Exception:
        perf_accuracy = 0.0

    # Collect celebration and ball handling frame data for best performer
    celebration_count = len(best_track.get('celebration_frames', []))
    ball_handling_frames = best_track.get('count', 0)  # Ball-handling frames

    # Prepare detailed stats for all players
    team_stats = []
    for tr in tracks:
        player_celebrations = len(tr.get('celebration_frames', []))
        player_ball_handling = tr.get('count', 0)  # Ball-handling frames (count only tracks ball handlers)
        player_score = scores.get(tr['id'], 0)
        team_stats.append({
            'player_id': tr['id'],
            'ball_handling_frames': player_ball_handling,
            'celebrations': player_celebrations,
            'score': round(player_score, 2),
            'is_best': (tr['id'] == best_id)
        })
    
    # Sort by score descending
    team_stats.sort(key=lambda x: x['score'], reverse=True)

    return render_template('group_analysis.html', best_id=best_id, thumbs=thumbs, scores=scores, annotated_file=annotated_file, performance_accuracy=round(perf_accuracy,2), celebration_count=celebration_count, ball_handling_frames=ball_handling_frames, team_stats=team_stats)


# Group upload + analysis (upload form -> analysis and annotated video)
@app.route('/group_upload', methods=['GET','POST'])
def group_upload():
    if request.method == 'GET':
        return render_template('group_upload.html')

    # POST: handle uploaded video, run detection, track players, create annotated video
    file = request.files.get('file')
    if not file:
        return render_template('group_upload.html', error='No file uploaded')

    filename = secure_filename(file.filename)
    if filename == '':
        filename = file.filename
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    model = get_model()
    if model is None:
        return render_template('group_analysis.html', error='YOLO model not available')

    # Run detection without relying on ultralytics saving behavior
    try:
        results = model.predict(source=filepath, save=False)
    except Exception:
        try:
            results = model(filepath)
        except Exception as e:
            return render_template('group_analysis.html', error=f'Detection failed: {e}')

    results_list = results if isinstance(results, (list, tuple)) else [results]

    # Extract person and ball detections per frame - BALL HANDLING FOCUSED
    detections = []
    ball_detections = []
    names = getattr(model, 'names', {}) or {}
    
    # First pass: collect all ball detections
    for frame_idx, r in enumerate(results_list):
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        cls_idxs = getattr(boxes, 'cls', None)
        confs = getattr(boxes, 'conf', None)
        xyxy = getattr(boxes, 'xyxy', None)
        cls_list = cls_idxs.tolist() if (cls_idxs is not None and hasattr(cls_idxs, 'tolist')) else (list(cls_idxs) if cls_idxs is not None else [])
        conf_list = confs.tolist() if (confs is not None and hasattr(confs, 'tolist')) else (list(confs) if confs is not None else None)
        xy_list = xyxy.tolist() if (xyxy is not None and hasattr(xyxy, 'tolist')) else (list(xyxy) if xyxy is not None else None)
        
        for i, cls_idx in enumerate(cls_list):
            try:
                cls_name = names.get(int(cls_idx), str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
            except Exception:
                cls_name = str(cls_idx)
            lname = cls_name.lower()
            
            if lname in ('sports ball', 'ball'):
                # Collect ball detections first
                bconf = None
                if conf_list is not None and i < len(conf_list):
                    try:
                        bconf = float(conf_list[i])
                    except Exception:
                        bconf = None
                bbox = None
                if xy_list is not None and i < len(xy_list):
                    try:
                        bbox = xy_list[i]
                    except Exception:
                        bbox = None
                if bbox is not None:
                    ball_detections.append({'frame': frame_idx, 'box': bbox, 'conf': bconf})
    
    # Second pass: track only players handling the ball
    for frame_idx, r in enumerate(results_list):
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        cls_idxs = getattr(boxes, 'cls', None)
        confs = getattr(boxes, 'conf', None)
        xyxy = getattr(boxes, 'xyxy', None)
        cls_list = cls_idxs.tolist() if (cls_idxs is not None and hasattr(cls_idxs, 'tolist')) else (list(cls_idxs) if cls_idxs is not None else [])
        conf_list = confs.tolist() if (confs is not None and hasattr(confs, 'tolist')) else (list(confs) if confs is not None else None)
        xy_list = xyxy.tolist() if (xyxy is not None and hasattr(xyxy, 'tolist')) else (list(xyxy) if xyxy is not None else None)
        
        # Get frame dimensions for celebration detection
        frame_h = getattr(r, 'orig_shape', (480, 640))[0] if hasattr(r, 'orig_shape') else 480
        frame_w = getattr(r, 'orig_shape', (480, 640))[1] if hasattr(r, 'orig_shape') else 640
        
        # Check if there's a ball in this frame
        has_ball_this_frame = any(b['frame'] == frame_idx for b in ball_detections)
        
        for i, cls_idx in enumerate(cls_list):
            try:
                cls_name = names.get(int(cls_idx), str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
            except Exception:
                cls_name = str(cls_idx)
            lname = cls_name.lower()
            
            if lname in ('person',):
                # Always detect players
                conf = None
                if conf_list is not None and i < len(conf_list):
                    try:
                        conf = float(conf_list[i])
                    except Exception:
                        conf = None
                box = None
                if xy_list is not None and i < len(xy_list):
                    try:
                        box = xy_list[i]
                    except Exception:
                        box = None
                if box is not None:
                    # Check if this player is handling the ball (when ball is present)
                    is_handling_ball = False
                    if has_ball_this_frame:
                        ball_in_frame = [b for b in ball_detections if b['frame'] == frame_idx]
                        for ball in ball_in_frame:
                            try:
                                px1, py1, px2, py2 = map(float, box)
                                bx1, by1, bx2, by2 = map(float, ball['box'])
                                # Ball center
                                bcx = (bx1 + bx2) / 2.0
                                bcy = (by1 + by2) / 2.0
                                
                                # Check if ball is INSIDE player box (very close contact)
                                ball_in_player_box = (px1 <= bcx <= px2) and (py1 <= bcy <= py2)
                                
                                # If inside, it's definitely handling
                                if ball_in_player_box:
                                    is_handling_ball = True
                                    break
                                
                                # If outside, check distance but VERY strict
                                # Only within player height/4 (foot/hand area only)
                                player_h = max(1.0, py2 - py1)
                                max_dist = player_h / 4.0  # Much stricter
                                
                                pcx = (px1 + px2) / 2.0
                                pcy = (py1 + py2) / 2.0
                                dist = ((pcx - bcx) ** 2 + (pcy - bcy) ** 2) ** 0.5
                                
                                if dist <= max_dist:
                                    is_handling_ball = True
                                    break
                            except Exception:
                                continue
                    
                    # Check if this person is celebrating
                    is_celebrating = _is_celebrating(box, frame_h, frame_w)
                    detections.append({'frame': frame_idx, 'box': box, 'conf': conf, 'celebrating': is_celebrating, 'handling_ball': is_handling_ball})

    # Simple IoU tracker to group person detections into tracks
    def iou(a, b):
        try:
            ax1, ay1, ax2, ay2 = map(float, a)
            bx1, by1, bx2, by2 = map(float, b)
        except Exception:
            return 0.0
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    tracks = []
    for det in detections:
        if det['box'] is None:
            continue
        assigned = None
        for tr in tracks:
            if iou(det['box'], tr['last_box']) > 0.3:
                assigned = tr
                break
        if assigned:
            assigned['boxes'].append((det['frame'], det['box']))
            assigned['conf_sum'] += (det['conf'] or 0.0)
            assigned['count'] += 1
            # Track only ball-handling frames for scoring
            if det.get('handling_ball', False):
                assigned['ball_handling_count'] = assigned.get('ball_handling_count', 0) + 1
                assigned['ball_handling_conf_sum'] = assigned.get('ball_handling_conf_sum', 0.0) + (det['conf'] or 0.0)
            assigned['last_box'] = det['box']
            # Track celebrations
            if det.get('celebrating', False):
                assigned['celebration_frames'].append(det['frame'])
        else:
            tid = len(tracks) + 1
            tracks.append({
                'id': tid, 
                'last_box': det['box'], 
                'boxes': [(det['frame'], det['box'])], 
                'conf_sum': (det['conf'] or 0.0), 
                'count': 1,
                'ball_handling_count': 1 if det.get('handling_ball', False) else 0,
                'ball_handling_conf_sum': (det['conf'] or 0.0) if det.get('handling_ball', False) else 0.0,
                'celebration_frames': [det['frame']] if det.get('celebrating', False) else []
            })

    # initialize ball touch counters on tracks and associate ball detections to nearest player
    for tr in tracks:
        tr['ball_touches'] = 0
        tr['ball_positions'] = []

    try:
        def center(box):
            x1,y1,x2,y2 = map(float, box)
            return ((x1+x2)/2.0, (y1+y2)/2.0)

        for b in ball_detections:
            if b.get('box') is None:
                continue
            bx1,by1,bx2,by2 = map(float, b['box'])
            bcx = (bx1+bx2)/2.0
            bcy = (by1+by2)/2.0
            best_tr = None
            best_dist = None
            for tr in tracks:
                try:
                    tx, ty = center(tr['last_box'])
                except Exception:
                    continue
                dist = ((tx-bcx)**2 + (ty-bcy)**2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_tr = tr
            if best_tr is not None:
                try:
                    ax1,ay1,ax2,ay2 = map(float, best_tr['last_box'])
                    diag = ((ax2-ax1)**2 + (ay2-ay1)**2) ** 0.5
                    if best_dist is not None and best_dist <= max(50, diag*2.0):
                        best_tr['ball_touches'] += 1
                        best_tr['ball_positions'].append((b['frame'], b['box']))
                except Exception:
                    continue
    except Exception:
        pass

    if not tracks:
        return render_template('group_analysis.html', error='No players detected in video')

    # Choose best performer (include ball touches and celebrations in scoring)
    scores = {}
    for tr in tracks:
        # Ball-handling focused scoring: use ball_handling_count (frames while handling)
        ball_handling_frames = tr.get('ball_handling_count', 0)
        avg_conf_while_handling = (tr.get('ball_handling_conf_sum', 0.0) / ball_handling_frames) if ball_handling_frames else 0.0
        celebrations = len(tr.get('celebration_frames', []))
        # Primary score: ball handling time + celebration bonus
        scores[tr['id']] = (ball_handling_frames * avg_conf_while_handling) + (celebrations * 3.0)
    best_id = max(scores.items(), key=lambda x: x[1])[0]
    best_track = next(t for t in tracks if t['id'] == best_id)

    # Create annotated video focusing on best performer using OpenCV (avoids ultralytics save bug)
    annotated_name = f"annotated_group_{os.path.splitext(filename)[0]}.mp4"
    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_name)
    try:
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # If width/height are zero, read a frame to determine size
        if width <= 0 or height <= 0:
            ret0, frm0 = cap.read()
            if ret0 and frm0 is not None:
                h0, w0 = frm0.shape[:2]
                height, width = h0, w0
                # reset to frame 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if width <= 0 or height <= 0:
            raise RuntimeError("Could not determine video dimensions")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))
        # if writer failed to open, try fallback codec/extension
        if not out.isOpened():
            try:
                annotated_path_alt = os.path.splitext(annotated_path)[0] + '.avi'
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(annotated_path_alt, fourcc, fps, (width, height))
                annotated_path = annotated_path_alt
                annotated_name = os.path.basename(annotated_path)
            except Exception:
                out = None

        if out is None or not out.isOpened():
            raise RuntimeError('VideoWriter could not be opened')

        frame_map = {f:box for f,box in best_track['boxes']}
        frame_map_ball = {f:box for f,box in best_track.get('ball_positions', [])}
        celebration_frames = set(best_track.get('celebration_frames', []))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        written = 0
        for fi in range(total_frames):
            ret, frm = cap.read()
            if not ret or frm is None:
                break
            if fi in frame_map and frame_map[fi] is not None:
                try:
                    x1, y1, x2, y2 = map(int, frame_map[fi])
                    h, w = frm.shape[:2]
                    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
                    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
                    cv2.rectangle(frm, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frm, f"Player {best_id}", (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    
                    # Draw celebration indicator if celebrating this frame
                    if fi in celebration_frames:
                        # Draw yellow star/badge at top of player box
                        star_x = x1 + 10
                        star_y = max(10, y1 - 20)
                        cv2.circle(frm, (star_x, star_y), 12, (0,215,255), -1)  # gold/yellow filled circle
                        cv2.putText(frm, '!', (star_x-5, star_y+5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
                except Exception:
                    pass
            # draw ball positions if available for this frame
            if fi in frame_map_ball and frame_map_ball[fi] is not None:
                try:
                    bx1, by1, bx2, by2 = map(int, frame_map_ball[fi])
                    bcx = int((bx1+bx2)/2)
                    bcy = int((by1+by2)/2)
                    cv2.circle(frm, (bcx, bcy), max(4, int(max(bx2-bx1, by2-by1)/4)), (0,0,255), -1)
                    cv2.putText(frm, 'Ball', (bcx+6, bcy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                except Exception:
                    pass
            try:
                # ensure frame has correct size
                if frm.shape[1] != width or frm.shape[0] != height:
                    frm = cv2.resize(frm, (width, height))
                out.write(frm)
                written += 1
            except Exception:
                continue
        cap.release()
        out.release()

        # Validate written file
        if written == 0 or not os.path.exists(annotated_path) or os.path.getsize(annotated_path) == 0:
            try:
                if os.path.exists(annotated_path):
                    os.remove(annotated_path)
            except Exception:
                pass
            annotated_name = None
        else:
            # verify the file can be opened by OpenCV
            try:
                vcap = cv2.VideoCapture(annotated_path)
                ok = vcap.isOpened()
                vcap.release()
                if not ok:
                    try:
                        os.remove(annotated_path)
                    except Exception:
                        pass
                    annotated_name = None
                else:
                    annotated_name = os.path.basename(annotated_path)
            except Exception:
                annotated_name = None

        # Force browser-friendly H.264 MP4 output
        if annotated_name:
            try:
                source_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_name)
                target_name = f"annotated_group_{os.path.splitext(filename)[0]}.mp4"
                target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_name)
                if _transcode_to_h264_mp4(source_path, target_path):
                    annotated_name = target_name
                elif not os.path.exists(target_path) and source_path != target_path:
                    try:
                        shutil.copy2(source_path, target_path)
                        annotated_name = target_name
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception as e:
        app.logger.error(f"Annotated video creation failed: {e}")
        try:
            if 'out' in locals() and out is not None:
                out.release()
        except Exception:
            pass
        annotated_name = None

    # Extract thumbnails for best performer (up to 5)
    thumbs = []
    try:
        cap = cv2.VideoCapture(filepath)
        saved = 0
        for frame_idx, box in best_track['boxes']:
            if saved >= 5:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frm = cap.read()
            if not ret:
                continue
            try:
                x1, y1, x2, y2 = map(int, box)
                h, w = frm.shape[:2]
                x1 = max(0, min(w-1, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h-1, y1))
                y2 = max(0, min(h, y2))
                crop = frm[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                thumb_name = f"best_player_{os.path.splitext(filename)[0]}_{best_id}_{saved}.jpg"
                thumb_path = os.path.join(app.config['UPLOAD_FOLDER'], thumb_name)
                cv2.imwrite(thumb_path, crop)
                thumbs.append(thumb_name)
                saved += 1
            except Exception:
                continue
        cap.release()
    except Exception as e:
        app.logger.error(f"Thumbnail extraction failed: {e}")

    # compute performance accuracy for best performer (average confidence *100)
    perf_accuracy = 0.0
    try:
        if best_track and best_track.get('count', 0):
            perf_accuracy = (best_track.get('conf_sum', 0.0) / best_track.get('count', 1)) * 100.0
    except Exception:
        perf_accuracy = 0.0

    # Collect celebration and ball handling frame data for best performer
    celebration_count = len(best_track.get('celebration_frames', []))
    ball_handling_frames = best_track.get('count', 0)  # Ball-handling frames

    # Prepare detailed stats for all players
    team_stats = []
    for tr in tracks:
        player_celebrations = len(tr.get('celebration_frames', []))
        player_ball_handling = tr.get('count', 0)  # Ball-handling frames (count only tracks ball handlers)
        player_score = scores.get(tr['id'], 0)
        team_stats.append({
            'player_id': tr['id'],
            'ball_handling_frames': player_ball_handling,
            'celebrations': player_celebrations,
            'score': round(player_score, 2),
            'is_best': (tr['id'] == best_id)
        })
    
    # Sort by score descending
    team_stats.sort(key=lambda x: x['score'], reverse=True)

    return render_template('group_analysis.html', best_id=best_id, thumbs=thumbs, scores=scores, annotated_file=annotated_name, performance_accuracy=round(perf_accuracy,2), celebration_count=celebration_count, ball_handling_frames=ball_handling_frames, team_stats=team_stats)

# ADMIN DASHBOARD
@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('role') not in ('admin', 'coach'):
        return redirect('/login')
    db = get_db()
    rows = db.execute("SELECT id,name,email,role FROM users WHERE lower(role) NOT IN (?,?) ORDER BY id DESC", ('admin','coach')).fetchall()
    users = []
    for r in rows:
        try:
            uid = r[0]
            name = r[1]
            email = r[2]
            role = r[3] if len(r) > 3 else 'user'
        except Exception:
            continue
        users.append({
            'id': uid,
            'name': name,
            'email': email,
            'role': role,
            # provide fallback fields the template may reference
            'user_code': uid,
            'joined_at': '',
            'status': 'Active'
        })
    # compute initial stats for dashboard
    total_users = 0
    video_analyses = 0
    growth_rate = 'N/A'
    try:
        total_users = db.execute("SELECT COUNT(*) FROM users WHERE lower(role) NOT IN (?,?)", ('admin','coach')).fetchone()[0]
    except Exception:
        total_users = 0

    try:
        # count uploads with video-like extensions
        video_analyses = db.execute("SELECT COUNT(*) FROM uploads WHERE lower(file) LIKE '%.mp4' OR lower(file) LIKE '%.mov' OR lower(file) LIKE '%.avi' OR lower(file) LIKE '%.mkv' OR lower(file) LIKE '%.webm'").fetchone()[0]
    except Exception:
        video_analyses = 0

    # compute 7-day growth based on uploads.created_at (if available)
    try:
        now = datetime.utcnow()
        end_recent = now
        start_recent = now - timedelta(days=7)
        start_prev = now - timedelta(days=14)
        end_prev = start_recent
        recent_count = db.execute("SELECT COUNT(*) FROM uploads WHERE created_at>=? AND created_at<=?", (start_recent.isoformat(), end_recent.isoformat())).fetchone()[0]
        prev_count = db.execute("SELECT COUNT(*) FROM uploads WHERE created_at>=? AND created_at<?", (start_prev.isoformat(), start_recent.isoformat())).fetchone()[0]
        if prev_count == 0:
            growth_rate = 'N/A' if recent_count == 0 else '+100%'
        else:
            gr = (recent_count - prev_count) / float(prev_count) * 100.0
            growth_rate = f"{gr:+.1f}%"
    except Exception:
        growth_rate = 'N/A'

    return render_template("dashboard_admin.html", users=users, total_users=total_users, video_analyses=video_analyses, growth_rate=growth_rate)


@app.route('/admin_user/<int:user_id>')
def admin_user(user_id):
    if session.get('role') not in ('admin', 'coach'):
        return redirect('/login')
    db = get_db()
    user = db.execute("SELECT id,name,email,role FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        return redirect('/admin_dashboard')
    
    uploads_raw = db.execute("SELECT * FROM uploads WHERE user_id=? ORDER BY id DESC", (user_id,)).fetchall()
    posts_raw = db.execute("SELECT * FROM posts WHERE user_id=? ORDER BY created_at DESC", (user_id,)).fetchall()
    
    # Process uploads: prefer annotated files and ensure they're playable
    uploads = []
    for row in uploads_raw:
        file = row[2]
        analysis_raw = row[3] or ''
        
        # Try to find annotated file (same logic as results.html)
        annotated_file = None
        try:
            parsed = json.loads(analysis_raw) if analysis_raw else None
            if isinstance(parsed, dict):
                af = parsed.get('annotated_file')
                if af:
                    candidate = os.path.join(app.config['UPLOAD_FOLDER'], af)
                    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                        annotated_file = af
        except Exception:
            pass
        
        # Fallback: scan folder for annotated file
        if not annotated_file:
            try:
                base = os.path.splitext(file)[0]
                for f in os.listdir(app.config['UPLOAD_FOLDER']):
                    if f.startswith(f"annotated_{base}"):
                        candidate = os.path.join(app.config['UPLOAD_FOLDER'], f)
                        try:
                            if os.path.getsize(candidate) > 0:
                                annotated_file = f
                                break
                        except Exception:
                            continue
            except Exception:
                pass
        
        # Ensure file is h264 compatible
        display_file = annotated_file or file
        try:
            ext = os.path.splitext(display_file)[1].lower()
            if ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm'}:
                display_path = os.path.join(app.config['UPLOAD_FOLDER'], display_file)
                if os.path.exists(display_path) and os.path.getsize(display_path) > 0:
                    # Try transcoding for browser compatibility
                    if ext != '.mp4':
                        healed = os.path.splitext(display_file)[0] + '.mp4'
                        _transcode_to_h264_mp4(display_path, os.path.join(app.config['UPLOAD_FOLDER'], healed))
                        display_file = healed
                    else:
                        _transcode_to_h264_mp4(display_path, display_path)
        except Exception:
            pass
        
        # Store processed row with display file and analysis
        uploads.append((row[0], row[1], display_file, analysis_raw, row[4] if len(row) > 4 else None))
    
    # Process posts: same annotation logic
    posts = []
    for row in posts_raw:
        file = row[2] if row[2] else None
        
        annotated_file = None
        if file:
            try:
                base = os.path.splitext(file)[0]
                for f in os.listdir(app.config['UPLOAD_FOLDER']):
                    if f.startswith(f"annotated_{base}"):
                        candidate = os.path.join(app.config['UPLOAD_FOLDER'], f)
                        try:
                            if os.path.getsize(candidate) > 0:
                                annotated_file = f
                                break
                        except Exception:
                            continue
            except Exception:
                pass
        
        display_file = annotated_file or file
        if display_file:
            try:
                ext = os.path.splitext(display_file)[1].lower()
                if ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm'}:
                    display_path = os.path.join(app.config['UPLOAD_FOLDER'], display_file)
                    if os.path.exists(display_path) and os.path.getsize(display_path) > 0:
                        if ext != '.mp4':
                            healed = os.path.splitext(display_file)[0] + '.mp4'
                            _transcode_to_h264_mp4(display_path, os.path.join(app.config['UPLOAD_FOLDER'], healed))
                            display_file = healed
                        else:
                            _transcode_to_h264_mp4(display_path, display_path)
            except Exception:
                pass
        
        posts.append((row[0], row[1], display_file, row[3], row[4] if len(row) > 4 else None))
    
    messages = db.execute("SELECT message,from_role,created_at FROM messages WHERE user_id=? ORDER BY id DESC", (user_id,)).fetchall()
    messages = [{'message': m[0], 'from_role': m[1] or 'user', 'created_at': m[2] or ''} for m in messages]
    return render_template('admin_user.html', user=user, uploads=uploads, posts=posts, messages=messages)


@app.route('/admin_post/<int:post_id>')
def admin_post(post_id):
    if session.get('role') not in ('admin', 'coach'):
        return redirect('/login')
    db = get_db()
    post = db.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
    if not post:
        return redirect('/admin_dashboard')
    # find annotated video for the post's file
    post_file = post[2] if len(post) > 2 else None
    annotated_file = None
    try:
        if post_file:
            base = os.path.splitext(post_file)[0]
            for f in os.listdir(app.config['UPLOAD_FOLDER']):
                if f.startswith(f"annotated_{base}"):
                    candidate = os.path.join(app.config['UPLOAD_FOLDER'], f)
                    try:
                        if os.path.getsize(candidate) == 0:
                            continue
                    except Exception:
                        continue
                    # validate video
                    a_ext = os.path.splitext(candidate)[1].lower()
                    if a_ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm'):
                        try:
                            vcap = cv2.VideoCapture(candidate)
                            ok = vcap.isOpened()
                            vcap.release()
                            if ok:
                                annotated_file = f
                                break
                        except Exception:
                            continue
    except Exception:
        annotated_file = None

    # compute performance accuracy from uploads.analysis if available
    perf_accuracy = None
    try:
        if post_file:
            row = db.execute("SELECT analysis FROM uploads WHERE file=? ORDER BY id DESC LIMIT 1", (post_file,)).fetchone()
            if row and row[0]:
                parsed = None
                try:
                    parsed = json.loads(row[0])
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    # direct field
                    if 'performance_accuracy' in parsed:
                        perf_accuracy = parsed.get('performance_accuracy')
                    # normalized field name
                    elif 'performanceAccuracy' in parsed:
                        perf_accuracy = parsed.get('performanceAccuracy')
                    elif 'perf_scores' in parsed and isinstance(parsed['perf_scores'], dict):
                        vals = list(parsed['perf_scores'].values())
                        if vals:
                            best = max(vals)
                            try:
                                bestf = float(best)
                                if bestf <= 1.0:
                                    perf_accuracy = round(bestf * 100.0, 2)
                                elif bestf <= 100.0:
                                    perf_accuracy = round(bestf, 2)
                                else:
                                    perf_accuracy = round(min(bestf, 100.0), 2)
                            except Exception:
                                perf_accuracy = None
    except Exception:
        perf_accuracy = None

    if perf_accuracy is None:
        perf_display = 'N/A'
    else:
        # ensure numeric display
        try:
            perf_display = f"{float(perf_accuracy):.2f}%" if float(perf_accuracy) <= 100 else f"{float(perf_accuracy):.2f}%"
        except Exception:
            perf_display = str(perf_accuracy)

    return render_template('admin_post.html', post=post, annotated_file=annotated_file, performance_accuracy=perf_display)


@app.route('/admin_chat/<int:user_id>')
def admin_chat(user_id):
    if session.get('role') not in ('admin', 'coach'):
        return redirect('/login')
    db = get_db()
    user = db.execute("SELECT id,name,email FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        return redirect('/admin_dashboard')
    msgs = db.execute("SELECT message,from_role,created_at FROM messages WHERE user_id=? ORDER BY id ASC", (user_id,)).fetchall()
    messages = [{'message': m[0], 'from_role': m[1] or 'user', 'created_at': m[2] or ''} for m in msgs]
    return render_template('admin_chat.html', user={'id': user[0], 'name': user[1], 'email': user[2]}, messages=messages)


@app.route('/admin_stats')
def admin_stats():
    if session.get('role') not in ('admin', 'coach'):
        return (json.dumps({'error': 'unauthorized'}), 403, {'Content-Type': 'application/json'})
    db = get_db()
    try:
        total_users = db.execute("SELECT COUNT(*) FROM users WHERE lower(role) NOT IN (?,?)", ('admin','coach')).fetchone()[0]
    except Exception:
        total_users = 0
    try:
        video_analyses = db.execute("SELECT COUNT(*) FROM uploads WHERE lower(file) LIKE '%.mp4' OR lower(file) LIKE '%.mov' OR lower(file) LIKE '%.avi' OR lower(file) LIKE '%.mkv' OR lower(file) LIKE '%.webm'").fetchone()[0]
    except Exception:
        video_analyses = 0

    growth_rate = 'N/A'
    try:
        now = datetime.utcnow()
        end_recent = now
        start_recent = now - timedelta(days=7)
        start_prev = now - timedelta(days=14)
        recent_count = db.execute("SELECT COUNT(*) FROM uploads WHERE created_at>=? AND created_at<=?", (start_recent.isoformat(), end_recent.isoformat())).fetchone()[0]
        prev_count = db.execute("SELECT COUNT(*) FROM uploads WHERE created_at>=? AND created_at<?", (start_prev.isoformat(), start_recent.isoformat())).fetchone()[0]
        if prev_count == 0:
            growth_rate = 'N/A' if recent_count == 0 else '+100%'
        else:
            gr = (recent_count - prev_count) / float(prev_count) * 100.0
            growth_rate = f"{gr:+.1f}%"
    except Exception:
        growth_rate = 'N/A'

    return json.dumps({'total_users': int(total_users), 'video_analyses': int(video_analyses), 'growth_rate': str(growth_rate)}), 200, {'Content-Type': 'application/json'}

# UPLOAD
@app.route('/upload', methods=['GET','POST'])
def upload():
    # require login for uploads
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return redirect('/upload')

        filename = secure_filename(file.filename)
        if filename == '':
            filename = file.filename
        original_ext = os.path.splitext(filename)[1].lower()
        is_video_upload = original_ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

        # Ensure upload folder exists and save file using sanitized name
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # YOLO detection
        model = get_model()
        detection_summary = None
        counts = {}
        perf_scores = {}
        best_performer = None
        if model is None:
            app.logger.error('YOLO model not available; skipping detection.')
            detection_summary = "YOLO model not available."
        else:
            # Use model.predict with safe fallbacks to avoid ultralytics "need at least one array to stack" when saving
            import time
            run_dir = os.path.join(os.getcwd(), 'runs', 'detect', f'upload_{int(time.time())}')
            try:
                os.makedirs(run_dir, exist_ok=True)
            except Exception:
                pass

            results = None
            try:
                # Preferred: try saving annotated outputs
                try:
                    results = model.predict(source=filepath, save=True, save_dir=run_dir)
                except Exception as e_save:
                    app.logger.debug(f"model.predict(save=True) failed: {e_save}; trying non-saving predict.")
                    try:
                        results = model.predict(source=filepath, save=False)
                    except Exception as e_nosave:
                        app.logger.debug(f"model.predict(save=False) failed: {e_nosave}; trying direct call model(filepath).")
                        try:
                            results = model(filepath)
                        except Exception as e_model:
                            app.logger.error(f"YOLO detection failed: {e_model}")
                            results = None
            except Exception as e:
                app.logger.error(f"Unexpected detection error: {e}")
                results = None

            # Normalize results into a list for safe extraction
            results_list = []
            if results is None:
                results_list = []
            else:
                try:
                    results_list = results if isinstance(results, (list, tuple)) else [results]
                except Exception:
                    results_list = []

            # If no results, skip extraction gracefully
            if not results_list:
                detection_summary = "No detections (detection step failed or returned empty results)."
            else:
                try:
                    detections = []
                    names = getattr(model, 'names', {}) or {}
                    for frame_idx, r in enumerate(results_list):
                        boxes = getattr(r, 'boxes', None)
                        if boxes is None:
                            continue
                        cls_idxs = getattr(boxes, 'cls', None)
                        confs = getattr(boxes, 'conf', None)
                        xyxy = getattr(boxes, 'xyxy', None)
                        if cls_idxs is None:
                            continue
                        cls_list = cls_idxs.tolist() if (hasattr(cls_idxs, 'tolist')) else (list(cls_idxs) if cls_idxs is not None else [])
                        conf_list = confs.tolist() if (hasattr(confs, 'tolist')) else (list(confs) if confs is not None else None)
                        xy_list = xyxy.tolist() if (hasattr(xyxy, 'tolist')) else (list(xyxy) if xyxy is not None else None)
                        for i, cls_idx in enumerate(cls_list):
                            try:
                                cls_name = names.get(int(cls_idx), str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                            except Exception:
                                cls_name = str(cls_idx)
                            conf = None
                            if conf_list is not None and i < len(conf_list):
                                try:
                                    conf = float(conf_list[i])
                                except Exception:
                                    conf = None
                            box = None
                            if xy_list is not None and i < len(xy_list):
                                try:
                                    box = xy_list[i]
                                except Exception:
                                    box = None
                            detections.append({"class": cls_name, "conf": conf, "box": box, "frame": frame_idx})

                    # Build player tracks and scores if we have detections
                    if detections:
                        def _iou(a, b):
                            try:
                                ax1, ay1, ax2, ay2 = map(float, a)
                                bx1, by1, bx2, by2 = map(float, b)
                            except Exception:
                                return 0.0
                            inter_x1 = max(ax1, bx1)
                            inter_y1 = max(ay1, by1)
                            inter_x2 = min(ax2, bx2)
                            inter_y2 = min(ay2, by2)
                            inter_w = max(0.0, inter_x2 - inter_x1)
                            inter_h = max(0.0, inter_y2 - inter_y1)
                            inter_area = inter_w * inter_h
                            area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
                            area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
                            union = area_a + area_b - inter_area
                            if union <= 0:
                                return 0.0
                            return inter_area / union

                        player_tracks = []
                        for det in detections:
                            if det['class'].lower() != 'person':
                                continue
                            if det['box'] is None:
                                continue
                            assigned = None
                            for tr in player_tracks:
                                if _iou(det['box'], tr['last_box']) > 0.3:
                                    assigned = tr
                                    break
                            if assigned is not None:
                                assigned['last_box'] = det['box']
                                assigned['count'] += 1
                                assigned['conf_sum'] += (det['conf'] or 0.0)
                                assigned['boxes'].append(det['box'])
                            else:
                                tid = len(player_tracks) + 1
                                player_tracks.append({
                                    'id': tid,
                                    'last_box': det['box'],
                                    'count': 1,
                                    'conf_sum': (det['conf'] or 0.0),
                                    'boxes': [det['box']],
                                })

                        # compute scores
                        for tr in player_tracks:
                            cnt = tr['count']
                            avg_conf = (tr['conf_sum'] / cnt) if cnt else 0.0
                            score = cnt * avg_conf
                            perf_scores[f"player_{tr['id']}"] = score
                            counts[f"player_{tr['id']}"] = cnt

                        if perf_scores:
                            best_performer = max(perf_scores.items(), key=lambda x: x[1])[0]
                            detection_summary = f"Players detected: {len(player_tracks)}; best_performer={best_performer}"
                        else:
                            detection_summary = "No players detected"
                    else:
                        detection_summary = "No detections"
                except Exception as e:
                    app.logger.error(f"Error extracting detections: {e}")
                    detection_summary = "Detection summary unavailable"

            # Try to create a quick annotated image from the first Results object (if possible)
                try:
                    annotated_file = None
                    first_r = results_list[0] if results_list else None
                    if first_r is not None:
                        plot_fn = getattr(first_r, 'plot', None)
                        if callable(plot_fn):
                            img = plot_fn()
                            try:
                                import numpy as _np
                                from PIL import Image
                                if isinstance(img, _np.ndarray):
                                    # Use original filename as base to avoid double-prefixing
                                    orig_base = os.path.splitext(filename)[0]
                                    annotated_name = f"annotated_{orig_base}.jpg"
                                    annotated_dest = os.path.join(app.config['UPLOAD_FOLDER'], annotated_name)
                                    Image.fromarray(img).convert('RGB').save(annotated_dest)
                                    # validate file
                                    try:
                                        if os.path.getsize(annotated_dest) > 0:
                                            from PIL import Image as _PILImage
                                            _PILImage.open(annotated_dest).verify()
                                            annotated_file = annotated_name
                                    except Exception:
                                        try:
                                            os.remove(annotated_dest)
                                        except Exception:
                                            pass
                            except Exception as e:
                                app.logger.debug(f"Could not save plotted image via PIL: {e}")
                except Exception as e:
                    app.logger.debug(f"No annotated image created: {e}")

        # Try to locate YOLO's annotated output for this upload and copy it into uploads
        annotated_src = None
        try:
            search_dirs = []
            try:
                if 'run_dir' in locals() and os.path.isdir(run_dir):
                    search_dirs.append(run_dir)
            except Exception:
                pass
            search_dirs.append(os.path.join(os.getcwd(), 'runs', 'detect'))

            candidates = []
            base_name = os.path.splitext(filename)[0].lower()
            for runs_dir in search_dirs:
                if not os.path.isdir(runs_dir):
                    continue
                for root, _, files in os.walk(runs_dir):
                    for f in files:
                        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.jpg', '.png', '.webm')):
                            stem = os.path.splitext(f)[0].lower()
                            # keep only files that likely belong to the current upload
                            if base_name not in stem and stem not in base_name:
                                continue
                            full = os.path.join(root, f)
                            try:
                                mtime = os.path.getmtime(full)
                            except Exception:
                                mtime = 0
                            candidates.append((mtime, full))

            if candidates:
                # pick the most recently modified file
                candidates.sort(key=lambda x: x[0], reverse=True)
                annotated_src = candidates[0][1]
        except Exception:
            annotated_src = None

        if annotated_src:
            try:
                orig_base = os.path.splitext(filename)[0]
                src_ext = os.path.splitext(annotated_src)[1] or ''
                # For video uploads, always publish MP4 for browser compatibility.
                if is_video_upload:
                    annotated_name = f"annotated_{orig_base}.mp4"
                else:
                    annotated_name = f"annotated_{orig_base}{src_ext}"
                annotated_dest = os.path.join(app.config['UPLOAD_FOLDER'], annotated_name)
                import shutil

                copied_ok = False
                try:
                    if is_video_upload:
                        if src_ext.lower() == '.mp4':
                            shutil.copy2(annotated_src, annotated_dest)
                            copied_ok = True
                        else:
                            copied_ok = _convert_video_to_mp4(annotated_src, annotated_dest)
                            if not copied_ok:
                                # fallback to original upload converted/copied as best effort
                                if original_ext == '.mp4':
                                    shutil.copy2(filepath, annotated_dest)
                                    copied_ok = True
                                else:
                                    copied_ok = _convert_video_to_mp4(filepath, annotated_dest)
                    else:
                        shutil.copy2(annotated_src, annotated_dest)
                        copied_ok = True
                except Exception as e:
                    app.logger.debug(f"Could not process annotated file from {annotated_src}: {e}")

                # If file exists and non-empty, keep it and expose to results (don't aggressively delete)
                try:
                    if copied_ok and os.path.exists(annotated_dest) and os.path.getsize(annotated_dest) > 0:
                        # Final transcode pass for browser compatibility (H.264 baseline-friendly output)
                        if is_video_upload and annotated_dest.lower().endswith('.mp4'):
                            _transcode_to_h264_mp4(annotated_dest, annotated_dest)
                        annotated_file = annotated_name
                except Exception:
                    annotated_file = None
            except Exception as e:
                app.logger.error(f"Failed to copy annotated result: {e}")

        # Gemini Analysis (include detection summary for richer context)
        analysis = analyze_text("Analyze sports performance", detections=detection_summary)

        # Merge our detection metadata (counts, best performer, perf_scores) into the analysis JSON
        try:
            parsed = json.loads(analysis)
            if isinstance(parsed, dict):
                parsed.setdefault('detection_summary', detection_summary)
                parsed.setdefault('detection_counts', {k: int(v) for k, v in (counts.items() if isinstance(counts, dict) or hasattr(counts, 'items') else [])})
                parsed['best_performer'] = best_performer
                parsed['perf_scores'] = {k: float(v) for k, v in (perf_scores.items() if isinstance(perf_scores, dict) or hasattr(perf_scores, 'items') else [])}
                # expose annotated file (if created) so results can use it directly
                try:
                    if 'annotated_file' in locals() and annotated_file:
                        parsed['annotated_file'] = annotated_file
                except Exception:
                    pass
                final_analysis = json.dumps(parsed)
            else:
                final_analysis = analysis
        except Exception:
            final_analysis = analysis

        db = get_db()
        try:
            user_id = session.get('user_id')
        except Exception:
            user_id = None
        created_at = datetime.utcnow().isoformat()
        db.execute("INSERT INTO uploads (user_id,file,analysis,created_at) VALUES (?,?,?,?)",
               (user_id, filename, final_analysis, created_at))
        db.commit()

        return redirect('/results')

    return render_template("upload.html")


# CREATE POST (user-facing)
@app.route('/create_post', methods=['GET','POST'])
def create_post():
    if request.method == 'POST':
        file = request.files.get('file')
        content = request.form.get('content', '')

        filename = None
        if file:
            filename = secure_filename(file.filename)
            if filename == '':
                filename = file.filename
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(dest)

        created_at = datetime.utcnow().isoformat()
        db = get_db()
        db.execute("INSERT INTO posts (user_id,file,content,created_at) VALUES (?,?,?,?)",
                   (session.get('user_id'), filename, content, created_at))
        db.commit()
        return redirect('/user_dashboard')

    return render_template('create_post.html')

# RESULTS
@app.route('/results')
def results():
    # require login to view results
    if 'user_id' not in session:
        return redirect('/login')

    db = get_db()
    rows = db.execute("SELECT * FROM uploads WHERE user_id=?",
                      (session['user_id'],)).fetchall()

    out = []
    video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    browser_video_exts = {'.mp4', '.webm', '.mov'}
    image_exts = {'.jpg', '.jpeg', '.png', '.gif'}
    for row in rows:
        file = row[2]
        analysis_raw = row[3] or ''
        parsed = None
        try:
            parsed = json.loads(analysis_raw)
            # normalize non-dict analysis into a dict with a summary so templates render
            if parsed is None:
                parsed = None
            elif not isinstance(parsed, dict):
                parsed = {'summary': str(parsed)}
        except Exception:
            parsed = None

        ext = os.path.splitext(file)[1].lower()
        is_video = ext in video_exts
        is_image = ext in image_exts

        # Prefer annotated file if present in uploads (annotated_<basename>.*) and validate readability
        # Prefer annotated file declared in analysis JSON
        annotated_file = None
        try:
            if isinstance(parsed, dict):
                af = parsed.get('annotated_file')
                if af:
                    candidate = os.path.join(app.config['UPLOAD_FOLDER'], af)
                    try:
                        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                            annotated_file = af
                    except Exception:
                        annotated_file = None
        except Exception:
            annotated_file = None

        # If not present in analysis JSON, fallback to scanning uploads folder
        if not annotated_file:
            try:
                base = os.path.splitext(file)[0]
                for f in os.listdir(app.config['UPLOAD_FOLDER']):
                    if f.startswith(f"annotated_{base}"):
                        a_ext = os.path.splitext(f)[1].lower()
                        candidate = os.path.join(app.config['UPLOAD_FOLDER'], f)
                        try:
                            if os.path.getsize(candidate) == 0:
                                continue
                        except Exception:
                            continue
                        # prefer files that exist; keep without aggressive validation
                        annotated_file = f
                        break
            except Exception:
                annotated_file = None

        display_file = annotated_file or file
        # If the selected video extension is often not browser-playable, fallback to original upload.
        try:
            if display_file:
                d_ext = os.path.splitext(display_file)[1].lower()
                o_ext = os.path.splitext(file)[1].lower()
                display_path = os.path.join(app.config['UPLOAD_FOLDER'], display_file)
                original_path = os.path.join(app.config['UPLOAD_FOLDER'], file)

                # Heal existing outputs lazily: transcode selected annotated/original video to H.264 MP4.
                if d_ext in video_exts and os.path.exists(display_path) and os.path.getsize(display_path) > 0:
                    if d_ext != '.mp4':
                        healed_name = os.path.splitext(display_file)[0] + '.mp4'
                        healed_path = os.path.join(app.config['UPLOAD_FOLDER'], healed_name)
                        if _transcode_to_h264_mp4(display_path, healed_path):
                            display_file = healed_name
                            display_path = healed_path
                            d_ext = '.mp4'
                    elif d_ext == '.mp4':
                        _transcode_to_h264_mp4(display_path, display_path)

                if d_ext in video_exts and d_ext not in browser_video_exts and o_ext in browser_video_exts:
                    display_file = file

                # If annotated display is unusable, fall back to original and normalize it too.
                chosen_path = os.path.join(app.config['UPLOAD_FOLDER'], display_file)
                if display_file != file and (not os.path.exists(chosen_path) or os.path.getsize(chosen_path) == 0):
                    display_file = file
                    if o_ext in video_exts and os.path.exists(original_path) and os.path.getsize(original_path) > 0:
                        if o_ext != '.mp4':
                            fixed_original = os.path.splitext(file)[0] + '.mp4'
                            fixed_original_path = os.path.join(app.config['UPLOAD_FOLDER'], fixed_original)
                            if _transcode_to_h264_mp4(original_path, fixed_original_path):
                                display_file = fixed_original
                        else:
                            _transcode_to_h264_mp4(original_path, original_path)
        except Exception:
            pass
        display_ext = os.path.splitext(display_file)[1].lower()
        out.append({
            'id': row[0],
            'file': display_file,
            'original_file': file,
            'is_video': display_ext in video_exts,
            'is_image': display_ext in image_exts,
            'analysis': parsed,
            'raw_analysis': analysis_raw,
        })

    return render_template("results.html", data=out)

# COMMUNICATION
@app.route('/message', methods=['POST'])
def message():
    if 'user_id' not in session:
        return redirect('/login')
    msg = request.form.get('msg', '').strip()
    if not msg:
        return redirect('/communication')
    db = get_db()
    from_role = session.get('role', 'user')
    created_at = datetime.utcnow().isoformat()
    # store message for this user
    db.execute("INSERT INTO messages (user_id,message,from_role,created_at) VALUES (?,?,?,?)",
               (session['user_id'], msg, from_role, created_at))
    db.commit()
    return redirect('/communication')


@app.route('/communication')
def communication():
    if 'user_id' not in session:
        return redirect('/login')
    db = get_db()
    if session.get('role') in ('admin','coach'):
        # load all users and their messages
        users = []
        rows = db.execute("SELECT id,name,email FROM users WHERE lower(role) NOT IN (?,?) ORDER BY id", ('admin','coach')).fetchall()
        for r in rows:
            uid = r[0]
            msgs = db.execute("SELECT message,from_role,created_at FROM messages WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
            users.append({'id': uid, 'name': r[1], 'email': r[2], 'messages': [{'message': m[0], 'from_role': m[1] or 'user', 'created_at': m[2] or ''} for m in msgs]})
        return render_template('communication.html', users=users)
    else:
        msgs = db.execute("SELECT message,from_role,created_at FROM messages WHERE user_id=? ORDER BY id DESC", (session['user_id'],)).fetchall()
        messages = [{'message': m[0], 'from_role': m[1] or 'user', 'created_at': m[2] or ''} for m in msgs]
        return render_template('communication.html', messages=messages)


@app.route('/reply_message', methods=['POST'])
def reply_message():
    # coach/admin replies to a specific user
    if session.get('role') not in ('admin','coach'):
        return redirect('/user_dashboard')
    user_id = request.form.get('user_id')
    msg = request.form.get('msg', '').strip()
    if not user_id or not msg:
        return redirect('/communication')
    db = get_db()
    created_at = datetime.utcnow().isoformat()
    from_role = session.get('role') or 'coach'
    db.execute("INSERT INTO messages (user_id,message,from_role,created_at) VALUES (?,?,?,?)",
               (int(user_id), msg, from_role, created_at))
    db.commit()
    try:
        return redirect(url_for('admin_chat', user_id=int(user_id)))
    except Exception:
        return redirect('/communication')

if __name__ == "__main__":
    try:
        init_db()
    except Exception as e:
        app.logger.error(f"Failed to initialize database: {e}")
    app.run(debug=True)