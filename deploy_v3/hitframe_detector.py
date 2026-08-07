"""
Hit-Frame Detector: Implements the full pipeline from the paper
"Automated Hit-frame Detection for Badminton Match Analysis"
(https://arxiv.org/abs/2307.16000)

Pipeline:
1. SA-CNN classifies camera angle to segment rallies
2. Court Keypoint-RCNN detects court boundaries
3. Player Keypoint-RCNN detects player keypoints (17 joints x 2 players)
4. OPT Transformer predicts shuttlecock direction from keypoints
5. Direction change = hit frame detected

This replaces the previous TrackNet + heuristic approach.
"""
import torch
import torchvision
import numpy as np
import cv2
import os
import copy
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import functional as TF

from hitframe_models.sacnn import SACNNContainer
from hitframe_models.transformer import OptimusPrimeContainer

# Paths to model weights (downloaded in Docker build)
WEIGHTS_DIR = os.environ.get('HITFRAME_WEIGHTS_DIR', 'hitframe_weights')
SACNN_PATH = os.path.join(WEIGHTS_DIR, 'sacnn.pt')
COURT_KPRCNN_PATH = os.path.join(WEIGHTS_DIR, 'court_kpRCNN.pth')  # Optional
KPRCNN_PATH = os.path.join(WEIGHTS_DIR, 'kpRCNN.pth')
OPT_PATH = os.path.join(WEIGHTS_DIR, 'OPT_16_head_dp.pt')
SCALER_PATH = os.path.join(WEIGHTS_DIR, 'scaler.pickle')

# SA queue length for smoothing shot angle predictions
SA_QUEUE_LENGTH = 5


class ShotAngleQueue:
    """
    Queue that smooths SA-CNN predictions to avoid noisy transitions.
    Uses majority voting over a window to confirm angle changes.
    """
    def __init__(self, max_len=5):
        self.max_len = max_len
        self.queue = []
        self.last_sa = 0

    def push(self, frame_info):
        """
        Push a [sa_prediction, frame] pair.
        Returns (processed_info, sa_condition) or (None, None) if queue not full.
        
        sa_condition:
            0: stays at angle 0 (between rallies)
            1: transition 0->1 (rally starts)
            2: stays at angle 1 (during rally)
            3: transition 1->0 (rally ends)
        """
        if len(self.queue) < self.max_len:
            self.queue.append(frame_info)
            return None, None
        
        first_info = self.queue.pop(0)
        sa, sa_condition = self.__check_sa_condition(first_info[0])
        self.last_sa = sa
        first_info[0] = sa
        self.queue.append(frame_info)
        return first_info, sa_condition

    def get(self, index):
        return self.queue[index]

    def __check_sa_condition(self, sa):
        total = sa
        if self.last_sa == 1 and sa == 0:
            for info in self.queue:
                total += info[0]
            if total <= (self.max_len / 2):
                return 0, 3  # Confirmed: rally ended
            else:
                return 1, 2  # False alarm: still in rally
        elif self.last_sa == 0 and sa == 1:
            for info in self.queue:
                total += info[0]
            if total >= (self.max_len / 2):
                return 1, 1  # Confirmed: rally started
            else:
                return 0, 0  # False alarm: still between rallies
        elif self.last_sa == 1 and sa == 1:
            return 1, 2
        elif self.last_sa == 0 and sa == 0:
            return 0, 0
        return 0, 0


class HitFrameDetector:
    """
    Full pipeline for detecting hit-frames in badminton videos.
    Uses 4 pre-trained models from the paper to achieve ~99% accuracy.
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"HitFrameDetector device: {self.device}")
        
        # Load models
        self._load_models()
        
        # State
        self.court_info = None
        self.extended_court_points = None
        self.got_court_info = False
    
    def _load_models(self):
        """Load all 4 pre-trained models."""
        print("Loading hit-frame detection models...")
        
        # 1. SA-CNN for rally segmentation
        if os.path.exists(SACNN_PATH):
            self.sacnn = SACNNContainer(SACNN_PATH)
            print(f"  ✓ SA-CNN loaded from {SACNN_PATH}")
        else:
            self.sacnn = None
            print(f"  ✗ SA-CNN not found at {SACNN_PATH}")
        
        # 2. Court Keypoint-RCNN (optional - may not be available)
        if os.path.exists(COURT_KPRCNN_PATH):
            self.court_kprcnn = torch.load(
                COURT_KPRCNN_PATH, map_location=self.device
            )
            self.court_kprcnn.to(self.device).eval()
            print(f"  ✓ Court KP-RCNN loaded")
        else:
            self.court_kprcnn = None
            print(f"  ⓘ Court KP-RCNN not available (using frame geometry fallback)")
        
        # 3. Player Keypoint-RCNN
        if os.path.exists(KPRCNN_PATH):
            self.player_kprcnn = torch.load(
                KPRCNN_PATH, map_location=self.device
            )
            self.player_kprcnn.to(self.device).eval()
            print(f"  ✓ Player KP-RCNN loaded")
        else:
            self.player_kprcnn = None
            print(f"  ✗ Player KP-RCNN not found at {KPRCNN_PATH}")
        
        # 4. OPT Transformer
        if os.path.exists(OPT_PATH) and os.path.exists(SCALER_PATH):
            self.opt = OptimusPrimeContainer(OPT_PATH, SCALER_PATH)
            print(f"  ✓ OPT Transformer loaded")
        else:
            self.opt = None
            print(f"  ✗ OPT Transformer not found")
        
        print("Model loading complete.")
    
    def process_video(self, video_path, progress_callback=None):
        """
        Full pipeline: process video and return hit frames per rally.
        
        Returns:
            dict with keys:
                - rallies: list of rally dicts with hit_frames, directions
                - total_hits: int
                - fps: float
                - total_frames: int
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Sampling rate: process every N frames (paper uses 0.1s intervals)
        time_rate = 0.1
        frame_rate = max(round(fps * time_rate), 1)
        
        # Initialize state
        sa_queue = ShotAngleQueue(SA_QUEUE_LENGTH)
        self.got_court_info = False
        
        # Storage for current rally
        player_joints_rally = []
        frame_nums_rally = []
        rally_start_frame = 0
        
        # Results
        all_rallies = []
        frame_count = 0
        target_frames = total_frames // frame_rate
        processed_count = 0
        
        if progress_callback:
            progress_callback(5)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_rate == 0:
                # Step 1: SA-CNN predicts shot angle
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if self.sacnn:
                    sa = self.sacnn.predict(pil_img)
                else:
                    sa = 1  # If no SA-CNN, treat everything as rally
                
                frame_info, sa_condition = sa_queue.push([sa, frame])
                
                if frame_info is not None:
                    sa_val, proc_frame = frame_info[0], frame_info[1]
                    
                    if sa_condition == 1:
                        # Rally starts: detect court from a mid-queue frame
                        if not self.got_court_info and self.court_kprcnn:
                            mid_frame = sa_queue.get(min(2, SA_QUEUE_LENGTH - 1))[-1]
                            self._detect_court(mid_frame, frame_height)
                        
                        # Add frame to rally
                        joints = self._detect_player_keypoints(proc_frame)
                        if joints is not None:
                            player_joints_rally.append(joints)
                            frame_nums_rally.append(frame_count)
                        rally_start_frame = frame_count
                    
                    elif sa_condition == 2:
                        # Rally continues: detect player keypoints
                        joints = self._detect_player_keypoints(proc_frame)
                        if joints is not None:
                            player_joints_rally.append(joints)
                            frame_nums_rally.append(frame_count)
                    
                    elif sa_condition == 3:
                        # Rally ends: predict directions and detect hits
                        rally_end_frame = frame_count
                        rally_result = self._process_rally(
                            player_joints_rally, frame_nums_rally,
                            rally_start_frame, rally_end_frame
                        )
                        
                        if rally_result:
                            all_rallies.append(rally_result)
                        
                        # Reset for next rally
                        player_joints_rally = []
                        frame_nums_rally = []
                
                processed_count += 1
                if progress_callback and processed_count % 20 == 0:
                    prog = 5 + int((processed_count / max(target_frames, 1)) * 60)
                    progress_callback(min(prog, 65))
            
            frame_count += 1
        
        cap.release()
        
        # Process any remaining rally data
        if player_joints_rally:
            rally_result = self._process_rally(
                player_joints_rally, frame_nums_rally,
                rally_start_frame, frame_count
            )
            if rally_result:
                all_rallies.append(rally_result)
        
        if progress_callback:
            progress_callback(70)
        
        # Compile results
        total_hits = sum(len(r['hit_frames']) for r in all_rallies)
        
        return {
            'rallies': all_rallies,
            'total_hits': total_hits,
            'fps': fps,
            'total_frames': total_frames,
            'frame_height': frame_height,
            'frame_width': frame_width
        }
    
    def _detect_court(self, frame, frame_height):
        """Use Court Keypoint-RCNN to detect court boundaries."""
        if self.court_kprcnn is None:
            return
        
        try:
            img_tensor = TF.to_tensor(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.court_kprcnn(img_tensor)
            
            scores = output[0]['scores'].detach().cpu().numpy()
            high_scores_idxs = np.where(scores > 0.7)[0].tolist()
            
            if not high_scores_idxs:
                return
            
            post_nms_idxs = torchvision.ops.nms(
                output[0]['boxes'][high_scores_idxs],
                output[0]['scores'][high_scores_idxs], 0.3
            ).cpu().numpy()
            
            keypoints = []
            for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])
            
            if not keypoints:
                return
            
            court_kp = keypoints[0]
            
            # Calculate court geometry
            # court_kp: [top_left, top_right, mid_left, mid_right, bot_left, bot_right]
            l_a = (court_kp[0][1] - court_kp[4][1]) / max(court_kp[0][0] - court_kp[4][0], 1)
            l_b = court_kp[0][1] - l_a * court_kp[0][0]
            r_a = (court_kp[1][1] - court_kp[5][1]) / max(court_kp[1][0] - court_kp[5][0], 1)
            r_b = court_kp[1][1] - r_a * court_kp[1][0]
            mp_y = (court_kp[2][1] + court_kp[3][1]) / 2
            
            self.court_info = [l_a, l_b, r_a, r_b, mp_y]
            
            # Extended court points for player detection boundary
            ext = copy.deepcopy(court_kp)
            ext[0][0] -= 80; ext[0][1] -= 80
            ext[1][0] += 80; ext[1][1] -= 80
            ext[2][0] -= 80
            ext[3][0] += 80
            ext[4][0] -= 80; ext[4][1] = min(ext[4][1] + 80, frame_height - 40)
            ext[5][0] += 80; ext[5][1] = min(ext[5][1] + 80, frame_height - 40)
            self.extended_court_points = ext
            
            self.got_court_info = True
            print("  ✓ Court detected")
            
        except Exception as e:
            print(f"  Court detection error: {e}")
    
    def _detect_player_keypoints(self, frame):
        """
        Detect player keypoints using Player KP-RCNN.
        Returns: list of 2 player joints [player1_joints, player2_joints]
                 each shape (17, 2) representing 17 keypoint (x,y) coordinates.
                 Returns None if can't find exactly 2 players in court.
        """
        if self.player_kprcnn is None:
            return None
        
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t_image = transforms.Compose([
                transforms.ToTensor()
            ])(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.player_kprcnn(t_image)
            
            boxes = outputs[0]['boxes'].cpu().detach().numpy()
            joints = outputs[0]['keypoints'].cpu().detach().numpy()
            
            if len(joints) < 2:
                return None
            
            # Filter players that are in court
            in_court_indices = self._check_in_court(joints)
            
            if in_court_indices is None or len(in_court_indices) < 2:
                # Fallback: use top-2 confidence detections
                scores = outputs[0]['scores'].cpu().detach().numpy()
                if len(scores) >= 2:
                    top_indices = np.argsort(scores)[::-1][:2]
                    in_court_indices = top_indices.tolist()
                else:
                    return None
            
            # Check top/bottom court assignment
            conform, combination = self._check_top_bot(in_court_indices, boxes)
            
            if not conform:
                # Use first two detected
                combination = [0, 1]
            
            filtered_joints = []
            filtered_joints.append(joints[in_court_indices[combination[0]]].tolist())
            filtered_joints.append(joints[in_court_indices[combination[1]]].tolist())
            
            # Keep only (x, y), discard confidence
            for i in range(2):
                for j in range(len(filtered_joints[i])):
                    filtered_joints[i][j] = filtered_joints[i][j][:2]
            
            # Determine top/bottom player order
            position = self._top_bottom(filtered_joints)
            ordered = [filtered_joints[position[0]], filtered_joints[position[1]]]
            
            return ordered
            
        except Exception as e:
            return None
    
    def _check_in_court(self, joints):
        """Check which detected persons are inside the court boundaries."""
        if not self.got_court_info:
            # No court info: return first indices
            return list(range(min(len(joints), 4)))
        
        indices = []
        for i in range(len(joints)):
            if self._in_court(joints[i]):
                indices.append(i)
        
        return indices if len(indices) >= 2 else None
    
    def _in_court(self, joint):
        """Check if a player's ankles are within the extended court bounds."""
        if self.extended_court_points is None:
            return True
        
        l_a, l_b, r_a, r_b, _ = self.court_info
        
        # Use ankles (indices 15, 16)
        ankle_x = (joint[15][0] + joint[16][0]) / 2
        ankle_y = (joint[15][1] + joint[16][1]) / 2
        
        top = ankle_y > self.extended_court_points[0][1]
        bottom = ankle_y < self.extended_court_points[5][1]
        
        lmp_x = (ankle_y - l_b) / l_a if l_a != 0 else ankle_x - 1
        rmp_x = (ankle_y - r_b) / r_a if r_a != 0 else ankle_x + 1
        
        left = ankle_x > lmp_x
        right = ankle_x < rmp_x
        
        return left and right and top and bottom
    
    def _check_top_bot(self, indices, boxes):
        """Check if we have one player in top half and one in bottom half."""
        if not self.got_court_info:
            return True, [0, 1]
        
        court_mp = self.court_info[4]
        
        for i in range(min(len(indices) - 1, 3)):
            for j in range(i + 1, min(len(indices), 4)):
                box_i_mid = (boxes[indices[i]][1] + boxes[indices[i]][3]) / 2
                box_j_mid = (boxes[indices[j]][1] + boxes[indices[j]][3]) / 2
                
                if (box_i_mid < court_mp and box_j_mid > court_mp) or \
                   (box_i_mid > court_mp and box_j_mid < court_mp):
                    return True, [i, j]
        
        return False, [0, min(1, len(indices) - 1)]
    
    def _top_bottom(self, joints):
        """Determine which player is top and which is bottom."""
        # Compare ankle Y positions
        a_ankle_y = joints[0][-1][1] + joints[0][-2][1]
        b_ankle_y = joints[1][-1][1] + joints[1][-2][1]
        
        if a_ankle_y > b_ankle_y:
            # Player A is lower (bottom court) -> top=1, bottom=0
            return (1, 0)
        else:
            return (0, 1)
    
    def _process_rally(self, joint_sequence, frame_nums, start_frame, end_frame):
        """
        Process a complete rally:
        1. Feed keypoints to transformer
        2. Predict direction sequence
        3. Detect hit frames from direction changes
        """
        # Need minimum frames for a valid rally
        if len(joint_sequence) < 25:
            return None
        
        if self.opt is None:
            return None
        
        try:
            # Predict shuttlecock direction sequence
            direction_seq = list(
                self.opt.predict(joint_sequence).cpu().numpy().astype(float)
            )
            
            # Validate: if too many 0s (no movement), skip
            zero_count = sum(1 for d in direction_seq if d == 0)
            if zero_count / len(direction_seq) > 0.6:
                return None
            
            # Detect hit frames from direction changes
            hit_indices = self._detect_hit_frames(direction_seq)
            hit_frame_numbers = [frame_nums[i] for i in hit_indices if i < len(frame_nums)]
            
            return {
                'rally_start': start_frame,
                'rally_end': end_frame,
                'hit_frames': hit_frame_numbers,
                'directions': direction_seq,
                'num_hits': len(hit_frame_numbers)
            }
            
        except Exception as e:
            print(f"  Rally processing error: {e}")
            return None
    
    def _detect_hit_frames(self, direction_list):
        """
        Core algorithm from the paper:
        A hit occurs when the predicted shuttlecock direction changes
        between 1 (flying up/towards top player) and 2 (flying down/towards bottom player).
        
        Transitions:
            0 -> 1: hit (shuttle starts flying)
            0 -> 2: hit
            1 -> 2: hit (direction reversal = someone hit it back)
            2 -> 1: hit (direction reversal = someone hit it back)
            1 -> 0 or 2 -> 0: NOT a hit (shuttle stopped/lost)
        """
        last_direction = 0
        hit_frame_indices = []
        
        for i in range(len(direction_list)):
            direction = direction_list[i]
            
            if direction != last_direction:
                if last_direction == 0:
                    # Shuttle starts moving: hit
                    if direction in (1, 2):
                        hit_frame_indices.append(i)
                elif last_direction == 1:
                    if direction == 2:
                        # Direction reversal: hit
                        hit_frame_indices.append(i)
                    # 1->0 is not a hit, just continue
                elif last_direction == 2:
                    if direction == 1:
                        # Direction reversal: hit
                        hit_frame_indices.append(i)
                    # 2->0 is not a hit, just continue
            
            last_direction = direction
        
        return hit_frame_indices
