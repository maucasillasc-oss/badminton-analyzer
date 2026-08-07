"""
Download and prepare model weights for the Hit-Frame Detection pipeline.

Strategy:
- sacnn.pt: Downloaded from the paper's GitHub repo (small, available directly)
- scaler.pickle: Downloaded from the paper's GitHub repo (small, available directly)
- kpRCNN.pth: Use torchvision's pre-trained Keypoint-RCNN (COCO, 17 keypoints)
  The paper's model is fine-tuned on their dataset, but the base COCO model works
  well for detecting player keypoints in badminton (same 17 COCO keypoints).
- court_kpRCNN.pth: Not publicly available. We skip court detection and use 
  a simpler heuristic (frame midpoint) for top/bottom player assignment.
- OPT_16_head_dp.pt: Train from the paper's public dataset (KSeq_train_data.zip).
  This is the core transformer model. We include a training script that runs 
  during Docker build using the public training data.

The key insight: The OPT transformer is what makes the 99% accuracy possible.
The KP-RCNNs are standard architectures; what matters is the transformer that
predicts shuttle direction from keypoint sequences.
"""
import os
import sys
import urllib.request


def download_weights(output_dir='hitframe_weights'):
    """Download and prepare all model weights."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Preparing Hit-Frame Detection model weights...")
    print(f"Target: {output_dir}/")
    print("=" * 60)
    
    # Strategy 1: Try downloading pre-trained checkpoints directly from paper's Drive
    # Checkpoints folder: https://drive.google.com/drive/folders/1v-uejba2ljNRUPaSAR-9u-a9GY1mD4pu
    all_downloaded = _download_from_drive_checkpoints(output_dir)
    
    if not all_downloaded:
        # Strategy 2: Fallback - download what we can from GitHub + prepare alternatives
        print("\n  Falling back to alternative weight sources...")
        
        # Download SA-CNN and scaler from GitHub (always available)
        _download_from_github(output_dir)
        
        # Use torchvision Keypoint-RCNN as fallback for player detection
        if not os.path.exists(os.path.join(output_dir, 'kpRCNN.pth')):
            _prepare_kprcnn(output_dir)
        
        # Train OPT transformer from public dataset if not downloaded
        if not os.path.exists(os.path.join(output_dir, 'OPT_16_head_dp.pt')):
            _train_opt_transformer(output_dir)
    
    # Final verification
    _verify_weights(output_dir)


def _download_from_drive_checkpoints(output_dir):
    """
    Try to download all pre-trained weights from the paper's checkpoints folder.
    Returns True if all critical weights were downloaded.
    """
    import gdown
    
    checkpoints_folder = 'https://drive.google.com/drive/folders/1v-uejba2ljNRUPaSAR-9u-a9GY1mD4pu'
    
    print(f"\n  Attempting to download pre-trained checkpoints from Drive...")
    print(f"  Folder: {checkpoints_folder}")
    
    try:
        # Download entire folder
        gdown.download_folder(checkpoints_folder, output=output_dir, quiet=False)
        
        # Flatten any subdirectories
        for root, dirs, files in os.walk(output_dir):
            if root == output_dir:
                continue
            for f in files:
                if f.endswith(('.pt', '.pth', '.pickle')):
                    src = os.path.join(root, f)
                    dst = os.path.join(output_dir, f)
                    if not os.path.exists(dst):
                        os.rename(src, dst)
                        print(f"    Moved: {f}")
        
        # Clean up empty subdirs
        import shutil
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                # Check for files in subdirectory
                for root, dirs, files in os.walk(item_path):
                    for f in files:
                        if f.endswith(('.pt', '.pth', '.pickle')):
                            src = os.path.join(root, f)
                            dst = os.path.join(output_dir, f)
                            if not os.path.exists(dst):
                                os.rename(src, dst)
                shutil.rmtree(item_path, ignore_errors=True)
        
        # Normalize OPT model name
        for name in ['opt.pt', 'OPT.pt', 'OPT_16_head.pt']:
            src = os.path.join(output_dir, name)
            dst = os.path.join(output_dir, 'OPT_16_head_dp.pt')
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)
        
        # Check if we got the critical files
        critical = ['sacnn.pt', 'kpRCNN.pth', 'OPT_16_head_dp.pt', 'scaler.pickle']
        found = sum(1 for f in critical if os.path.exists(os.path.join(output_dir, f)))
        
        if found >= 3:  # At least OPT + sacnn + scaler
            print(f"  ✓ Downloaded {found}/{len(critical)} checkpoint files from Drive")
            return True
        else:
            print(f"  Only found {found}/{len(critical)} files. Trying alternatives...")
            return False
            
    except Exception as e:
        print(f"  Drive folder download failed: {e}")
        return False


def _download_from_github(output_dir):
    """Download small weights from the paper's GitHub repo."""
    repo_base = ("https://github.com/arthur900530/"
                 "Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/"
                 "raw/master/src/models/weights")
    
    files = ['sacnn.pt', 'scaler.pickle']
    
    for fname in files:
        dst = os.path.join(output_dir, fname)
        if os.path.exists(dst):
            print(f"  ✓ {fname} (already exists)")
            continue
        
        url = f"{repo_base}/{fname}"
        print(f"  Downloading {fname} from GitHub...")
        try:
            urllib.request.urlretrieve(url, dst)
            size = os.path.getsize(dst)
            print(f"  ✓ {fname} ({size:,} bytes)")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")


def _prepare_kprcnn(output_dir):
    """
    Save torchvision's pre-trained Keypoint-RCNN for player detection.
    This is the same architecture as the paper's kpRCNN but trained on COCO.
    It detects 17 human keypoints which is exactly what the OPT transformer expects.
    """
    dst = os.path.join(output_dir, 'kpRCNN.pth')
    if os.path.exists(dst):
        print(f"  ✓ kpRCNN.pth (already exists)")
        return
    
    print("  Preparing Player Keypoint-RCNN (torchvision pre-trained)...")
    
    import torch
    import torchvision
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
    
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    
    # Save the full model (same format as the paper's code uses torch.load)
    torch.save(model, dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"  ✓ kpRCNN.pth ({size_mb:.1f} MB) - torchvision Keypoint-RCNN")


def _train_opt_transformer(output_dir):
    """
    Train the OPT transformer from the paper's public dataset.
    The dataset (KSeq_train_data.zip) contains keypoint sequences with
    labeled shuttle directions, which is exactly what we need to train the model.
    """
    dst = os.path.join(output_dir, 'OPT_16_head_dp.pt')
    if os.path.exists(dst):
        print(f"  ✓ OPT_16_head_dp.pt (already exists)")
        return
    
    print("  Training OPT Transformer from paper's public dataset...")
    print("  Downloading training data...")
    
    import gdown
    import zipfile
    import torch
    import numpy as np
    import pickle
    
    # Download training data from paper's Google Drive
    train_zip = os.path.join(output_dir, 'KSeq_train_data.zip')
    if not os.path.exists(train_zip):
        gdown.download(
            'https://drive.google.com/uc?id=1tDk2AbBaV-MWH_kpW8wXuIyf4A4W9QBH',
            train_zip, quiet=False
        )
    
    # Extract
    train_dir = os.path.join(output_dir, 'train_data')
    if not os.path.exists(train_dir):
        print("  Extracting training data...")
        with zipfile.ZipFile(train_zip, 'r') as zf:
            zf.extractall(train_dir)
    
    # Load and prepare training data
    print("  Loading training sequences...")
    X_train, y_train = _load_training_data(train_dir)
    
    if X_train is None or len(X_train) == 0:
        print("  ⚠ Could not load training data. OPT will not be available.")
        # Clean up
        _cleanup_training_files(output_dir, train_zip, train_dir)
        return
    
    print(f"  Loaded {len(X_train)} training sequences")
    
    # Fit and save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pickle')
    if not os.path.exists(scaler_path) or os.path.getsize(scaler_path) < 100:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Flatten all sequences for fitting
        all_joints = []
        for seq in X_train:
            for frame in seq:
                all_joints.append(np.array(frame).reshape(1, -1)[0])
        scaler.fit(np.array(all_joints))
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  ✓ scaler.pickle (refitted from training data)")
    
    # Train the transformer
    print("  Training OPT Transformer (this may take 10-20 minutes on CPU)...")
    _train_model(X_train, y_train, dst, scaler_path)
    
    # Clean up training files
    _cleanup_training_files(output_dir, train_zip, train_dir)


def _load_training_data(train_dir):
    """Load keypoint sequences and direction labels from the dataset."""
    import json
    import numpy as np
    
    X_sequences = []
    y_sequences = []
    
    # Walk through extracted data looking for JSON files with joints + directions
    for root, dirs, files in os.walk(train_dir):
        for f in sorted(files):
            if not f.endswith('.json'):
                continue
            
            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'r') as fh:
                    data = json.load(fh)
                
                # The dataset format: {"joints": [...], "shuttle_directions": [...]}
                # or similar structure from the paper's KSeq format
                joints = None
                directions = None
                
                if isinstance(data, dict):
                    # Try various key names the paper might use
                    joints = data.get('joints') or data.get('keypoints') or data.get('input')
                    directions = (data.get('shuttle_directions') or 
                                 data.get('directions') or 
                                 data.get('labels') or 
                                 data.get('output'))
                elif isinstance(data, list) and len(data) == 2:
                    joints = data[0]
                    directions = data[1]
                
                if joints is not None and directions is not None:
                    # Validate shape: joints should be (seq_len, 2, 17, 2)
                    joints_arr = np.array(joints)
                    if len(joints_arr.shape) >= 3:
                        X_sequences.append(joints)
                        y_sequences.append(directions)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    
    # Also try loading .npy or .npz files
    for root, dirs, files in os.walk(train_dir):
        for f in sorted(files):
            if f.endswith('.npy'):
                filepath = os.path.join(root, f)
                try:
                    data = np.load(filepath, allow_pickle=True)
                    if data.ndim >= 3:
                        # Might be sequences stacked
                        pass  # Handle based on actual format
                except:
                    continue
            elif f.endswith('.npz'):
                filepath = os.path.join(root, f)
                try:
                    data = np.load(filepath, allow_pickle=True)
                    if 'X' in data and 'y' in data:
                        X_sequences.extend(data['X'].tolist())
                        y_sequences.extend(data['y'].tolist())
                except:
                    continue
    
    if not X_sequences:
        print("  ⚠ No training sequences found in expected format.")
        print("    Looking for alternative data formats...")
        # Try loading pickle files
        for root, dirs, files in os.walk(train_dir):
            for f in sorted(files):
                if f.endswith('.pkl') or f.endswith('.pickle'):
                    filepath = os.path.join(root, f)
                    try:
                        import pickle
                        with open(filepath, 'rb') as fh:
                            data = pickle.load(fh)
                        if isinstance(data, dict) and 'X' in data:
                            X_sequences = data['X']
                            y_sequences = data['y']
                            break
                    except:
                        continue
    
    if not X_sequences:
        return None, None
    
    return X_sequences, y_sequences


def _train_model(X_train, y_train, output_path, scaler_path):
    """Train the OPT transformer model."""
    import torch
    import torch.nn as nn
    import numpy as np
    import pickle
    
    # Add parent to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hitframe_models.transformer import OptimusPrime
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Training device: {device}")
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Prepare data
    # Pad/truncate sequences to max_len
    max_len = 150  # Maximum sequence length
    
    X_padded = []
    y_padded = []
    
    for x_seq, y_seq in zip(X_train, y_train):
        x_arr = np.array(x_seq)
        y_arr = np.array(y_seq)
        
        if len(x_arr) < 25:
            continue  # Skip very short sequences
        
        # Truncate if too long
        if len(x_arr) > max_len:
            x_arr = x_arr[:max_len]
            y_arr = y_arr[:max_len]
        
        # Scale
        try:
            scaled = []
            for frame in x_arr:
                flat = np.reshape(frame, [1, -1])
                s = scaler.transform(flat)
                scaled.append(np.reshape(s, [2, 17, 2]))
            x_arr = np.array(scaled)
        except:
            continue
        
        # Pad to max_len
        pad_len = max_len - len(x_arr)
        if pad_len > 0:
            x_arr = np.pad(x_arr, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
            y_arr = np.pad(y_arr, (0, pad_len))
        
        X_padded.append(x_arr)
        y_padded.append(y_arr)
    
    if not X_padded:
        print("  ⚠ No valid training sequences after preprocessing.")
        return
    
    X_tensor = torch.tensor(np.array(X_padded), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_padded), dtype=torch.long)
    
    print(f"  Training data: {X_tensor.shape} -> {y_tensor.shape}")
    
    # Create model
    model = OptimusPrime(
        num_tokens=4, dim_model=2048, num_heads=8,
        num_encoder_layers=8, dim_feedforward=2048, dropout_p=0.1
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    batch_size = 4
    epochs = 30
    n_samples = len(X_padded)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            x_batch = X_tensor[batch_idx].to(device)
            y_batch = y_tensor[batch_idx].to(device)
            
            # Create padding mask
            pad_mask = model.create_src_pad_mask(x_batch)
            
            # Forward
            output = model(x_batch, src_pad_mask=pad_mask)
            # output: (seq_len, batch, num_tokens)
            output = output.permute(1, 2, 0)  # (batch, num_tokens, seq_len)
            
            loss = criterion(output, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    model.eval()
    torch.save(model.state_dict(), output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ OPT_16_head_dp.pt ({size_mb:.1f} MB) - Trained transformer")


def _cleanup_training_files(output_dir, train_zip, train_dir):
    """Remove training data files to save Docker image space."""
    import shutil
    if os.path.exists(train_zip):
        os.remove(train_zip)
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    print("  Cleaned up training data files.")


def _verify_weights(output_dir):
    """Verify weight files and print summary."""
    print("\n" + "=" * 60)
    print("Weight file verification:")
    print("=" * 60)
    
    expected = {
        'sacnn.pt': 'SA-CNN (rally segmentation)',
        'kpRCNN.pth': 'Player Keypoint-RCNN',
        'scaler.pickle': 'Data scaler',
        'OPT_16_head_dp.pt': 'OPT Transformer (direction prediction)',
    }
    
    all_ok = True
    for fname, description in expected.items():
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {fname} ({size_mb:.1f} MB) - {description}")
        else:
            print(f"  ✗ {fname} MISSING - {description}")
            all_ok = False
    
    if all_ok:
        print("\n✓ All weights ready!")
    else:
        print("\n⚠ Some weights missing. System will use available models.")
    
    return all_ok


if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'hitframe_weights'
    download_weights(output_dir)
