"""
Download trained model weights from S3.
Weights were trained from the paper's public dataset in Colab/SageMaker.
"""
import boto3
import os
import urllib.request

WEIGHTS_DIR = 'hitframe_weights'
S3_BUCKET = 'badminton-cdmx'
S3_PREFIX = 'models/'

def download_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    s3 = boto3.client(
        's3',
        region_name='us-east-2',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    
    # Download OPT Transformer weights from S3
    files = ['OPT_16_head_dp.pt', 'scaler.pickle']
    
    for fname in files:
        dst = os.path.join(WEIGHTS_DIR, fname)
        if os.path.exists(dst):
            print(f"  Already exists: {fname}")
            continue
        
        s3_key = S3_PREFIX + fname
        print(f"  Downloading {s3_key} from S3...")
        try:
            s3.download_file(S3_BUCKET, s3_key, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f"  OK: {fname} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  FAILED: {fname}: {e}")
    
    # Download SA-CNN from GitHub (small file)
    sacnn_dst = os.path.join(WEIGHTS_DIR, 'sacnn.pt')
    if not os.path.exists(sacnn_dst):
        url = ("https://github.com/arthur900530/"
               "Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/"
               "raw/master/src/models/weights/sacnn.pt")
        print(f"  Downloading sacnn.pt from GitHub...")
        try:
            urllib.request.urlretrieve(url, sacnn_dst)
            print(f"  OK: sacnn.pt")
        except Exception as e:
            print(f"  FAILED: sacnn.pt: {e}")
    
    # Download Player KP-RCNN (torchvision pre-trained)
    kprcnn_dst = os.path.join(WEIGHTS_DIR, 'kpRCNN.pth')
    if not os.path.exists(kprcnn_dst):
        print(f"  Preparing Player KP-RCNN (torchvision)...")
        try:
            import torch
            from torchvision.models.detection import keypointrcnn_resnet50_fpn
            from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
            model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
            model.eval()
            torch.save(model, kprcnn_dst)
            size_mb = os.path.getsize(kprcnn_dst) / (1024 * 1024)
            print(f"  OK: kpRCNN.pth ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  FAILED: kpRCNN.pth: {e}")
    
    # Verify
    print("\nWeight files:")
    for f in os.listdir(WEIGHTS_DIR):
        size = os.path.getsize(os.path.join(WEIGHTS_DIR, f))
        print(f"  {f}: {size/1024/1024:.1f} MB")


if __name__ == '__main__':
    download_weights()
