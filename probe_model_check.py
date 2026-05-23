import os,sys,traceback
p = 'yolo_model.pt'
print('cwd:', os.getcwd())
print('path:', p)
try:
    print('size:', os.path.getsize(p))
except Exception as e:
    print('os.path.getsize error:', e)
try:
    import torch
    print('torch version:', getattr(torch, '__version__', 'unknown'))
    try:
        ckpt = torch.load(p, map_location='cpu')
        print('torch.load succeeded, type:', type(ckpt))
    except Exception as e:
        print('torch.load exception:', type(e).__name__, e)
        traceback.print_exc()
except Exception as e:
    print('import torch failed:', e)
    traceback.print_exc()
