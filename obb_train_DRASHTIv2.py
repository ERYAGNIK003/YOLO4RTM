from ultralytics import YOLO
import torch
import os
#"yolo11n-obb.yaml","yolo11s-obb.yaml","yolo11m-obb.yaml",
modelConfigs=["yolo11l-obb.yaml","yolo11x-obb.yaml"]
torch.cuda.empty_cache()
for model_config in modelConfigs:
    print(f"Working on {model_config} model")
    modelName=os.path.splitext(model_config)[0]
    # Create a new YOLO11n-OBB model from scratch
    model = YOLO(model_config)
    # Train the model on the DOTAv1 dataset
    results = model.train(data="DRASHTIv2.yaml", epochs=30,imgsz=1024,device=0,workers=8,
                          batch=16,multi_scale=True,save_period=10,plots=True,resume=True,cos_lr=True,augment=True,
                          degrees=180,scale=0.5,mixup=0.1,val=True,pretrained=False,mosaic=0.3,nms=True,iou=0.6,deterministic=True,
                          project='YOLO11-obb_4k', name=modelName)
    
    del model
    del results
torch.cuda.empty_cache()
