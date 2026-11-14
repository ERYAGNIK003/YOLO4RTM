import train
train.run(data="dataset/DRASHTI-HaOBB-yolo5.yaml",
          hyp="data/hyps/obb/hyp.finetune_dota_HaAug_scratch.yaml",
          imgsz=1024, epochs=100, batch_size=32, device=0,
          weights='weights/yolov5n.pt', name="HPC_test")

torch.cuda.empty_cache()
