import torch
import torch.onnx as onnx
import torchvision.models as models

from yolov7 import create_yolov7_model


# 1. Load the model architecture (replace with your actual model)
model = create_yolov7_model(architecture="yolov7-tiny", num_classes=7, pretrained=False)

# 2. Load the state_dict (replace 'model_state.pth' with your path)
loaded = torch.load('/home/tvdev/Yolov7-training/examples/best_model.pt')
model.load_state_dict(loaded['model_state_dict'])

# 3. Set to evaluation mode
model.eval()

# 4. Create a dummy input
dummy_input = torch.randn(1, 3, 640, 640) # Example for ResNet18

# 5. Export to ONNX
onnx_path = "model.onnx"
torch.onnx.export(model, 
                  dummy_input, 
                  onnx_path, 
                  verbose=True, 
                  input_names=['input'], 
                  output_names=['output'],
                  opset_version=11) # Recommended to specify opset_version

print(f"Model exported to {onnx_path}")