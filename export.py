from model import GenderAge
import torch.nn as nn
import torch


MODEL_PATH = 'models-2020-11-20-14-37/best-epoch47-0.9314.pt'


def reload_model(model):
    own_state = model.state_dict()
    state_dict = torch.load(MODEL_PATH)
    for name, param in state_dict.items():
        if name not in own_state:
            print('layer {} skip, not exist'.format(name))
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        if own_state[name].shape != param.shape:
            print('layer {} skip, shape not same'.format(name))
            continue
        own_state[name].copy_(param)


if __name__ == '__main__':
    try:
        torch_model = torch.load(MODEL_PATH)
        torch_model = torch_model.cpu()
        torch_model.eval()
    except AttributeError:
        torch_model = GenderAge(3, [2, 2, 2, 2])
        reload_model(torch_model)
        torch_model = torch_model.cpu()
        torch_model.eval()

    x = torch.randn(1, 3, 64, 64)
    export_onnx_file = MODEL_PATH.replace('.pt', '.onnx')
    torch.onnx.export(torch_model, x, export_onnx_file,
                      opset_version=12,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
