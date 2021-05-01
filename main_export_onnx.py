import torch
import torch.nn as nn
import argparse
import cv2
import copy
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size
from models.common import Conv, Contract
from utils.activations import Hardswish, SiLU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
save = [4, 6, 10, 14, 17, 20, 23]
anchors = [[4,5,  8,10,  13,16], [23,29,  43,55,  73,105], [146,217,  231,300,  335,433]]
stride = [8, 16, 32]
nc = 1
na = len(anchors[0]) // 2
no = nc + 5 + 10
nl = len(anchors)

def test_export(opt):
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    # Load model
    img_size = 2048
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(opt.image)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + opt.image
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    x = copy.deepcopy(img)
    onnxmodel = model.model
    y = []
    for m in onnxmodel:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)  # run
        y.append(x if m.i in save else None)  # save output
    print(torch.equal(x[0], pred))
    return onnxmodel, img

class my_yolov5_model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.contract = Contract(gain=2)
        self.len_model = len(model)
        self.na = 3
        self.no = 16
    def forward(self, x):
        x = self.contract(x)
        x = self.model[0].conv(x)
        y = [None]
        for i in range(1, self.len_model):
            if self.model[i].f != -1:  # if not from previous layer
                x = y[self.model[i].f] if isinstance(self.model[i].f, int) else [x if j == -1 else y[j] for j in self.model[i].f]
            x = self.model[i](x)  # run
            y.append(x if self.model[i].i in save else None)

        ny, nx = x[0].shape[2:]
        x[0] = x[0].view(self.na, self.no, ny, nx).permute(0, 2, 3, 1).contiguous()
        x[0] = x[0].view(-1, self.no)
        ny, nx = x[1].shape[2:]
        x[1] = x[1].view(self.na, self.no, ny, nx).permute(0, 2, 3, 1).contiguous()
        x[1] = x[1].view(-1, self.no)
        ny, nx = x[2].shape[2:]
        x[2] = x[2].view(self.na, self.no, ny, nx).permute(0, 2, 3, 1).contiguous()
        x[2] = x[2].view(-1, self.no)
        return torch.cat(x, 0)
        # return x

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s-face.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='data/images/test.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=2048, help='inference size (pixels)')
    opt = parser.parse_args()
    onnxmodel,img = test_export(opt)

    onnxmodel[-1].export = True
    net = my_yolov5_model(onnxmodel).to(device)
    net.eval()
    # with torch.no_grad():
    #     out = net(img)
    # print(out)

    f = opt.weights.replace('.pt', '2.onnx')  # filename
    input = torch.zeros(1, 3, 2048, 2048).to(device)
    # Update model
    for k, m in net.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    torch.onnx.export(net, input, f, verbose=False, opset_version=12, input_names=['data'], output_names=['out'])

    cvnet = cv2.dnn.readNet(f)
    input = cv2.imread(opt.image)
    input = cv2.resize(input, (2048,2048))
    blob = cv2.dnn.blobFromImage(input)
    cvnet.setInput(blob)
    outs = cvnet.forward(cvnet.getUnconnectedOutLayersNames())