import os
import time
import pdb
import cv2
import numpy as np
import pdb

import grpc
from tritongrpcclient import grpc_service_pb2
from tritongrpcclient import grpc_service_pb2_grpc
import BVT
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class GeneratorRpcClient:
    def __init__(self, address, model_name, model_version=1):

        
        self.model_name = model_name
        self.model_version = str(model_version)
        # pdb.set_trace()
        MAX_MESSAGE_LENGTH = 1024*1024*1024
        options = [
               ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
               ]
        channel = grpc.insecure_channel(address,options=options)
        self.stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(channel)
        meta_req = grpc_service_pb2.ModelMetadataRequest(name=model_name, version="1")
        self.meta = self.stub.ModelMetadata(meta_req)

    def infer(self, img,batch_size=1):
        
        req = grpc_service_pb2.ModelInferRequest()
        req.model_name = self.model_name
        input0 = grpc_service_pb2.ModelInferRequest().InferInputTensor()
        input0.name = self.meta.inputs[0].name
        input0.datatype = self.meta.inputs[0].datatype
        input0.shape.extend([batch_size] + self.meta.inputs[0].shape[1:])
        input0_content = grpc_service_pb2.InferTensorContents()
        # pdb.set_trace()
        input0_content.raw_contents = img.tobytes()
        input0.contents.CopyFrom(input0_content)


        
        req.inputs.extend([input0])
        output1 = grpc_service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        output1.name = self.meta.outputs[0].name

        req.outputs.extend([output1])
        # pdb.set_trace()

        resp = self.stub.ModelInfer(req)
        pred1 = resp.outputs[0].contents.raw_contents
        pred_shape1 = resp.outputs[0].shape
        
        res1 = np.frombuffer(pred1, np.float32).reshape(*pred_shape1)

        
        return res1


if __name__ == '__main__':

    bs = 1
    
    # pdb.set_trace()
    input_img = np.random.randn(bs,1,3,64,64).astype(np.float32)
  
    
    client = GeneratorRpcClient("127.0.0.1:8079", "selfop")
    for _ in range(100):
        t1 = time.time()   
        res1 = client.infer(input_img,batch_size=bs)
        pdb.set_trace()
        t2 = time.time()
        
    
        print('average cost: {}s'.format((t2-t1)))
    
   
    



