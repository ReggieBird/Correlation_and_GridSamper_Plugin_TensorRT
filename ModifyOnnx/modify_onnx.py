import os

import torch
import torch.onnx.symbolic_opset11 as sym_opset
import torch.onnx.symbolic_helper as sym_help
from torch.onnx import register_custom_op_symbolic
import onnx_graphsurgeon as gs
import onnx

import numpy as np

def grid_sampler(g, input, grid, mode, padding_mode, align_corners):  # long, long, long: contants dtype
    mode_i = sym_help._maybe_get_scalar(mode)
    paddingmode_i = sym_help._maybe_get_scalar(padding_mode)
    aligncorners_i = sym_help._maybe_get_scalar(align_corners)

    return g.op("GridSampler", input, grid, interpolationmode_i=mode_i, paddingmode_i=paddingmode_i,
                aligncorners_i=aligncorners_i)  # just a dummy definition for onnx runtime since we don't need onnx inference


sym_opset.grid_sampler = grid_sampler
register_custom_op_symbolic("::grid_sampler", grid_sampler, 11)


def Correlation(g, input1, input2, kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH,
                dilation_patchW, dH, dW):
    patchSize_i = sym_help._maybe_get_scalar(patchH)
    dilation_i = sym_help._maybe_get_scalar(dilation_patchH)
    return g.op('Correlation', input1, input2, patchsize_i=patchSize_i, dilation_i=dilation_i)


sym_opset.Correlation = Correlation
register_custom_op_symbolic("mynamespace::correlation", Correlation, 11)

script_root = os.path.dirname(__file__)
from glob import glob

ops_so = glob(os.path.join(script_root, 'correlation_pytorch/build/*/*.so'))[0]
torch.ops.load_library(ops_so)


def correlation(input1, input2):
    out = torch.ops.mynamespace.correlation(
        input1, input2, \
        1, 1, \
        9, 9, \
        0, 0, \
        0, 0, \
        1, 1, \
        1, 1
    ) / 81.
    return out


def modify_onnx(onnx_model_file):
    graph = gs.import_onnx(onnx.load(onnx_model_file))
    assert (graph is not None)

    for node in graph.nodes:
        if node.op == 'Correlation':  # 修改correlation
            patchSize = node.attrs['patchsize']
            dilation = node.attrs['dilation']

            buffer = np.array([patchSize ** 2, patchSize, patchSize, dilation], dtype=np.int32).tobytes('C')

            node.attrs = {'name': 'Correlation_TRT', 'version': '1', 'namespace': '', 'data': buffer}
            node.op = 'TRT_PluginV2'

        elif node.op == 'GridSampler':  # 修改grid_sample
            align_corners = node.attrs['aligncorners']
            inter_mode = node.attrs['interpolationmode']
            pad_mode = node.attrs['paddingmode']

            buffer = np.array([inter_mode, pad_mode], dtype=np.int32).tobytes('C') + \
                     np.array([align_corners], dtype=np.bool).tobytes('C')

            node.attrs = {'name': 'GridSampler_TRT', 'version': '1', 'namespace': "", 'data': buffer}
            node.op = 'TRT_PluginV2'

    onnx.save(gs.export_onnx(graph), onnx_model_file)