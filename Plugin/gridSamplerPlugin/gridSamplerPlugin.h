/*
Created by Reggie Bird on 2023/07/23
Reference from https://github.com/ltkong218/FastFlowNet/blob/main/tensorrt_workspace/TensorRT/plugin
 */
 
#ifndef TRT_GRID_SAMPLER_PLUGIN_H
#define TRT_GRID_SAMPLER_PLUGIN_H

#include "gridSampler.h"
#include "common/plugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

using torch::detail::GridSamplerInterpolation;
using torch::detail::GridSamplerPadding;
using torch::detail::GridSamplerDataType;

class GridSamplerPlugin : public IPluginV2DynamicExt
{
public:
    GridSamplerPlugin(const std::string name, GridSamplerInterpolation interpolationMode, GridSamplerPadding paddingMode, bool alignCorners);

    GridSamplerPlugin(const std::string name, void const* buffer, size_t length);

    ~GridSamplerPlugin() override = default;

    int32_t getNbOutputs() const noexcept override;

    DimsExprs getOutputDimensions(int32_t index, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs,
        const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(const DynamicPluginTensorDesc* inputs, int32_t nbInputs,
        const DynamicPluginTensorDesc* outputs, int32_t nbOutputs) noexcept override;

    void detachFromContext() noexcept override;

private:
    std::string mPluginNamespace;

    const std::string mLayerName;
    GridSamplerInterpolation mInterpolationMode;
    GridSamplerPadding mPaddingMode;
    bool mAlignCorners;
};

class GridSamplerPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    GridSamplerPluginCreator();

    ~GridSamplerPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GRID_SAMPLER_PLUGIN_H
