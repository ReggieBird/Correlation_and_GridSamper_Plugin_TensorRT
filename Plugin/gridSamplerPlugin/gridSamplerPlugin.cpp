/*
Created by Reggie Bird on 2023/07/23
Reference from https://github.com/ltkong218/FastFlowNet/blob/main/tensorrt_workspace/TensorRT/plugin
 */
 
#include "gridSamplerPlugin.h"
#include <cstring>
#include <iostream>
#include <sstream>

using namespace nvinfer1;
using nvinfer1::plugin::GridSamplerPlugin;
using nvinfer1::plugin::GridSamplerPluginCreator;

namespace
{
char const* const kGRID_SAMPLER_PLUGIN_VERSION{"1"};
char const* const kGRID_SAMPLER_PLUGIN_NAME{"GridSampler_TRT"};
} // namespace

PluginFieldCollection GridSamplerPluginCreator::mFC{};
std::vector<PluginField> GridSamplerPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GridSamplerPluginCreator);

GridSamplerPlugin::GridSamplerPlugin(const std::string name, GridSamplerInterpolation interpolationMode,
    GridSamplerPadding paddingMode, bool alignCorners)
    : mLayerName(name)
    , mInterpolationMode(interpolationMode)
    , mPaddingMode(paddingMode)
    , mAlignCorners(alignCorners)
{
}

GridSamplerPlugin::GridSamplerPlugin(const std::string name, void const* buffer, size_t length)
    : mLayerName(name)
{
    char const* d = static_cast<char const*>(buffer);
    char const* a = d;
    mInterpolationMode = read<GridSamplerInterpolation>(d);
    mPaddingMode = read<GridSamplerPadding>(d);
    mAlignCorners = read<bool>(d);
    PLUGIN_ASSERT(d == a + sizeof(GridSamplerInterpolation) + sizeof(GridSamplerPadding) + sizeof(bool));
}

int32_t GridSamplerPlugin::getNbOutputs() const noexcept
{
    // Plugin layer has 1 output
    return 1;
}

DimsExprs GridSamplerPlugin::getOutputDimensions(int32_t index, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(inputs[0].nbDims == 4);
    PLUGIN_ASSERT(inputs[1].nbDims == 4);

    DimsExprs output(inputs[0]);
    output.d[2] = inputs[1].d[1];
    output.d[3] = inputs[1].d[2];
    return output;
}

int32_t GridSamplerPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void GridSamplerPlugin::terminate() noexcept {}

size_t GridSamplerPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t GridSamplerPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    GridSamplerDataType dataType = (inputDesc->type == DataType::kFLOAT ? GridSamplerDataType::GFLOAT : GridSamplerDataType::GHALF);

    nvinfer1::Dims inputDims = inputDesc[0].dims;
    int32_t batchSize = inputDims.d[0];
    int32_t inChannels = inputDims.d[1];
    int32_t inHeight = inputDims.d[2];
    int32_t inWidth = inputDims.d[3];

    nvinfer1::Dims gridDims = inputDesc[1].dims;
    int32_t gridHeight = gridDims.d[1];
    int32_t gridWidth = gridDims.d[2];

    return grid_sampler_2d_cuda(batchSize, inputs[0], inputs[1], outputs[0],
        inChannels, inHeight, inWidth, gridHeight, gridWidth,
        inChannels*inHeight*inWidth, inHeight*inWidth, inWidth, 1,
        gridHeight*gridWidth*2, gridWidth*2, 2, 1,
        inChannels*gridHeight*gridWidth, gridHeight*gridWidth, gridWidth, 1,
        mInterpolationMode, mPaddingMode, mAlignCorners, dataType, stream);
}

size_t GridSamplerPlugin::getSerializationSize() const noexcept
{
    return sizeof(GridSamplerInterpolation) + sizeof(GridSamplerPadding) + sizeof(bool);
}

void GridSamplerPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write<GridSamplerInterpolation>(d, mInterpolationMode);
    write<GridSamplerPadding>(d, mPaddingMode);
    write<bool>(d, mAlignCorners);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

bool GridSamplerPlugin::supportsFormatCombination(
    int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);

    bool condition = inOut[pos].format == PluginFormat::kLINEAR;

    condition &= inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF;
    condition &= inOut[pos].type == inOut[0].type;
    return condition;
}

// Set plugin namespace
void GridSamplerPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

char const* GridSamplerPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
DataType GridSamplerPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// Configure the layer with input and output data types.
void GridSamplerPlugin::configurePlugin(const DynamicPluginTensorDesc* inputs, int32_t nbInputs,
    const DynamicPluginTensorDesc* outputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);

    // we only support 2d grid sampler now.
    PLUGIN_ASSERT(inputs[0].desc.dims.nbDims == 4);
    PLUGIN_ASSERT(inputs[1].desc.dims.nbDims == 4);

    PLUGIN_ASSERT(inputs[0].desc.dims.d[0] == inputs[1].desc.dims.d[0]);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void GridSamplerPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void GridSamplerPlugin::detachFromContext() noexcept {}

char const* GridSamplerPlugin::getPluginType() const noexcept
{
    return kGRID_SAMPLER_PLUGIN_NAME;
}

char const* GridSamplerPlugin::getPluginVersion() const noexcept
{
    return kGRID_SAMPLER_PLUGIN_VERSION;
}

void GridSamplerPlugin::destroy() noexcept
{
    delete this;
}

// Clone the plugin
IPluginV2DynamicExt* GridSamplerPlugin::clone() const noexcept
{
    try
    {
        // Create a new instance
        IPluginV2DynamicExt* plugin
            = new GridSamplerPlugin(mLayerName, mInterpolationMode, mPaddingMode, mAlignCorners);

        // Set the namespace
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

GridSamplerPluginCreator::GridSamplerPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GridSamplerPluginCreator::getPluginName() const noexcept
{
    return kGRID_SAMPLER_PLUGIN_NAME;
}

char const* GridSamplerPluginCreator::getPluginVersion() const noexcept
{
    return kGRID_SAMPLER_PLUGIN_VERSION;
}

PluginFieldCollection const* GridSamplerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* GridSamplerPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PluginField const* fields = fc->fields;
        int32_t interpolationMode = 0, paddingMode = 0, alignCorners = 0;

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "interpolation_mode"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                interpolationMode = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "padding_mode"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                paddingMode = *(static_cast<int32_t const*>(fields[i].data));
            }
            else if (!strcmp(attrName, "align_corners"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                alignCorners = *(static_cast<int32_t const*>(fields[i].data));
            }
        }

        GridSamplerPlugin* obj = new GridSamplerPlugin(name, static_cast<GridSamplerInterpolation>(interpolationMode)
            , static_cast<GridSamplerPadding>(paddingMode), static_cast<bool>(alignCorners));
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* GridSamplerPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call GridSamplerPlugin::destroy()
        GridSamplerPlugin* obj = new GridSamplerPlugin(name, serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
