/*
Created by Reggie Bird on 2023/07/23
Reference from https://github.com/ltkong218/FastFlowNet/blob/main/tensorrt_workspace/TensorRT/plugin
 */

#include "correlationPlugin.h"
#include "correlation.h"
#include "common/serialize.hpp"

using namespace nvinfer1;
using nvinfer1::plugin::CorrelationPlugin;
using nvinfer1::plugin::CorrelationPluginCreator;
using torch::detail::CorrelationDataType;

namespace
{
constexpr char const* kCORRELATION_VERSION{"1"};
constexpr char const* kCORRELATION_NAME{"Correlation_TRT"};
} // namespace

// // Static class fields initialization
PluginFieldCollection CorrelationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> CorrelationPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CorrelationPluginCreator);

CorrelationPlugin::CorrelationPlugin(int32_t outputChannel, int32_t patchHeight, int32_t patchWidth, int32_t dilation)
    : mOutputChannel(outputChannel)
    , mPatchHeight(patchHeight)
    , mPatchWidth(patchWidth)
    , mDilation(dilation)
{
}

int32_t CorrelationPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

CorrelationPlugin::CorrelationPlugin(void const* data, size_t length)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mOutputChannel);
    deserialize_value(&data, &length, &mPatchHeight);
    deserialize_value(&data, &length, &mPatchWidth);
    deserialize_value(&data, &length, &mDilation);
}

char const* CorrelationPlugin::getPluginType() const noexcept
{
    return kCORRELATION_NAME;
}

char const* CorrelationPlugin::getPluginVersion() const noexcept
{
    return kCORRELATION_VERSION;
}

int32_t CorrelationPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs CorrelationPlugin::getOutputDimensions(
    int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_ASSERT(inputs[0].nbDims == 4);
        PLUGIN_ASSERT(inputs[1].nbDims == 4);

        DimsExprs output(inputs[0]);
        output.d[1] = exprBuilder.constant(mOutputChannel);
        output.d[2] = inputs[1].d[2];
        output.d[3] = inputs[1].d[3];
        return output;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return DimsExprs{};
    }
}

void CorrelationPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept {}

// Detach the plugin object from its execution context.
void CorrelationPlugin::detachFromContext() noexcept {}


int32_t CorrelationPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return STATUS_FAILURE;
    }

    nvinfer1::Dims inputDims = inputDesc[0].dims;
    int32_t batchSize = inputDims.d[0];
    int32_t inChannels = inputDims.d[1];
    int32_t inHeight = inputDims.d[2];
    int32_t inWidth = inputDims.d[3];

    auto* output = static_cast<float*>(outputs[0]);

    CorrelationDataType dataType = (inputDesc->type == DataType::kFLOAT ? CorrelationDataType::GFLOAT : CorrelationDataType::GHALF);

    return correlation_cuda(batchSize, static_cast<float const*>(inputs[0]), static_cast<float const*>(inputs[1]), output,
                            inChannels, inHeight, inWidth, inHeight, inWidth,
                            inChannels*inHeight*inWidth, inHeight*inWidth, inWidth, 1,
                            mPatchHeight*mPatchWidth*inHeight*inWidth, inHeight*inWidth, inWidth,1,
                            1, 1, mPatchHeight,mPatchWidth, int32_t((mPatchHeight-1)/2), int32_t((mPatchWidth-1)/2),
                            mDilation, dataType, stream);
}

size_t CorrelationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mOutputChannel) + sizeof(mPatchHeight) + sizeof(mPatchWidth) + sizeof(mDilation);
}

void CorrelationPlugin::serialize(void* buffer) const noexcept
{
    PLUGIN_ASSERT(buffer != nullptr);
    auto* const start = reinterpret_cast<uint8_t*>(buffer);
    serialize_value(&buffer, mOutputChannel);
    serialize_value(&buffer, mPatchHeight);
    serialize_value(&buffer, mPatchWidth);
    serialize_value(&buffer, mDilation);
    PLUGIN_ASSERT(start + getSerializationSize() == reinterpret_cast<uint8_t*>(buffer));
}

bool CorrelationPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_ASSERT(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
        PLUGIN_VALIDATE(pos >= 0);
        return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
            && inOut[pos].type == inOut[0].type);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return false;
    }
}

void CorrelationPlugin::terminate() noexcept {}

void CorrelationPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* CorrelationPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new CorrelationPlugin(mOutputChannel, mPatchHeight, mPatchWidth, mDilation);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void CorrelationPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);

        nvinfer1::Dims inputDims = in[0].desc.dims;
        int32_t const batchSize = inputDims.d[0];
        int32_t const inChannels = inputDims.d[1];
        if (batchSize <= 0 || inChannels <= 0)
        {
            // Input size not yet known, nothing to configure.
            return;
        }

        PLUGIN_ASSERT(batchSize == in[1].desc.dims.d[0]);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

nvinfer1::DataType CorrelationPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(index == 0);
        return inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return DataType{};
    }
}

size_t CorrelationPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void CorrelationPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* CorrelationPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

CorrelationPluginCreator::CorrelationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("output_channel", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("patch_height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("patch_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* CorrelationPluginCreator::getPluginName() const noexcept
{
    return kCORRELATION_NAME;
}

char const* CorrelationPluginCreator::getPluginVersion() const noexcept
{
    return kCORRELATION_VERSION;
}

PluginFieldCollection const* CorrelationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

char const* CorrelationPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void CorrelationPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

IPluginV2DynamicExt* CorrelationPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);

        int32_t outputChannel{81};
        int32_t patchWidth{9};
        int32_t patchHeight{9};
        int32_t dilation{1};

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            PLUGIN_VALIDATE(fc->fields[i].name != nullptr);
            std::string fieldName(fc->fields[i].name);
            if (fieldName.compare("output_channel") == 0)
            {
                outputChannel = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            if (fieldName.compare("patch_height") == 0)
            {
                patchHeight = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            if (fieldName.compare("patch_width") == 0)
            {
                patchWidth = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            if (fieldName.compare("dilation") == 0)
            {
                dilation = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }

        CorrelationPlugin* plugin = new CorrelationPlugin(outputChannel, patchHeight, patchWidth,
                                                          dilation);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* CorrelationPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        CorrelationPlugin* plugin = new CorrelationPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
