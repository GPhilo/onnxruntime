// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {
class Graph;
class Node;

bool IsMaxPoolSupportedByXNNPack(const Node& nodeRef, bool input_is_nchw);
Status ReplaceMaxPool(Graph& main_graph, Node& nodeRef, bool& modified);

}  // namespace onnxruntime