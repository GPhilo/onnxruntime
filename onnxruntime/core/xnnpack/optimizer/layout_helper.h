// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/graph/schema_registry.h"

namespace onnxruntime {
Status ReplaceNode(Graph& main_graph, Node& old_node, const std::string& op_type, const std::string& description,
                   const NodeAttributes* attributes, const std::string& domain, Node** out);
Status TranposeNCHWToNHWC(Graph& main_graph, int rank, Node& nodeRef, Node** new_node = nullptr);
Status TransposeInput(Graph& main_graph, const std::vector<int64_t>& input_perm, int input_index, Node& node);
Status TransposeOutput(Graph& main_graph, const std::vector<int64_t>& output_perm, int output_index, Node& node,
                       Node** new_node);
}  // namespace onnxruntime