#include "core/xnnpack/optimizer/layout_helper.h"
#include "core/optimizer/transpose_optimizer/optimizer_utils.h"
#include "core/optimizer/selectors_actions/helpers.h"
namespace onnxruntime {

// It assumes the new node has the same number of inputs/outputs as the old node. And types of each node arg do not
// change. For each node arg, only the shape may change.
Status ReplaceNode(Graph& main_graph, Node& old_node, const std::string& op_type, const std::string& description,
                   const NodeAttributes* attributes, const std::string& domain, Node** out) {
  Node& new_node = main_graph.AddNode(old_node.Name(), op_type, description, {}, {}, attributes, domain);

  // Move all the inputs to the new node
  ValueMoveInfo move_info(ArgType::kInput, ArgType::kInput);
  ORT_RETURN_IF_ERROR(MoveInputOutput(main_graph, old_node, new_node, move_info, false));
  // Move all the outputs to the new node
  ValueMoveInfo move_info2(ArgType::kOutput, ArgType::kOutput);
  ORT_RETURN_IF_ERROR(MoveInputOutput(main_graph, old_node, new_node, move_info2, false));
  // Clear output shapes.
  for (NodeArg* p : new_node.MutableOutputDefs()) {
    if (p) p->ClearShape();
  }
  if (!main_graph.RemoveNode(old_node.Index())) {
    return Status(common::ONNXRUNTIME, common::FAIL, "remove node failed");
  }
  *out = &new_node;
  return Status::OK();
}


static Status CreateTransposeNode(Graph& main_graph, const std::vector<int64_t>& input_perm, bool create_input,
                           Node** new_node, NodeArg** newarg) {
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  if (create_input) {
    std::string output_name = main_graph.GenerateNodeArgName("trans");
    NodeArg& transpose_output = main_graph.GetOrCreateNodeArg(output_name, nullptr);
    *newarg = &transpose_output;
    inputs.push_back(&transpose_output);
  } else {
    std::string output_name = main_graph.GenerateNodeArgName("trans");
    NodeArg& transpose_output = main_graph.GetOrCreateNodeArg(output_name, nullptr);
    *newarg = &transpose_output;
    outputs.push_back(&transpose_output);
  }
  Node& transpose_node = main_graph.AddNode("", "Transpose", "", inputs, outputs, nullptr, kOnnxDomain);
  transpose_node.AddAttribute("perm", input_perm);
  *new_node = &transpose_node;
  return Status::OK();
}

Status TranposeNCHWToNHWC(Graph& main_graph, int rank, Node& nodeRef, Node** new_node) {
  std::vector<int64_t> input_perm = onnx_layout_transformation::ChannelFirstToLastPerm(rank);
  std::vector<int64_t> output_perm = onnx_layout_transformation::ChannelLastToFirstPerm(rank);
  ORT_RETURN_IF_ERROR(TransposeInput(main_graph, input_perm, 0, nodeRef));
  ORT_RETURN_IF_ERROR(main_graph.UpdateShapeInference(nodeRef));
  ORT_RETURN_IF_ERROR(TransposeOutput(main_graph, output_perm, 0, nodeRef, new_node));
  return Status::OK();
}

Status TransposeInput(Graph& main_graph, const std::vector<int64_t>& input_perm, int input_index, Node& node) {
  InOutDefSlot src_slot{ArgType::kInput, input_index};
  Node* transpose_node = nullptr;
  NodeArg* new_arg = nullptr;
  ORT_RETURN_IF_ERROR(CreateTransposeNode(main_graph, input_perm, false, &transpose_node, &new_arg));
  // Append a single slot to transpose_node. As the dest is empty, it will be the first one.
  ORT_RETURN_IF_ERROR(MoveInputOutput(main_graph, node, *transpose_node,
                                          ValueMoveInfo(src_slot, ArgType::kInput, false, false), false));
  ORT_RETURN_IF_ERROR(main_graph.UpdateShapeInference(*transpose_node));
  main_graph.AddEdge(transpose_node->Index(), node.Index(), 0, input_index);
  return Status::OK();
}

Status TransposeOutput(Graph& main_graph, const std::vector<int64_t>& output_perm, int output_index, Node& node,
                       Node** new_node) {
  InOutDefSlot src_slot{ArgType::kOutput, output_index};
  Node* transpose_node = nullptr;
  NodeArg* transpose_input = nullptr;
  ORT_RETURN_IF_ERROR(CreateTransposeNode(main_graph, output_perm, true, &transpose_node, &transpose_input));
  node.MutableOutputDefs()[output_index]->ClearShape();
  // Append a single slot to dest. As the dest is empty, it will be the first one.
  ORT_RETURN_IF_ERROR(MoveInputOutput(main_graph, node, *transpose_node,
                                          ValueMoveInfo(src_slot, ArgType::kOutput, false, false), false));
  node.MutableOutputDefs()[output_index] = transpose_input;
  main_graph.AddEdge(node.Index(), transpose_node->Index(), output_index, 0);
  ORT_RETURN_IF_ERROR(main_graph.UpdateShapeInference(node));
  ORT_RETURN_IF_ERROR(main_graph.UpdateShapeInference(*transpose_node));

  if (new_node) {
    *new_node = transpose_node;
  }
  return Status::OK();
}
}  // namespace onnxruntime