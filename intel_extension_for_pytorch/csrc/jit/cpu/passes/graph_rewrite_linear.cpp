#include <ATen/code_template.h>
#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"

namespace torch {
namespace jit {
namespace graph_rewrite {

using namespace at::jit;
using namespace torch_ipex::cpu;

void replaceFrozenIPEXLinearWithAtenLinear(
    Block* b,
    std::vector<Node*>& get_data_handle_nodes) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      replaceFrozenIPEXLinearWithAtenLinear(block, get_data_handle_nodes);
    }
    if (n->kind() == Symbol::fromQualString("torch_ipex::ipex_linear")) {
      if (!(constant_as<at::Tensor>(n->namedInput("weight")).has_value())) {
        continue;
      }

      auto input_size_option = n->inputs()
                                   .at(0)
                                   ->type()
                                   ->cast<TensorType>()
                                   ->sizes()
                                   .concrete_sizes();
      if (!(input_size_option.has_value() &&
            input_size_option.value().size() >= 2)) {
        continue;
      }
      auto prepack_node = n->inputs().at(3)->node()->inputs().at(0);
      // For graph before "freeze", cannot get custom class to repack
      if (!toIValue(prepack_node).has_value())
        continue;
      auto linear_op_ctx =
          toIValue(prepack_node).value().toCustomClass<LinearOpContext>();
      at::Tensor weight_tensor = linear_op_ctx->to_public(
          constant_as<at::Tensor>(n->namedInput("weight")).value());
      WithInsertPoint guard(n);
      auto graph = n->owningGraph();

      auto aten_linear = graph->insertNode(graph->create(aten::linear));
      aten_linear->addInput(n->inputs().at(0));
      IValue weight_value(weight_tensor);
      auto weight = graph->insertConstant(weight_value);
      aten_linear->addInput(weight);
      aten_linear->addInput(n->inputs().at(2));
      aten_linear->output()->setType(n->output()->type()->cast<TensorType>());
      n->output()->replaceAllUsesWith(aten_linear->output());
      get_data_handle_nodes.emplace_back(n->inputs().at(3)->node());
    }
  }
  EliminateDeadCode(b);
}

void replaceFrozenIPEXLinearWithAtenLinear(std::shared_ptr<Graph>& graph) {
  std::vector<Node*> get_data_handle_nodes;
  replaceFrozenIPEXLinearWithAtenLinear(graph->block(), get_data_handle_nodes);
  for (auto& n : get_data_handle_nodes) {
    n->destroy();
  }
  EliminateDeadCode(graph);
}

void insertPrePackedLinearOp(Block* b, std::unordered_set<Node*>& aten_linear) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      insertPrePackedLinearOp(block, aten_linear);
    }
    if (n->kind() != aten::linear)
      continue;
    WithInsertPoint guard(n);
    auto graph = n->owningGraph();
    auto input_size_option =
        n->inputs().at(0)->type()->cast<TensorType>()->sizes().concrete_sizes();
    if (!(input_size_option.has_value() &&
          input_size_option.value().size() >= 2)) {
      continue;
    }
    auto input_size = input_size_option.value();
    int64_t b_size = std::accumulate(
                         input_size.begin(),
                         input_size.end(),
                         1,
                         std::multiplies<double>()) /
        input_size[input_size.size() - 1];
    IValue batch_size_value(b_size);
    auto batch_size = graph->insertConstant(batch_size_value);
    auto tt = n->inputs().at(1)->type()->cast<TensorType>();
    auto weight_size_option = tt->sizes().concrete_sizes();
    if (!(weight_size_option.has_value() &&
          weight_size_option.value().size() == 2)) {
      continue;
    }
    auto weight_dtype_option = tt->scalarType();
    if (!(weight_dtype_option.has_value() &&
              (weight_dtype_option.value() == at::ScalarType::BFloat16) ||
          aten_linear.find(n) == aten_linear.end())) {
      continue;
    }
    auto weight_size = weight_size_option.value();

    // Note that once creating a graph node, make sure it is also inserted into
    // the graph, for: PyTorch (when disabled TE) has a check on the graph node,
    // pointing out that every mutable value in the system has a corresponding
    // element. So if creating a graph node but not inserted, it will not pass
    // the check since its graph element is not initialized. Details please
    // refer to
    // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/ir/alias_analysis.cpp#L1956
    auto prepack_node = graph->create(
        Symbol::fromQualString("ipex_prepack::linear_prepack"), 1);
    for (auto i = 1; i < n->inputs().size(); ++i) {
      Value* v = n->inputs().at(i);
      prepack_node->addInput(v);
    }
    prepack_node->addInput(batch_size);
    prepack_node->output()->setType(
        getCustomClass("__torch__.torch.classes.ipex_prepack.LinearOpContext"));
    graph->insertNode(prepack_node);
    auto prepack_linear = graph->insertNode(
        graph->create(Symbol::fromQualString("ipex_prepack::linear_run"), 1));
    prepack_linear->addInput(n->inputs().at(0));
    prepack_linear->addInput(prepack_node->output());
    prepack_linear->output()->setType(n->output()->type()->cast<TensorType>());
    auto v = n->outputs().at(0);
    n->output()->replaceAllUsesWith(prepack_linear->output());
  }
  EliminateDeadCode(b);
}

void insertPrePackedLinearOp(
    std::shared_ptr<Graph>& graph,
    std::unordered_set<Node*>& aten_linear) {
  insertPrePackedLinearOp(graph->block(), aten_linear);
}

void RecordAtenLinearNodes(Block* b, std::unordered_set<Node*>& aten_linear) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      RecordAtenLinearNodes(block, aten_linear);
    }
    if (n->kind() == aten::linear) {
      aten_linear.insert(n);
    }
  }
  EliminateDeadCode(b);
}

void RecordAtenLinearNodes(
    std::shared_ptr<Graph>& graph,
    std::unordered_set<Node*>& aten_linear) {
  RecordAtenLinearNodes(graph->block(), aten_linear);
  EliminateDeadCode(graph);
}

void fuseLinearWithEltwise(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter_relu, rewriter_gelu, rewriter_silu,
      rewriter_sigmoid, rewriter_swish, rewriter_tanh;
  std::array<std::string, 2> relu_operators = {"relu", "relu_"};
  std::array<std::string, 2> sigmoid_operators = {"sigmoid", "sigmoid_"};
  std::array<std::string, 2> silu_operators = {"silu", "silu_"};
  std::array<std::string, 2> mul_operators = {"mul", "mul_"};
  std::array<std::string, 2> tanh_operators = {"tanh", "tanh_"};

  auto linear_relu_rstring = CodeTemplate(R"(
     graph(%input, %packed_weight):
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res = aten::${relu}(%x)
        return (%res))");

  std::string linear_relu_fused = R"(
    graph(%input, %packed_weight):
        %res = ipex_prepack::linear_relu_run(%input, %packed_weight)
        return (%res))";

  auto linear_tanh_rstring = CodeTemplate(R"(
    graph(%input, %packed_weight):    
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res = aten::${tanh}(%x)
        return (%res))");

  std::string linear_tanh_fused = R"(
    graph(%input, %packed_weight):
        %res = ipex_prepack::linear_tanh_run(%input, %packed_weight)
        return (%res))";

  std::string linear_gelu = R"(
    graph(%input, %approximate, %packed_weight):
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res= aten::gelu(%x, %approximate)
        return (%res))";

  std::string linear_gelu_fused = R"(
    graph(%input, %approximate, %packed_weight):
        %res = ipex_prepack::linear_gelu_run(%input, %packed_weight, %approximate)
        return (%res))";

  auto linear_sigmoid_rstring = CodeTemplate(R"(
    graph(%input, %packed_weight):
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res= aten::${sigmoid}(%x)
        return (%res))");

  auto linear_silu_rstring = CodeTemplate(R"(
    graph(%input, %packed_weight):
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res= aten::${silu}(%x)
        return (%res))");

  auto linear_sigmoid_mul_rstring = CodeTemplate(R"(
    graph(%input, %packed_weight):
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %y = aten::${sigmoid}(%x)
        %res = aten::${mul}(%x, %y)
        return (%res))");

  std::string linear_swish_fused = R"(
    graph(%input, %packed_weight):
        %res = ipex_prepack::linear_swish_run(%input, %packed_weight)
        return (%res))";

  std::string linear_sigmoid_fused = R"(
    graph(%input, %packed_weight):
        %res = ipex_prepack::linear_sigmoid_run(%input, %packed_weight)
        return (%res))";

  for (const auto& relu : relu_operators) {
    TemplateEnv env;
    env.s("relu", relu);
    rewriter_relu.RegisterRewritePattern(
        linear_relu_rstring.format(env), linear_relu_fused);
  }

  for (const auto& tanh : tanh_operators) {
    TemplateEnv env;
    env.s("tanh", tanh);
    rewriter_tanh.RegisterRewritePattern(
        linear_tanh_rstring.format(env), linear_tanh_fused);
  }

  for (const auto& silu : silu_operators) {
    TemplateEnv env;
    env.s("silu", silu);
    rewriter_silu.RegisterRewritePattern(
        linear_silu_rstring.format(env), linear_swish_fused);
  }

  for (const auto& sigmoid : sigmoid_operators) {
    TemplateEnv env;
    env.s("sigmoid", sigmoid);
    rewriter_sigmoid.RegisterRewritePattern(
        linear_sigmoid_rstring.format(env), linear_sigmoid_fused);
    for (const auto& mul : mul_operators) {
      env.s("mul", mul);
      rewriter_swish.RegisterRewritePattern(
          linear_sigmoid_mul_rstring.format(env), linear_swish_fused);
    }
  }
  rewriter_silu.runOnGraph(graph);
  rewriter_sigmoid.runOnGraph(graph);
  rewriter_swish.runOnGraph(graph);
  rewriter_gelu.RegisterRewritePattern(linear_gelu, linear_gelu_fused);

  rewriter_relu.runOnGraph(graph);
  rewriter_tanh.runOnGraph(graph);
  rewriter_gelu.runOnGraph(graph);
}

void fuseLinearAddRelu(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter_add_v1, rewriter_add_v2;
  std::array<std::string, 2> add_operators = {"add", "add_"};

  // linear   Y
  //   \   /
  //    add
  // output = linear_output + alpha*Y
  auto linear_add_rstring_v1 = CodeTemplate(R"(
    graph(%input, %accumu, %alpha, %packed_weight):
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res = aten::${add}(%x, %accumu, %alpha)
        return (%res))");

  //  Y     linear
  //   \   /
  //    add
  // output = Y + alpha*linear_output, alpha need to one or none.
  auto linear_add_rstring_v2 = CodeTemplate(R"(
    graph(%input, %accumu, %alpha, %packed_weight):
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res = aten::${add}(%accumu, %x, %alpha)
        return (%res))");

  std::string linear_add_fused = R"(
    graph(%input, %accumu, %alpha, %packed_weight):
        %res = ipex_prepack::linear_add_run(%input, %accumu, %alpha, %packed_weight)
        return (%res))";

  // linear + add
  for (const auto& add : add_operators) {
    TemplateEnv env;
    env.s("add", add);
    rewriter_add_v1.RegisterRewritePattern(
        linear_add_rstring_v1.format(env), linear_add_fused);
    rewriter_add_v2.RegisterRewritePattern(
        linear_add_rstring_v2.format(env), linear_add_fused);
  }

  rewriter_add_v1.runOnGraph(graph, fuse_add_filter_v1);
  rewriter_add_v2.runOnGraph(graph, fuse_add_filter_v2);
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch
