# Graph Pipeline

The `GraphPipeline` provides directed acyclic graph (DAG) execution with natural output sharing. However, **you don't need to use Pipeline explicitly** - use the simpler connection patterns instead.

## Quick Start (No Pipeline Required)

### PyTorch-Style Chaining

```python
from regenbogen import execute, Node

source = SourceNode(name="source")
processor = ProcessorNode()(source)  # Chain nodes PyTorch-style

# Execute without explicit Pipeline
output = execute(processor)
```

### Explicit Connections

```python
source = SourceNode(name="source")
processor = ProcessorNode(name="processor")

# Connect using .connect() method
source.outputs.frame.connect(processor.inputs.frame)

# Or use static method
Node.connect(source.outputs.frame, processor.inputs.frame)

# Execute
output = execute(processor)
```

## Natural Branching

Multiple nodes can read from the same output:

```python
source = SourceNode()
proc1 = ProcessorNode1()(source)
proc2 = ProcessorNode2()(source)  # Both read from source

# Execute both
results = execute(proc1, proc2, return_all=True)
```

## Node Definition

Nodes define inputs via process() method parameters with type annotations:

```python
from regenbogen import Node
from regenbogen.interfaces import Frame
import numpy as np

class MyNode(Node):
    def process(self, frame: Frame) -> np.ndarray:
        # Process and return pointcloud
        return pointcloud
```

## Connection Methods

### 1. PyTorch-Style Chaining
```python
node2 = Node2()(node1)
```

### 2. Port Connect Method
```python
node1.outputs.result.connect(node2.inputs.input)
```

### 3. Static Connect Method
```python
Node.connect(node1.outputs.result, node2.inputs.input)
```

## Multi-Input Nodes

Nodes with multiple inputs use explicit connections:

```python
merger = MergerNode()
source1.outputs.data.connect(merger.inputs.input_a)
source2.outputs.data.connect(merger.inputs.input_b)

output = execute(merger)
```

## Streaming Data

Source nodes that yield data automatically enable streaming:

```python
class SourceNode(Node):
    def process(self, input_data=None):
        for item in dataset:
            yield item

source = SourceNode()
processor = ProcessorNode()(source)

# Streams automatically
for result in execute(processor):
    print(result)
```

## Using Pipeline Class (Optional)

For more control, you can still use GraphPipeline explicitly:

```python
from regenbogen import GraphPipeline

pipeline = GraphPipeline(name="MyPipeline")
pipeline.add_node(source)
pipeline.add_node(processor, inputs={"frame": source.outputs.frame})

results = pipeline.process()
```

## Validation

```python
is_valid, errors = pipeline.validate()
if not is_valid:
    for error in errors:
        print(error)
```

## Examples

See `examples/pytorch_style_example.py` for complete examples demonstrating:
- PyTorch-style chaining
- Explicit connections
- Natural branching
- Mixed connection styles
- Multi-input nodes
