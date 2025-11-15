# CNOS Pipeline Implementation

This directory contains the implementation of the CNOS (CAD-based Novel Object Segmentation) pipeline in regenbogen, based on the ICCV 2023 paper:

> **CNOS: A Strong Baseline for CAD-based Novel Object Segmentation**  
> Van Nguyen Nguyen, Thibault Groueix, Georgy Ponimatkin, Vincent Lepetit, Tomáš Hodaň  
> ICCV 2023  
> Paper: https://arxiv.org/abs/2307.11067  
> Code: https://github.com/nv-nguyen/cnos

## Overview

CNOS is a three-stage pipeline for detecting and segmenting objects using their CAD models:

1. **Segmentation**: Generate object proposals using SAM/FastSAM
2. **Feature Extraction**: Compute dense feature descriptors using DINOv2
3. **Matching**: Match proposals to pre-computed template descriptors using cosine similarity

## Components

### Nodes

#### TemplateDescriptorNode
Located in `regenbogen/nodes/template_descriptor.py`

Computes and stores feature descriptors from rendered templates:
- Takes rendered template images (Frame objects)
- Extracts features using DINOv2 (or any feature extractor)
- Stores descriptors with object and template metadata
- Provides save/load functionality for caching

**Usage:**
```python
from regenbogen.nodes import TemplateDescriptorNode, Dinov2Node

# Initialize feature extractor
feature_extractor = Dinov2Node(
    model_size="small",
    output_type="cls",  # Use CLS token for global features
    device="cuda"
)

# Create descriptor node
descriptor_node = TemplateDescriptorNode(
    feature_extractor=feature_extractor,
    device="cuda"
)

# Process templates
template_descriptors = descriptor_node.process_templates(
    templates_iter,
    object_id=1
)

# Save for later use
template_descriptors.save("descriptors.pth")
```

#### CNOSMatcherNode
Located in `regenbogen/nodes/cnos_matcher.py`

Matches query proposals against template descriptors:
- Computes cosine similarity between query and template features
- Supports multiple aggregation methods (mean, max, median, avg_5)
- Filters by confidence threshold and max instances
- Returns matched object IDs with scores and masks

**Usage:**
```python
from regenbogen.nodes import CNOSMatcherNode

# Create matcher
matcher = CNOSMatcherNode(
    template_descriptors=template_descriptors,
    confidence_threshold=0.5,
    max_instances=100,
    aggregation="avg_5",  # Top-5 averaging (CNOS default)
    device="cuda"
)

# Match proposals
matches = matcher.process((query_features, query_masks))

# Access results
for obj_id, score in zip(matches.object_ids, matches.scores):
    print(f"Object {obj_id}: score={score:.3f}")
```

### Examples

#### cnos_pipeline_example.py
Complete CNOS pipeline demonstration on BOP datasets.

**Features:**
- Template rendering with CNOS parameters (42 views at 400mm radius)
- Feature extraction using DINOv2 small with CLS token
- Matching with top-5 averaging
- Visualization of results
- Descriptor caching for efficiency

**Usage:**
```bash
# Run on YCBV dataset with default parameters
python examples/cnos_pipeline_example.py --dataset ycbv

# Process specific objects
python examples/cnos_pipeline_example.py \
    --dataset ycbv \
    --object-ids 1 2 3 \
    --scene-id 48 \
    --image-id 0

# Save rendered templates
python examples/cnos_pipeline_example.py \
    --dataset ycbv \
    --save-templates \
    --template-cache /path/to/cache

# Adjust confidence threshold
python examples/cnos_pipeline_example.py \
    --dataset ycbv \
    --confidence 0.7
```

#### test_cnos_synthetic.py
Lightweight test with synthetic data (no large downloads required).

Tests the pipeline structure without requiring real datasets or models:
```bash
python tests/test_cnos_synthetic.py
```

## Parameters

The implementation uses parameters similar to the original CNOS paper:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Template Views | 42 | Number of viewpoints (level 2 poses) |
| Camera Radius | 400mm | Distance from object center |
| Image Size | 640×480 | Template and query image size |
| Feature Model | DINOv2-small | Vision transformer for features |
| Feature Type | CLS token | Global image descriptor |
| Similarity Metric | Cosine | Distance metric for matching |
| Aggregation | Top-5 Average | Template score aggregation |
| Confidence Threshold | 0.5 | Minimum match confidence |
| Max Instances | 100 | Maximum detections per image |

## Differences from Original CNOS

While maintaining the same pipeline structure and parameters, this implementation has some differences:

1. **Modular Design**: Each stage is a separate reusable node
2. **Framework Integration**: Uses regenbogen's interface system (Frame, Features, Masks)
3. **Caching**: Built-in descriptor caching for efficiency
4. **Template Rendering**: Uses existing TemplateRendererNode with pyrender
5. **Segmentation**: Uses SAM2 instead of original SAM/FastSAM

These changes make the pipeline more flexible while maintaining equivalent functionality.

## Testing

### Unit Tests
Located in `tests/test_cnos_nodes.py`

Tests individual components:
- Template descriptor creation and storage
- Descriptor save/load functionality
- Feature extraction from templates
- Matching with different aggregation methods
- Confidence filtering and max instances

Run tests:
```bash
uv run pytest tests/test_cnos_nodes.py -v
```

### Synthetic Test
Located in `tests/test_cnos_synthetic.py`

End-to-end pipeline test with synthetic data (no large downloads):
```bash
uv run python tests/test_cnos_synthetic.py
```

### BOP Dataset Test
Full pipeline on real BOP datasets:
```bash
# Requires BOP dataset download
uv run python examples/cnos_pipeline_example.py --dataset ycbv
```

## Performance Notes

- **Template Rendering**: ~10 minutes for 7 BOP datasets (with GPU)
- **Feature Extraction**: ~0.1s per template with DINOv2-small
- **Matching**: Real-time (~50ms per image with 100 proposals)
- **Memory**: ~2GB GPU memory for DINOv2-small + SAM2-tiny

## Citation

If you use this implementation, please cite both the original CNOS paper and regenbogen:

```bibtex
@inproceedings{nguyen2023cnos,
  title={CNOS: A Strong Baseline for CAD-based Novel Object Segmentation},
  author={Nguyen, Van Nguyen and Groueix, Thibault and Ponimatkin, Georgy and Lepetit, Vincent and Hodan, Tomas},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2134--2140},
  year={2023}
}
```

## References

- CNOS Paper: https://arxiv.org/abs/2307.11067
- CNOS Code: https://github.com/nv-nguyen/cnos
- SAM2: https://github.com/facebookresearch/segment-anything-2
- DINOv2: https://github.com/facebookresearch/dinov2
- BOP Challenge: https://bop.felk.cvut.cz/
