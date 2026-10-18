# Data and Manifest Guidelines

## Manifest Schema

Manifest files must be CSV files with the following columns:

```text
image_path,label,source_dataset,generator_family,content_group,split,format
```

### Column Descriptions:
- `image_path`: Relative or absolute path to the decoded image file.
- `label`: Integer: `0` for real/authentic, `1` for AI-generated.
- `source_dataset`: Identifier for the dataset origin (e.g., `coco`, `defactify`).
- `generator_family`: Generation source, e.g., `real`, `sd21`, `sdxl`, `sd3`, `dalle3`, `midjourney`.
- `content_group`: Semantic cluster or identifier used to prevent near-duplicate leakage across splits.
- `split`: One of `train`, `validation`, or `test`.
- `format`: Lowercase image format, e.g., `jpeg`, `png`, `webp`.

## Leakage Prevention Rules:
1. No image or content group may cross between `train`, `validation`, and `test` splits.
2. Generator family holdout sets must be strictly reserved for generalization testing.
3. Class and format balances should be monitored across splits.

