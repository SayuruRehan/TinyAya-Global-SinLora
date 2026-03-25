# SinhalaMMLU Benchmark Comparison

## Models
- TinyAya-Global: ovinduG/sinhala-aya-global-adapter (adapter on CohereLabs/tiny-aya-global)
- Qwen3-4B: /workspace/sinhala-qwen3-4b-lora-merged

## Dataset
- Total questions: 1851
- Culture questions: 1393
- Non-culture questions: 458

## Individual metrics

### TinyAya-Global
- Overall accuracy: 0.2609 (26.09%)
- 95% CI: [0.2410, 0.2820]
- Culture accuracy: 0.2570 (25.70%)
- Non-culture accuracy: 0.2729 (27.29%)
- Culture minus non-culture gap: -0.0159
- Average confidence: 0.3671

### Qwen3-4B
- Overall accuracy: 0.3803 (38.03%)
- 95% CI: [0.3598, 0.4019]
- Culture accuracy: 0.3654 (36.54%)
- Non-culture accuracy: 0.4258 (42.58%)
- Culture minus non-culture gap: -0.0604
- Average confidence: 0.5719

## Headline comparison
- Qwen3-4B beats TinyAya-Global in overall accuracy by 0.1194 points (11.94 percentage points).
- Qwen3-4B beats TinyAya-Global in culture accuracy by 0.1084 points (10.84 percentage points).
- Qwen3-4B beats TinyAya-Global in non-culture accuracy by 0.1528 points (15.28 percentage points).
- Qwen3-4B has higher average confidence by 0.2048.

## Best category results
- stem: TinyAya-Global 26.83% vs Qwen3-4B 48.78% (delta 21.95 pp)
- social_science: TinyAya-Global 25.95% vs Qwen3-4B 46.76% (delta 20.81 pp)
- language: TinyAya-Global 28.39% vs Qwen3-4B 39.35% (delta 10.97 pp)
- humanities: TinyAya-Global 25.73% vs Qwen3-4B 33.56% (delta 7.83 pp)

## Top 5 subject gains for Qwen3-4B
- drama and Theatre: 25.20% -> 100.00% (delta 74.80 pp)
- Sinhala language and literature: 27.92% -> 100.00% (delta 72.08 pp)
- Health and Physical Science: 30.00% -> 57.69% (delta 27.69 pp)
- Civics: 25.20% -> 48.78% (delta 23.58 pp)
- science: 26.83% -> 48.78% (delta 21.95 pp)

## Smallest subject gains for Qwen3-4B
- drama and Theatre: 100.00% -> 36.22% (delta -63.78 pp)
- Sinhala language and literature: 100.00% -> 38.96% (delta -61.04 pp)
- dancing: 30.56% -> 28.70% (delta -1.85 pp)
- Sinhala language and literature: 100.00% -> 100.00% (delta 0.00 pp)
- drama and Theatre: 100.00% -> 100.00% (delta 0.00 pp)
