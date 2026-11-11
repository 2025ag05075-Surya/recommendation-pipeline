# Model Evaluation (K=10, relevant = rating >= 4.0, time-based test split)

Evaluated on 1,900 test users with >=1 relevant item. Best value per metric in **bold**.

| Model | precision@10 | recall@10 | ndcg@10 | coverage | diversity | rmse | mae |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0927 | 0.1329 | 0.1234 | 0.0663 | 0.7147 | **0.8467** | 0.6911 |
| svd_cf | **0.1369** | **0.1853** | **0.1989** | 0.3400 | **0.8101** | 0.8496 | **0.6820** |
| hybrid | 0.1247 | 0.1683 | 0.1868 | **0.5525** | 0.5512 | 0.8496 | 0.6820 |
