# STP / IP Ablation Report

| Condition | STP | IP | Accuracy | Balanced Acc | Macro F1 | Mean Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline | on | on | 0.9946 | 0.9946 | 0.9946 | 0.0095 |
| no_ip | on | off | 0.9375 | 0.9375 | 0.9375 | 0.0017 |
| no_stp | off | on | 0.9964 | 0.9964 | 0.9964 | 0.0114 |
| no_stp_no_ip | off | off | 0.9321 | 0.9321 | 0.9324 | 0.0017 |
