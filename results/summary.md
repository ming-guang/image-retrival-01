# Retrieval evaluation

| Set | Feature | MAP@3 | MAP@5 | MAP@11 | MAP@21 |
|---|---|---|---|---|---|
| basic | color_hist | 0.3808 | 0.2843 | 0.2119 | 0.2002 |
| basic | correlogram | 0.3101 | 0.2320 | 0.1686 | 0.1598 |
| basic | sift | 0.2521 | 0.1785 | 0.1310 | 0.1267 |
| basic | orb | 0.1727 | 0.1260 | 0.0927 | 0.0892 |
| improved | color_hist | 0.3754 | 0.2810 | 0.2083 | 0.1972 |
| improved | correlogram | 0.3190 | 0.2383 | 0.1754 | 0.1645 |
| improved | sift | 0.2211 | 0.1623 | 0.1180 | 0.1143 |
| improved | orb | 0.1033 | 0.0748 | 0.0565 | 0.0569 |
| improved | hu | 0.0293 | 0.0215 | 0.0160 | 0.0157 |
| improved | lbp | 0.2819 | 0.2076 | 0.1517 | 0.1461 |
| improved | hog | 0.2253 | 0.1558 | 0.1047 | 0.0985 |
| improved | combined | 0.2135 | 0.1478 | 0.0996 | 0.0898 |
| improved | combinedw | 0.4326 | 0.3310 | 0.2454 | 0.2312 |

## Query speed (combined feature)

| Method | Median search time (ms) | Speedup |
|---|---|---|
| Brute-force | 1.936 | 1.0x |
| FLANN kd-tree | 0.163 | 11.9x |
