# Параллельные версии волновой схемы решения задачи Дирихле для уравнения Пуассона


| **TBB (Arm)**| **TBB (Arm)**                                                     ||||||
|:---------:|:-------:|:-------:|:-----------:|:-----------:|:-----------:|:-----------:|
|           | 1       | 2       | ускр. 2     | ускр. 4     | ускр. 6     | ускр. 8     |
| 100х100   | 0.0334  | 0.0526  | 0.634980989 | 0.257517348 | 0.218158067 | 0.2193      |
| 1000х1000 | 4.4119  | 2.8819  | 1.530899754 | 1.417796774 | 1.4005      | 1.260939152 |
| 1500х1500 | 8.7724  | 5.7567  | 1.523859155 | 1.50625     | 1.560147969 | 1.43396103  |
| 2000х2000 | 19.6145 | 12.8203 | 1.529956397 | 1.808354692 | 2.126993938 | 1.981522826 |
| 2500х2500 | 33.1644 | 22.2561 | 1.490126302 | 1.871390039 | 2.19548912  | 2.168855289 |
| 3000х3000 | 52.0017 | 33.4556 | 1.554349646 | 1.928824976 | 2.346287122 | 2.367210652 |

| **STD (Arm)**                                                                     |
|:---------:|:-------:|:-------:|:-----------:|:-----------:|:-----------:|:-------:|
|           | 1       | 2       | ускр. 2     | ускр.4      | ускр. 6     | ускр. 8 |
| 100х100   | 0.0334  | 0.0979  | 0.341164454 | 0.2336      | 0.1697      | 0.1467  |
| 1000х1000 | 4.4119  | 2.6829  | 1.644451899 | 2.1621      | 2.5021      | 2.2042  |
| 1500х1500 | 8.7724  | 5.3203  | 1.648854388 | 2.4659      | 2.817627032 | 2.4364  |
| 2000х2000 | 19.6145 | 11.0023 | 1.782763604 | 2.7800      | 3.5737      | 3.0325  |
| 2500х2500 | 33.1644 | 18.2964 | 1.812618876 | 2.934461187 | 3.9488      | 3.2139  |
| 3000х3000 | 52.0017 | 29.0831 | 1.788038414 | 2.7779      | 3.713929638 | 3.3796  |

| **OMP (Arm)**                                                            |
|:---------:|:-------:|:-------:|:-----------:|:------:|:-------:|:-------:|
|           | 1       | 2       | ускр. 2     | ускр.4 | ускр. 6 | ускр. 8 |
| 100х100   | 0.0334  | 0.0558  | 0.598566308 | 0.3728 | 0.3422  | 0.2892  |
| 1000х1000 | 4.4119  | 3.6589  | 1.205799557 | 1.6750 | 1.9643  | 2.1643  |
| 1500х1500 | 8.7724  | 6.9521  | 1.261834554 | 1.8402 | 2.1928  | 2.5824  |
| 2000х2000 | 19.6145 | 15.9368 | 1.230767783 | 1.7968 | 2.4333  | 2.9823  |
| 2500х2500 | 33.1644 | 27.2911 | 1.215209354 | 1.9052 | 2.6458  | 3.2125  |
| 3000х3000 | 52.0017 | 43.0229 | 1.208698158 | 1.8803 | 2.5148  | 2.9003  |

| **MPI (Arm)**                                                                |
|:-------------:|:-------:|:-------:|:-----------:|:------:|:-------:|:-------:|
|               | 1       | 2       | ускр. 2     | ускр.4 | ускр. 6 | ускр. 8 |
| 100х100       | 0.0334  | 0.0298  | 1.120805369 | 0.8068 | 0.6018  | 0.1692  |
| 1000х1000     | 4.4119  | 2.4920  | 1.770425361 | 2.3749 | 2.8976  | 1.0932  |
| 1500х1500     | 8.7724  | 4.7979  | 1.828383251 | 2.7002 | 3.0689  | 1.0060  |
| 2000х2000     | 19.6145 | 10.5804 | 1.853852406 | 2.9297 | 3.6958  | 1.0154  |
| 2500х2500     | 33.1644 | 18.0182 | 1.84060561  | 3.0647 | 4.0946  | 1.0077  |
| 3000х3000     | 52.0017 | 28.6209 | 1.816913514 | 3.0188 | 3.9676  | 1.3529  |

| **Hybrid MPI & OMP (Arm)**                                                                                                       |
|:--------------------------:|:-------:|:----------:|:----------:|:-----------:|:-----------:|:---------:|:----------:|:----------:|
|                            |    1    | 2 MPI2 OMP | 2 MPI4 OMP |  ускр. 2(2) |  ускр. 2(4) | ускр.4(2) | ускр. 4(4) | ускр. 6(2) |
|           100х100          |  0.0334 |   0.0665   |   0.1182   | 0.502255639 | 0.282571912 |   0.1881  |   0.1180   |   0.0957   |
|          1000х1000         |  4.4119 |   2.1965   |   1.8651   | 2.008604598 |  2.36550319 |   2.3540  |   2.0639   |   1.9037   |
|          1500х1500         |  8.7724 |   4.2292   |   3.2803   |  2.07424572 | 2.674267597 |   2.5988  |   2.3539   |   2.2016   |
|          2000х2000         | 19.6145 |   8.2724   |   5.6700   | 2.371077317 | 3.459347443 |   3.3081  |   3.1678   |   2.9177   |
|          2500х2500         | 33.1644 |   13.5488  |   8.9398   | 2.447773973 | 3.709747422 |   3.4402  |   3.4506   |   3.1829   |
|          3000х3000         | 52.0017 |   21.4372  |   14.7268  | 2.425769224 | 3.531092973 |   3.5110  |   3.5966   |   3.2930   |
