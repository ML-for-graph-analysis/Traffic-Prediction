Traffic-Prediction
=================

This is the pytorch implementation for class project of AIGS703I(인공지능특론:그래프분석을 위한 기계학습)

It is based on [GraphWaveNet](https://github.com/nnzhan/Graph-WaveNet)

Dataset: Metr-LA, PeMS-Bay

# EDA 

* Role analysis of traffic graph data
* Community analysis of traffic graph data(Louvain.ipynb)


# Train Commands

## Model 1.2
```
python train.py --data data/METR-LA --savehorizon True --sheetname Model2_METR-LA_60min --adjdata data/sensor_graph/adj_mx.pkl --batch_size 32 --adjtype doubletransition
```

```
python train.py --data data/PEMS-BAY --savehorizon True --sheetname Model2_PEMS-BAY_60min --num_nodes 325 --adjdata data/sensor_graph/adj_mx_bay.pkl batch_size 32 --nhid 16 --adjtype doubletransition
```


## Model 2
```
python train.py --data data/METR-LA --savehorizon True --sheetname Model1_METR-LA_60min --adjdata data/sensor_graph/adj_mx.pkl --nhid 16 --batch_size 32 --adjtype transition
```

```
python train.py --data data/PEMS-BAY --savehorizon True --sheetname Model1_PEMS-BAY_60min --num_nodes 325 --adjdata data/sensor_graph/adj_mx_bay.pkl --nhid 16 --batch_size 32 --adjtype transition
```

## Model 3 

`--batch_size 32`

## Model 4 

`--batch_size 32`


