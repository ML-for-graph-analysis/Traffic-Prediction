# Traffic-Prediction
Dataset: Metr-LA, PeMS-Bay

* Role analysis of traffic graph data
* Community analysis of traffic graph data(Louvain.ipynb)


# Train Commands


## Attention 

### Model 1.2
```python train.py --data data/METR-LA --savehorizon True --sheetname Model2_METR-LA_60min --adjdata data/sensor_graph/adj_mx.pkl --batch_size 32 --adjtype doubletransition```

```python train.py --data data/PEMS-BAY --savehorizon True --sheetname Model2_PEMS-BAY_60min --num_nodes 325 --adjdata data/sensor_graph/adj_mx_bay.pkl batch_size 32 --nhid 16 --adjtype doubletransition```


### Model 2
```python train.py --data data/METR-LA --savehorizon True --sheetname Model1_METR-LA_60min --adjdata data/sensor_graph/adj_mx.pkl --nhid 16 --batch_size 32 --adjtype transition```

```python train.py --data data/PEMS-BAY --savehorizon True --sheetname Model1_PEMS-BAY_60min --num_nodes 325 --adjdata data/sensor_graph/adj_mx_bay.pkl --nhid 16 --batch_size 32 --adjtype transition```

### Model 3 

`--batch_size 32`

### Model 4 

`--batch_size 32`

# Hyperarameter 

\# of attention heads : 2
