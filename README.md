# Traffic-Prediction
Dataset: Metr-LA, PeMS-Bay

* Role analysis of traffic graph data
* Community analysis of traffic graph data(Louvain.ipynb)


# Train Commands

```python train.py --device cuda --data data/METR-LA --savehorizon True --sheetname Model1_METR-LA_60min --adjdata data/sensor_graph/adj_mx_bay.pkl --nhid 16 --batch_size 32 --adjtype transition```

```python train.py --device cuda --data data/PEMS-BAY --savehorizon True --sheetname Model1_PEMS-BAY_60min --num_nodes 325 --adjdata data/sensor_graph/adj_mx_bay.pkl --nhid 16 --batch_size 32 --adjtype transition```

# Hyperarameter 

\# of heads : 2
