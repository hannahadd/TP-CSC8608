# TP4 — GNN pour la classification de nœuds (Cora)

> **Auteur :** Hanna Haddaoui
> **Environnement :** CPU local (Apple M-series, macOS)


## 1. Structure du projet

```
TP4
├── __pycache__
│   ├── data.cpython-313.pyc
│   ├── models.cpython-313.pyc
│   └── utils.cpython-313.pyc
├── configs
│   ├── baseline_mlp.yaml
│   ├── gcn.yaml
│   └── sage_sampling.yaml
├── image.png
├── rapport.md
├── runs
│   ├── gcn.pt
│   ├── mlp.pt
│   └── sage.pt
└── src
    ├── __pycache__
    │   ├── data.cpython-311.pyc
    │   ├── data.cpython-313.pyc
    │   ├── models.cpython-311.pyc
    │   ├── models.cpython-313.pyc
    │   ├── utils.cpython-311.pyc
    │   └── utils.cpython-313.pyc
    ├── benchmark.py
    ├── data.py
    ├── models.py
    ├── smoke_test.py
    ├── train.py
    └── utils.py

6 directories, 23 files
```

---

## 2. Smoke test — Environnement & Dataset Cora

```
=== Environment ===
torch: 2.10.0
cuda available: False
device: cpu

=== Dataset (Cora) ===
num_nodes: 2708
num_edges: 10556
num_node_features: 1433
num_classes: 7
train/val/test: 140 500 1000

OK: smoke test passed.
```

---

## 3. Baseline MLP — Résultats

### Pourquoi calculer les métriques sur train/val/test séparément ?

On calcule les métriques sur trois masques distincts pour des raisons pratiques :
- **train_mask** : surveiller l'apprentissage, détecter le sur-apprentissage (si train_acc >> val_acc).
- **val_mask** : guider le choix des hyperparamètres (lr, hidden_dim, dropout) sans toucher aux données de test.
- **test_mask** : mesure finale non biaisée, utilisée une seule fois pour comparer les modèles.

En ingénierie, utiliser le test set pour ajuster les hyperparamètres provoquerait une fuite de données (data leakage) et donnerait des chiffres trop optimistes. La séparation train/val/test garantit que la métrique finale est représentative de la performance réelle sur des données inédites.

### Sortie terminal

```
epoch=080 loss=0.0040 train_acc=1.0000 val_acc=0.5400 test_acc=0.5780 train_f1=1.0000 val_f1=0.5347 test_f1=0.5682 epoch_time_s=0.0065
epoch=100 loss=0.0051 train_acc=1.0000 val_acc=0.5460 test_acc=0.5680 train_f1=1.0000 val_f1=0.5439 test_f1=0.5577 epoch_time_s=0.0062
epoch=150 loss=0.0046 train_acc=1.0000 val_acc=0.5480 test_acc=0.5710 train_f1=1.0000 val_f1=0.5391 test_f1=0.5589 epoch_time_s=0.0065
epoch=200 loss=0.0038 train_acc=1.0000 val_acc=0.5460 test_acc=0.5690 train_f1=1.0000 val_f1=0.5328 test_f1=0.5564 epoch_time_s=0.0065
total_train_time_s=1.3392
train_loop_time=1.8752
checkpoint_saved: .../runs/mlp.pt
```

On observe un sur-apprentissage massif (train_acc=1.0 dès l'epoch 80, val_acc stagne autour de 0.55) : le MLP mémorise les 140 nœuds d'entraînement mais ne généralise pas, faute de signal structurel.

---

## 4. Baseline GCN — Comparaison MLP vs GCN

### Sortie terminal GCN

```
device: cpu
model: gcn
epochs: 200
epoch=001 loss=1.9471 train_acc=0.9286 val_acc=0.7020 test_acc=0.6980 train_f1=0.9313 val_f1=0.7069 test_f1=0.7061 epoch_time_s=0.0487
epoch=050 loss=0.0090 train_acc=1.0000 val_acc=0.7740 test_acc=0.8120 train_f1=1.0000 val_f1=0.7555 test_f1=0.8073 epoch_time_s=0.0124
epoch=100 loss=0.0083 train_acc=1.0000 val_acc=0.7780 test_acc=0.8020 train_f1=1.0000 val_f1=0.7638 test_f1=0.7975 epoch_time_s=0.0119
epoch=200 loss=0.0072 train_acc=1.0000 val_acc=0.7760 test_acc=0.8070 train_f1=1.0000 val_f1=0.7651 test_f1=0.8010 epoch_time_s=0.0120
total_train_time_s=2.4697
train_loop_time=3.5396
checkpoint_saved: .../runs/gcn.pt
```

### Tableau comparatif MLP vs GCN

| Modèle | test_acc | test_f1 | total_train_time_s |
|--------|----------|---------|--------------------|
| MLP    | 0.5690   | 0.5564  | 1.3392             |
| GCN    | 0.8070   | 0.8010  | 2.4697             |

### Pourquoi GCN dépasse le MLP sur Cora ?

Cora est un graphe à forte homophilie : les nœuds connectés appartiennent souvent à la même classe. GCN exploite ce signal en agrégeant les features des voisins, ce qui enrichit la représentation de chaque nœud. Le MLP ignore complètement la structure du graphe et traite chaque nœud de façon indépendante.

Sur Cora, les features bag-of-words sont déjà discriminantes, mais le set d'entraînement est très petit (140 nœuds). Le MLP sur-apprend ces 140 nœuds (train_acc=1.0, test_acc≈0.57) sans généraliser. GCN profite du lissage spectral : chaque couche de convolution propage l'information dans le voisinage à 1-hop puis 2-hop, ce qui lui permet d'apprendre à partir des voisins non labellisés. Le gain est très net : +23 points d'accuracy et +24 points de F1 pour seulement 2× plus de temps d'entraînement.

---

## 5. GraphSAGE avec neighbor sampling

### Sortie terminal GraphSAGE

```
device: cpu
model: sage
epochs: 200
sampling: NeighborLoader unavailable ('NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'), falling back to full-batch
epoch=001 loss=1.9462 train_acc=0.9929 val_acc=0.7320 test_acc=0.7430 train_f1=0.9929 val_f1=0.7314 test_f1=0.7359 epoch_time_s=0.0766
epoch=050 loss=0.0020 train_acc=1.0000 val_acc=0.7580 test_acc=0.7980 train_f1=1.0000 val_f1=0.7596 test_f1=0.7936 epoch_time_s=0.0221
epoch=100 loss=0.0030 train_acc=1.0000 val_acc=0.7640 test_acc=0.8020 train_f1=1.0000 val_f1=0.7522 test_f1=0.7962 epoch_time_s=0.0222
epoch=200 loss=0.0029 train_acc=1.0000 val_acc=0.7720 test_acc=0.7940 train_f1=1.0000 val_f1=0.7636 test_f1=0.7892 epoch_time_s=0.0223
total_train_time_s=4.5432
train_loop_time=7.5097
checkpoint_saved: .../runs/sage.pt
```

> **Note** : Sur Mac ARM, `pyg-lib` et `torch-sparse` ne sont pas disponibles, ce qui empêche le neighbor sampling. GraphSAGE a donc été entraîné en full-batch, comme GCN. Sur un cluster Linux/GPU, le sampling fonctionnerait normalement avec les hyperparamètres définis (batch_size=64, num_neighbors=[10, 10]).

### Tableau comparatif des trois modèles

| Modèle    | test_acc | test_f1 | total_train_time_s |
|-----------|----------|---------|--------------------|
| MLP       | 0.5690   | 0.5564  | 1.3392             |
| GCN       | 0.8070   | 0.8010  | 2.4697             |
| GraphSAGE | 0.7940   | 0.7892  | 4.5432             |

### Compromis du neighbor sampling

Le neighbor sampling de GraphSAGE remplace le full-batch par un mini-batch où l'on échantillonne un sous-graphe autour de chaque nœud cible. Pour chaque couche, on fixe un fanout (ici 10 voisins). Cela accélère considérablement l'entraînement sur les grands graphes car le coût par epoch ne croît plus avec la taille totale : on ne traite que batch_size nœuds à la fois.

Le risque principal est la variance du gradient : à chaque batch, on n'observe qu'un sous-graphe bruité. Les nœuds "hub" (très connectés) sont sur-échantillonnés, ce qui peut biaiser les mises à jour. Avec un fanout trop petit, certains voisins importants sont ignorés, dégradant la qualité des agrégations. Sur Cora (2708 nœuds), cet effet est limité — c'est d'ailleurs pourquoi GraphSAGE en full-batch obtient ici une accuracy proche de GCN. Sur un graphe de millions de nœuds, le bon réglage du fanout et de la taille de batch est critique pour converger sans exploser en mémoire. Il y a aussi un coût CPU non négligeable : la construction des sous-graphes par NeighborLoader tourne sur CPU et peut devenir un goulot d'étranglement si le GPU est très rapide.

---

## 6. Benchmarks latence d'inférence

### Sortie terminal benchmark

```
model: mlp
device: cpu
avg_forward_ms: 2.2153
num_nodes: 2708
ms_per_node_approx: 0.00081807

model: gcn
device: cpu
avg_forward_ms: 4.6168
num_nodes: 2708
ms_per_node_approx: 0.00170486

model: sage
device: cpu
avg_forward_ms: 13.9842
num_nodes: 2708
ms_per_node_approx: 0.00516403
```

### Pourquoi le warmup et la synchronisation CUDA ?

Le GPU fonctionne de façon **asynchrone** : quand Python appelle `model(x)`, PyTorch enfile l'opération dans une queue CUDA et rend la main immédiatement, sans attendre la fin du calcul. Si on mesure le temps sans `torch.cuda.synchronize()`, on mesure seulement le temps de mise en queue (quelques microsecondes), pas le temps réel d'exécution sur GPU.

Le **warmup** sert à "chauffer" le GPU avant de mesurer : les premières inférences déclenchent la compilation JIT des kernels CUDA, l'allocation de mémoire et le chargement des poids dans les caches. Ces coûts ponctuels fausseraient les mesures si inclus. En faisant 10 passes de warmup, on s'assure que les runs chronométrés représentent le régime permanent.

La synchronisation avant/après chaque run (`sync_if_cuda`) garantit que le timer encadre exactement l'exécution GPU, donnant des mesures stables et comparables entre les modèles. Sur CPU (comme ici), ces appels sont des no-ops mais le protocole reste correct.

---

## 7. Synthèse finale

### Tableau comparatif complet

| Modèle      | test_acc | test_macro_f1 | total_train_time_s | train_loop_time | avg_forward_ms |
|-------------|----------|---------------|--------------------|-----------------|----------------|
| MLP         | 0.5690   | 0.5564        | 1.3392             | 1.8752          | 2.2153         |
| GCN         | 0.8070   | 0.8010        | 2.4697             | 3.5396          | 4.6168         |
| GraphSAGE   | 0.7940   | 0.7892        | 4.5432             | 7.5097          | 13.9842        |

### Recommandation ingénieur

Le choix du modèle dépend directement de la contrainte opérationnelle. Si la **latence d'inférence** est le critère principal (ex : service temps réel), le MLP s'impose avec 2.2 ms de forward — soit 2× plus rapide que GCN et 6× plus rapide que GraphSAGE. Cependant, sa qualité est médiocre (test_acc=0.569, test_f1=0.556) à cause du sur-apprentissage sur le très petit set d'entraînement de Cora. Il n'est pertinent que si la structure du graphe n'est pas disponible à l'inférence.

Si l'on veut **maximiser la qualité** avec le graphe disponible, **GCN est le meilleur compromis** sur Cora : test_acc=0.807, test_f1=0.801, pour seulement 2.5 s d'entraînement et 4.6 ms de latence. Il tire parti de l'homophilie du graphe bien mieux que le MLP.

**GraphSAGE** est le choix pour les **grands graphes** où le full-batch GCN ne tient pas en mémoire. Ici en full-batch, il est plus lent (4.5 s d'entraînement, 14 ms de forward) pour une qualité légèrement inférieure à GCN — ce qui s'explique par l'architecture SAGE (agrégation mean sans normalisation spectrale). Sur un graphe de millions de nœuds avec le vrai neighbor sampling, il deviendrait le seul choix viable.

### Risque de protocole

Un risque majeur ici est la **non-comparabilité des conditions d'exécution** : tous les modèles ont tourné sur CPU (Mac ARM sans CUDA), ce qui homogénéise les mesures mais les rend non représentatives d'un déploiement GPU réel. Sur GPU, les rapports de latence seraient très différents (les ops matricielles de GCN bénéficient bien plus du GPU que le MLP). Pour un vrai projet, on ferait tourner tous les benchmarks sur le même hardware cible (même device, même charge mémoire).

Un second risque est le **seed unique** : un seul run par modèle ne permet pas de distinguer la variance d'initialisation de la vraie performance. Avec seulement 140 nœuds d'entraînement, Cora est très sensible au seed. Dans un projet sérieux, on fixerait 5 seeds différents et on reporterait moyenne ± écart-type. Enfin, le **fallback full-batch de GraphSAGE** (dû à l'absence de `torch-sparse` sur Mac) rend la comparaison des temps d'entraînement non représentative du comportement réel avec sampling.







*Aucun fichier volumineux n'a été commité dans ce dépôt. Les données sont téléchargées automatiquement par PyG dans `~/.cache/pyg_data` et les checkpoints sont exclus via `.gitignore`.*