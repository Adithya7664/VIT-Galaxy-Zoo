# Hi, I'm Adithya

CS undergrad at BITS Pilani Hyderabad, working towards AI/ML engineering. I build projects alone, end to end: data, model, evaluation, and a deployed demo where it makes sense. Each repo's README has the details and results, so this page is just the map.

Currently: ML modelling on an internship at InMobi, and a small transformer + reinforcement-learning project (tiny GPT from scratch, then Q-learning / SARSA on a grid and a Decision Transformer).

## Projects

### Machine learning
| Project | What it is | Links |
|---|---|---|
| **Galaxy Zoo: ViT + MAE** | Vision Transformer and masked autoencoder written from scratch in PyTorch, pre-trained without labels on Galaxy Zoo and evaluated with a linear probe against a supervised baseline. | [Repo](https://github.com/Adithya7664/VIT-Galaxy-Zoo) |
| **BARC: power-grid contingency GNN** | Simulates single-element outages on IEEE bus systems (pandapower), turns each into a graph, and trains an edge-conditioned GNN (PyTorch Geometric) to flag critical elements. Tested on a 118-bus system at a load not used in training. | [Repo](https://github.com/Adithya7664/ML-project) |
| **Exoplanet Habitability Explorer** | Earth Similarity Index on NASA Exoplanet Archive data, four models benchmarked, SHAP explanations, Streamlit app and a Three.js 3D planet renderer. | [Repo](https://github.com/Adithya7664/Exoplanet-Habitability-Website) · [Live app](https://exoplanet-habitability-website-sthvjpfpl9ucwqvpdedmdd.streamlit.app/) |
| **Semantic drone safe-zone detection** | Encoder-decoder segmentation with a ResNet backbone plus MiDaS depth, combined in a decision pipeline that outputs a safe drop zone as a bounding box. Built for a TU Graz course. | Repo link coming |

### Agents and reinforcement learning
| Project | What it is | Links |
|---|---|---|
| **2048 RL agent + in-game AI tutor** | Maskable PPO agent (Stable-Baselines3) for a Flutter 2048 game, with a FastAPI tutor that explains each suggested move and a "watch AI play" mode. NumPy-only inference keeps the API small. | [Repo](https://github.com/Adithya7664/2048_AI_Game) · [Play it](https://adithya7664.github.io/2048_AI_Game/) |
| **Wumpus World search agent** | A* planning with Bayesian inference over noisy stench and breeze readings, multi-objective scoring, replay viewer and in-browser live mode (Pyodide), tests and CI. | [Repo](https://github.com/Adithya7664/CS_F407_2026_0144_0227_1257_1362_1365) |
| **LangGraph shape-matching agent** | LangGraph pipeline (hole detection, ring matching, probe under a tool-call budget) for a spatial puzzle harness. 12 of 15 seeds solved; the write-up covers the failure mode. | [Repo](https://github.com/Adithya7664/spatial-distractor-challenge) (see `NOTES.md`) |

### Tools
| Project | What it is | Links |
|---|---|---|
| **PyQt5 photo editor** | Desktop editor with sliders, paint and erase, and a neural style-transfer panel. | [Repo](https://github.com/Adithya7664/PyQt5-Photo-editor) |

## Stack
**Languages:** Python, Java, C++, SQL, JavaScript
**ML:** PyTorch, PyTorch Geometric, scikit-learn, LightGBM, SHAP, Stable-Baselines3, Gymnasium
**Tools:** FastAPI, Docker, GitHub Actions, Streamlit, Flutter, Three.js, PyQt5, Git

## Find me
[LinkedIn](https://www.linkedin.com/in/meduri-adithya-326a68333)
