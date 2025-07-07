# AutoML-Agent
This is the official implementation of **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML** (ICML 2025) 
> [[Paper](https://arxiv.org/abs/2410.02958)][[Poster](/static/pdfs/poster.pdf)][[Website](https://deepauto-ai.github.io/automl-agent/)]

## Setup
### Benchmark Datasets
| **Data Modality**                  | **Downstream Task**        | **Dataset Name**                                                                                                                                    | **# Features** | **# Train** | **# Valid** | **# Test** | **# Classes** | **Source**                   | **License** | **Evaluation Metric** |
| ---------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ----------- | ----------- | ---------- | ------------- | ---------------------------: | ----------: | --------------------: |
| _Main Datasets_                    |                            |                                                                                                                                                     |                |             |             |            |               |                              |             |                       |
| Image (Computer Vision)            | Image Classification       | [Butterfly Image](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)                                                       | 224x224        | 4,549       | 1,299       | 651        | 75            | Kaggle Dataset               | CC0         | Accuracy              |
|                                    |                            | [Shopee-IET](https://www.kaggle.com/competitions/demo-shopee-iet-competition/data)                                                                  | Varying        | 640         | 160         | 80         | 4             | Kaggle Competition           | Custom      |                       |
| Text (Natural Language Processing) | Text Classification        | [Ecommerce Text](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)                                                      | N/A            | 35,296      | 10,084      | 5,044      | 4             | Kaggle Dataset               | CC BY 4.0   | Accuracy              |
|                                    |                            | [Textual Entailment](https://github.com/guosyjlu/DS-Agent)                                                                                          | N/A            | 3,925       | 982         | 4,908      | 3             | Kaggle Dataset               | N/A         |                       |
| Tabular (Classic Machine Learning) | Tabular Classification     | [Banana Quality](https://www.kaggle.com/datasets/l3llff/banana/data)                                                                                | 7              | 5,600       | 1,600       | 800        | 2             | Kaggle Dataset               | Apache 2.0  | F1                    |
|                                    |                            | [Software Defects](https://github.com/guosyjlu/DS-Agent)                                                                                            | 21             | 73,268      | 18,318      | 91,587     | 2             | Kaggle Competition           | N/A         |                       |
|                                    | Tabular Clustering         | [Smoker Status](https://github.com/guosyjlu/DS-Agent)                                                                                               | 22             | 100,331     | 28,666      | 14,334     | 2             | Kaggle Competition           | N/A         | RI                    |
|                                    |                            | [Higher Education Students Performance](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation)                   | 31             | 101         | 29          | 15         | 8             | Research Dataset (UCI ML)    | CC BY 4.0   | RI                    |
|                                    | Tabular Regression         | [Crab Age](https://github.com/guosyjlu/DS-Agent)                                                                                                    | 8              | 53,316      | 13,329      | 66,646     | N/A           | Kaggle Competition           | CC0         | RMSLE                 |
|                                    |                            | [Crop Price](https://www.kaggle.com/datasets/varshitanalluri/crop-price-prediction-dataset)                                                         | 8              | 1,540       | 440         | 220        | N/A           | Kaggle Dataset               | MIT         | RMSLE                 |
| Graph (Graph Learning)             | Node Classification        | [Cora](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid)     | 1,433          | 2,708       | 2,708       | 2,708      | 7             | Research Dataset (Planetoid) | CC BY 4.0   | Accuracy              |
|                                    |                            | [Citeseer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid) | 3,703          | 3,327       | 3,327       | 3,327      | 6             | Research Dataset (Planetoid) | N/A         |                       |
| Time Series (Time Series Analysis) | Time-Series Forecasting    | [Weather](https://github.com/thuml/Time-Series-Library)                                                                                             | 21             | 36,887      | 10,539      | 5,270      | N/A           | Research Dataset (TSLib)     | CC BY 4.0   | RMSLE                 |
|                                    |                            | [Electricity](https://github.com/thuml/Time-Series-Library)                                                                                         | 321            | 18,412      | 5,260       | 2,632      | N/A           | Research Dataset (TSLib)     | CC BY 4.0   |                       |
| _Additional Datasets for SELA_     |                            |                                                                                                                                                     |                |             |             |            |               |                              |             |                       |
| Tabular (Classic Machine Learning) | Binary Classification      | [Smoker Status](https://github.com/geekan/MetaGPT/tree/main/metagpt/ext/sela)                                                                       | 22             | 85997       | 21500       | 143331     | 2             | Kaggle Competition           | N/A         | F1                    |
|                                    |                            | [Click Prediction Small](https://github.com/geekan/MetaGPT/tree/main/metagpt/ext/sela)                                                              | 11             | 19174       | 4794        | 7990       | 2             | OpenML                       |             |                       |
|                                    | Multi-Class Classification | [MFeat Factors](https://github.com/geekan/MetaGPT/tree/main/metagpt/ext/sela)                                                                       | 216            | 960         | 240         | 400        | 10            | OpenML                       |             |                       |
|                                    |                            | [Wine Quality White](https://github.com/geekan/MetaGPT/tree/main/metagpt/ext/sela)                                                                  | 11             | 2350        | 588         | 980        | 7             | OpenML                       |             |                       |
|                                    | Regression                 | [Colleges](https://github.com/geekan/MetaGPT/tree/main/metagpt/ext/sela)                                                                            | 44             | 3389        | 848         | 1413       | N/A           | OpenML                       |             | RMSE                  |
|                                    |                            | [House Prices](https://github.com/geekan/MetaGPT/tree/main/metagpt/ext/sela)                                                                        | 80             | 700         | 176         | 292        | N/A           | Kaggle Competition           |             |                       |

## Usage
We recommend using conda environment.
```bash
conda create --name amla python=3.11
pip install -r requirements.txt
```

### Run AutoML Development
1. Run the instruction-tuned LoRA adapter ([Download Link](https://www.dropbox.com/scl/fi/9mjm772d99xcr5e0905cx/adapter-mixtral.zip?rlkey=amoq17jhp3ye3sswpqgmzsgoo&st=woamltp6&dl=0)) for Prompt Agent via vLLM. `vllm==0.4.1` is strictly required to get correct parsed results.
> **Update:** We have implemented an alternative OpenAI version via `parse_openai(..)`
```bash
HF_TOKEN="Your HuggingFace Token" CUDA_VISIBLE_DEVICES="0,1,2,3" python -m vllm.entrypoints.openai.api_server --model mistralai/Mixtral-8x7B-Instruct-v0.1 --enable-lora --lora-modules prompt-llama=./adapter/adapter-mixtral/ --tensor-parallel-size 4
```
2. Setup Prompt Agent and LLM backbone(s) in `./configs.py`.
```python
AVAILABLE_LLMs = {
    "prompt-llm": {
        "api_key": "empty",
        "model": "prompt-llama",
        "base_url": "http://localhost:8000/v1",
    },
    "gpt-4": {"api_key": "YOUR OPENAI KEY", "model": "gpt-4o"},
    "gpt-3.5": {"api_key": "YOUR OPENAI KEY", "model": "gpt-3.5-turbo"},
}
```

3. Run chat with AutoML-Agent's Manager 🕴🏻!
```python
from agent_manager import AgentManager

data_path = "agent_workspace/datasets/banana_quality.csv" # assuming the data is uploaded via web interface / API
user_prompt = "Build a model to classify banana quality as good or bad based on their numerical information about bananas of different quality (size, weight, sweetness, softness, harvest time, ripeness, and acidity). We have uploaded the entire dataset for you here in the banana_quality.csv file."
manager = AgentManager(llm='gpt-4', interactive=False, data_path=data_path)

manager.initiate_chat(user_prompt)
```
Running in a Jupyter notebook is recommended. The generated output .py file will be in the `agent_workspace`.

## Citation
```bibtex
@inproceedings{AutoML_Agent,
  title={Auto{ML}-Agent: A Multi-Agent {LLM} Framework for Full-Pipeline Auto{ML}},
  author={Trirat, Patara and Jeong, Wonyong and Hwang, Sung Ju},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=p1UBWkOvZm}
}
```

## License
This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.
Commercial use is prohibited.
