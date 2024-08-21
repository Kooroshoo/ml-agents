# Unity ML Walking Agent 

![Initial](https://github.com/user-attachments/assets/cb34ef31-ee08-4f09-8b13-a63e024fa21f)



## ML-Agents Setup with Conda and Unity

Using Unity 2022.3 LTS

ML-Agents Release 21

### 1. Create Conda Environment
Create a new Conda environment with Python 3.9.18:

```bash
conda create -n mlagents python=3.9.18 && conda activate mlagents
```

### 2. Install PyTorch with CUDA 12.1
Install PyTorch 2.2.1 (or compatible version) with CUDA support:

```bash
pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install ML-Agents from Source
Install ML-Agents and ML-Agents Environments from the local source:

```bash
cd /path/to/ml-agents
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
```


### 4. Configure Unity for ML-Agents
- Open your Unity project.
- Go to Window > Package Manager.
- Click "+" > "Add package from disk..."
- Select **com.unity.ml-agents** and **com.unity.ml-agents.extensions** from your project root.


### 5. Verify Installation
```bash
mlagents-learn --help
```
If the help information appears, your setup is complete.

