# Clone

Create personal access token
User > Settings >  Developer settings > Tokens (classic) > Generate Token
Tick access levels needed, generate token and save it.

- `git clone  https://<user name>@github.com/Ignitarium-AI/PyTorch-gender-age-Resnet18` \
- `git checkout -b <feature branch>`

## Create environment and install requirements

- `python3.8 -m venv va`
- `source va/bin/activate`
- `pip install -r requirements.txt`

### To train model

Configure parameters and dataset dictionary in "config.yaml".
Run "main.py" to train the network.

- `python main.py`

Run "export.py" to convert the .pt model into onnx format.

- `python export.py`

### To run inference on an image

- `python image_demo.py`

### To run inference on a live video

- `python video_demo.py`
