[200~mkdir -p text2img/{app_frontend/static,service,training} && touch text2img/{requirements.txt,lightning_app.py,Dockerfile,README.md} && touch text2img/app_frontend/streamlit_app.py && touch text2img/service/{model.py,litserver_app.py,utils.py} && touch text2img/training/train.py~
cd
ls
mkdir
mkdir --help
[200~mkdir -p text2img/{app_frontend/static,service,training} && touch text2img/{requirements.txt,lightning_app.py,Dockerfile,README.md} && touch text2img/app_frontend/streamlit_app.py && touch text2img/service/{model.py,litserver_app.py,utils.py} && touch text2img/training/train.py
mkdir -p text2img/{app_frontend/static,service,training} && touch text2img/{requirements.txt,lightning_app.py,Dockerfile,README.md} && touch text2img/app_frontend/streamlit_app.py && touch text2img/service/{model.py,litserver_app.py,utils.py} && touch text2img/training/train.py
pip install -r requirements.txt
ls
cd text2img
ls
pip install -r requirements.txt
cd ..
mkdir -p text-to-image-generator
cd text-to-image-generator
mkdir -p src/models src/pipeline src/utils
touch src/__init__.py
touch src/models/__init__.py
touch src/models/attention.py
touch src/models/unet.py
touch src/models/evaluator.py
touch src/pipeline/__init__.py
touch src/pipeline/generator.py
touch src/pipeline/pipeline.py
touch src/utils/__init__.py
touch src/utils/image_utils.py
touch src/utils/config.py
touch app.py
touch requirements.txt
touch README.md
touch .gitignore
mkdir -p outputs cache
mkdir -p config
touch config/model_config.yaml
   # Install PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd text-to-image-generator/
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   mkdir -p outputs cache
python -c "
from diffusers import StableDiffusionPipeline
import torch

print('Downloading model... This will take 5-10 minutes.')
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16,
    safety_checker=None
)
print('Model cached successfully!')
"
python -c "from src.pipeline import GenerativeAIPipeline"
python app.py
chmod +x lightning_setup.sh
nvidia-smi
python app.py
pip install --upgrade numpy scikit-learn transformers diffusers
pip install --upgrade numpy scikit-learn transformers diffusers
pip uninstall -y numpy scipy scikit-learn transformers diffusers torch torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python app.py
python -c "from src.models import CrossAttentionLayer; print('✅ Models OK')"
python -c "from src.pipeline import GenerativeAIPipeline; print('✅ Pipeline OK')"
python -c "from src.utils import Config; print('✅ Utils OK')"
python app.py
cd text-to-image-generator/
cd text-to-image-generator/
lsof -i :8080
python app.py
lsof -i :8080
netstat -tuln | grep 7860
python app.py
