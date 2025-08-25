

# 🎨 Text-To-Image Generator

A simple Text-to-Image Generator built using Hugging Face Diffusers and Streamlit.<br>
This application allows users to generate stunning AI images from text prompts with the Stable Diffusion model.

# 🚀 Features

🖼️ Generate high-quality images from text using Stable Diffusion <br>

⚡ Real-time interactive interface with Streamlit <br>

🎲 Randomized seeds for diverse image outputs <br>

💾 Option to view and save generated images <br>

 # 🛠️ Tech Stack

Python<br>

Streamlit <br>

PyTorch<br>

Diffusers (Hugging Face) <br>

PIL (Python Imaging Library)
 
# 📂 Project Structure
Text-To-Image-Generator/<br>
│── app.py              # Main Streamlit app  <br>
│── requirements.txt    # Dependencies  <br>
│── README.md           # Project Documentation   <br>

# ⚙️ Installation

1️⃣ Clone the repository<br> 

git clone https://github.com/subhadipsinha722133/Text-To-Image-Generator.git  <br>
cd Text-To-Image-Generator <br>


2️⃣ Create a virtual environment and activate it <br>

python -m venv venv <br>
source venv/bin/activate   # On Mac/Linux <br>
venv\Scripts\activate      # On Windows <br>


3️⃣ Install dependencies <br>

pip install -r requirements.txt <br>


4️⃣ Add your Hugging Face Access Token (required for Stable Diffusion) <br>

huggingface-cli login <br>

▶️ Usage <br>

Run the Streamlit app: <br>

streamlit run app.py <br>


Enter your text prompt, wait a few seconds, and watch your imagination turn into images! 🎨✨ <br>

📸 Example <br>

Prompt: "A futuristic cityscape at sunset with flying cars" <br>

<p align="center"> <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sd_text2img.png" width="600"> </p>

 # 📜 License

This project is licensed under the MIT License. <br>

# 🙌 Acknowledgements

Hugging Face <br>
 for Diffusers <br>

Stability AI <br>
 for Stable  <br>

Streamlit <br>
 for interactive apps
