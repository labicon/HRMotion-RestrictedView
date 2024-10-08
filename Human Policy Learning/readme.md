# 🛤️ Trajectory Prediction with Diffusion Model

## 🔍 Overview
This project uses a 🌫️ diffusion-based machine learning model to predict safe trajectories for an agent 🤖 navigating through a complex environment. The model is trained using human-generated trajectory data and incorporates information about obstacles 🚧 to generate goal-oriented paths 🎯.

The project is divided into several components:
- **📦 Data Loading**: Prepares training and validation data, including obstacle information.
- **🧠 Model Training**: Trains a Class-Conditioned U-Net using a denoising diffusion probabilistic model (DDPM) approach.
- **🛤️ Trajectory Prediction**: Uses the trained model to predict trajectories and visualize results.

## 📂 Project Structure
- **`training_main.py`**: Script for training the model.
- **`prediction_main.py`**: Script for predicting trajectories using the trained model.
- **`data_loader.py`**: Utility functions for loading and normalizing data.
- **`model.py`**: Contains the definition of the Class-Conditioned U-Net model.
- **`training.py`**: Contains functions for training and validation.
- **`prediction.py`**: Utility functions for prediction, saving, and plotting.
- **`README.md`**: Project documentation.

## ⚙️ Installation
### 📋 Prerequisites
- 🐍 Python 3.7+
- 🔥 PyTorch
- [Diffusers](https://github.com/huggingface/diffusers) library

📥 Data Collection
To collect trajectory data, we developed a custom game environment. In this game, participants control a point-mass agent to navigate through a series of obstacles and reach a goal. The environment includes both static and dynamic obstacles, requiring players to make decisions that balance obstacle avoidance and goal achievement. The collected data includes:
Agent Trajectories: The paths taken by the agent, including position and orientation over time.
Obstacle Data: The positions and movements of obstacles in the environment.
This interactive approach allows us to gather realistic human decision-making data, which is then used to train our diffusion model for trajectory prediction.
You can try the game directly on the project website. If you need to collect your own data, you can download the game. Detailed instructions on how to play the game and locate the recorded data are provided within the game itself.
[🎮 Play Online](https://armlabstanford.github.io/HRMotion_RestrictedView)
[🎮 For Windows users](https://drive.google.com/drive/folders/14ypvx5AdhrXajyVRIjiBUe--D_0aLZb7?usp=sharing)
[🎮 For Mac users](https://drive.google.com/drive/folders/1a0fESG0gcEWz7aJbZoPYQMD9lzaZRVfz?usp=sharing)

## 🚀 Usage
### 🏋️‍♂️ Training the Model
Before training, you can download a pre-trained version of the model to get started quickly:

[⬇️ Download Pre-trained Model](https://drive.google.com/file/d/1PD49sdqR6KdXrwDjFw_OVpZ--cKLScV1/view?usp=sharing)

To train the model, use the `training_main.py`:
```sh
python training_main.py
```
This script will load the training data, train the model, and save the best model parameters.

### 🔮 Predicting Trajectories
To generate predicted trajectories using the trained model, use the `prediction_main.py`:
```sh
python prediction_main.py
```
This script will load the trained model, generate trajectories, save them to a text file, and produce a plot of the predicted trajectories.

### ⚙️ Parameters
- **initial_state**: The initial position `(x, y, theta)` of the agent 🤖.
- **goal**: The goal coordinates `(x, y)` for the agent 🎯. (Currently only support fixed goal location)
- **Obstacle Data**: Obstacle information is used from the training dataset.

## 📊 Results
- The generated trajectories are saved in `predicted_trajectories.txt`.
- A plot of the predicted trajectories is saved as `predicted_trajectories_plot.png`.

## 📝 Notes
- Ensure that you have the necessary training data in the `./data` directory before running the scripts.
- Modify the parameters in `prediction_main.py` to adapt to your use case (e.g., initial state, goal location).

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.

## 📧 Contact
For questions or suggestions, please put them in the issue section of the repository.

## 🚨 Important Notice
We are currently working on a second version of the game for data collection, which will be more reliable and provide higher-quality data. The current dataset is somewhat limited and contains a significant amount of low-quality data due to the challenging nature of the game. This has resulted in some model predictions missing the goal or failing to avoid obstacles in time. Additionally, the current model does not incorporate memory components like LSTMs or Transformers, which makes the predicted paths a bit noisy. We appreciate your understanding and welcome your feedback as we improve the system.